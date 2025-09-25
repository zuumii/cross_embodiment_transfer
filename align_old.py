# align.py  —— 替换整个文件

import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from td3 import Actor

# === 新增：MuJoCo / robosuite 的 SEW 可微工具 ===
import mujoco

def _normalize_torch(v, eps=1e-9):
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def sigma_from_SEW_torch(S, E, W, ref_dir=(0.0, 1.0, 0.0)):
    """
    纯 Torch 几何：S,E,W (...,3) -> stereographic SEW 角 sigma (tan(theta/2))
    返回 (sigma, theta)
    """
    u = _normalize_torch(W - S)  # SW 方向单位向量
    I = torch.eye(3, device=S.device, dtype=S.dtype).view(1, 3, 3).expand(S.shape[:-1] + (3, 3))
    P = I - u.unsqueeze(-1) @ u.unsqueeze(-2)  # 投影到垂直于 u 的平面
    v = E - S
    v_perp = (P @ v.unsqueeze(-1)).squeeze(-1)
    v_hat = _normalize_torch(v_perp)

    r = torch.tensor(ref_dir, device=S.device, dtype=S.dtype).view(1, 3).expand_as(S)
    e1 = (P @ r.unsqueeze(-1)).squeeze(-1)
    tiny = (e1.norm(dim=-1, keepdim=True) < 1e-6)
    alt = torch.tensor([1.0, 0.0, 0.0], device=S.device, dtype=S.dtype).view(1, 3).expand_as(S)
    e1 = torch.where(tiny, (P @ alt.unsqueeze(-1)).squeeze(-1), e1)
    e1 = _normalize_torch(e1)
    e2 = _normalize_torch(torch.cross(u, e1, dim=-1))

    x = (v_hat * e1).sum(-1)
    y = (v_hat * e2).sum(-1)
    theta = torch.atan2(y, x)          # (-pi, pi]
    sigma = torch.tan(0.5 * theta)     # stereographic，数值稳定
    return sigma, theta

# ------- MuJoCo unwrap helpers (robosuite / raw mujoco / other wrappers) -------
def _unwrap_mj(sim_or_env):
    """
    统一拿原生 mujoco.MjModel / MjData
    兼容：
      - robosuite env.sim（带 .model/.data，且 model/data 可能再包一层 _model/_data）
      - 直接给 sim 的情况
    """
    # 如果是 env，就取 env.sim；否则当作 sim 用
    sim = getattr(sim_or_env, "sim", sim_or_env)

    # 取 model / data
    model = getattr(sim, "model", None)
    data  = getattr(sim, "data", None)

    # robosuite 包了一层 _model / _data
    if model is not None:
        model = getattr(model, "_model", model)
    if data is not None:
        data = getattr(data, "_data", data)

    if model is None or data is None:
        raise RuntimeError("Cannot unwrap mujoco model/data from the given sim/env object.")
    return model, data

def _mj_forward(sim_or_env):
    import mujoco
    m, d = _unwrap_mj(sim_or_env)
    mujoco.mj_forward(m, d)

def joint_point_world(sim_or_env, joint_id):
    import numpy as np
    m, d = _unwrap_mj(sim_or_env)
    jid = int(joint_id)
    bid = int(m.jnt_bodyid[jid])
    p_local = np.array(m.jnt_pos[jid], dtype=np.float64).reshape(3,)
    R = d.xmat[bid].reshape(3, 3)
    t = d.xpos[bid]
    return t + R @ p_local

def analytic_point_jac(sim_or_env, joint_id, arm_vel_cols):
    import numpy as np, mujoco
    m, d = _unwrap_mj(sim_or_env)
    bid = int(m.jnt_bodyid[joint_id])
    p_world = joint_point_world(sim_or_env, joint_id).astype(np.float64)
    Jp = np.zeros((3, m.nv), dtype=np.float64)
    Jr = np.zeros((3, m.nv), dtype=np.float64)
    mujoco.mj_jac(m, d, Jp, Jr, p_world, bid)
    return Jp[:, np.array(arm_vel_cols, dtype=int)]

def _joint_list_for_arm(env, robot):
    # 按 qpos 顺序列出 arm 关节
    m = env.sim.model
    idx_raw = robot._ref_joint_pos_indexes
    arm_qpos_set = set(int(i) for i in np.array(idx_raw).ravel())
    items = []
    for jid in range(m.njnt):
        qadr = int(m.jnt_qposadr[jid])
        if qadr in arm_qpos_set:
            items.append({
                "name": m.joint_id2name(jid),
                "jid": jid,
                "qpos_adr": qadr,
                "body_id": int(m.jnt_bodyid[jid]),
            })
    items.sort(key=lambda it: it["qpos_adr"])
    return items

def _pick_sew_indices(robot_name, arm_joints):
    # 简单稳定的默认：Panda/Sawyer 等 7DoF -> 1,2,3；6DoF -> 1,2,4
    if "panda" in robot_name.lower():
        return 1, 2, 3
    dof = len(arm_joints)
    if dof >= 7:
        return 1, 2, 3
    if dof == 6:
        return 1, 2, 4
    mid = max(2, dof // 2)
    return 1, mid, max(dof - 2, 1)

class _SEWFromQ_MJ(torch.autograd.Function):
    """
    自定义 autograd：输入 q (torch)，内部用 MuJoCo 拿 Jp，
    几何端用 Torch 拿 ∂σ/∂S,E,W，链回得到 ∂σ/∂q。
    """
    @staticmethod
    def forward(ctx, q, sim_or_env, arm_qpos_addr, arm_vel_cols, sew_jids, ref_dir=(0.0, 1.0, 0.0)):
        device, dtype = q.device, q.dtype
        q_np = q.detach().cpu().numpy().astype(np.float64)

        # ---- 用原生 m,d 写 qpos 并前向 ----
        m, d = _unwrap_mj(sim_or_env)
        d.qpos[np.array(arm_qpos_addr, dtype=int)] = q_np
        import mujoco
        mujoco.mj_forward(m, d)

        # ---- 三个关键点世界坐标 ----
        jid_S, jid_E, jid_W = sew_jids
        S_np = joint_point_world(sim_or_env, jid_S)
        E_np = joint_point_world(sim_or_env, jid_E)
        W_np = joint_point_world(sim_or_env, jid_W)

        with torch.enable_grad():
            S = torch.tensor(S_np, device=device, dtype=dtype, requires_grad=True)
            E = torch.tensor(E_np, device=device, dtype=dtype, requires_grad=True)
            W = torch.tensor(W_np, device=device, dtype=dtype, requires_grad=True)
            sigma, _ = sigma_from_SEW_torch(S.view(1, 3), E.view(1, 3), W.view(1, 3), ref_dir=ref_dir)
            sigma = sigma.squeeze(0)
            gS, gE, gW = torch.autograd.grad(sigma, [S, E, W], retain_graph=False, create_graph=False)

        # ---- 解析雅可比（切 arm 列）----
        J_S = torch.from_numpy(analytic_point_jac(sim_or_env, jid_S, arm_vel_cols)).to(device=device, dtype=dtype)
        J_E = torch.from_numpy(analytic_point_jac(sim_or_env, jid_E, arm_vel_cols)).to(device=device, dtype=dtype)
        J_W = torch.from_numpy(analytic_point_jac(sim_or_env, jid_W, arm_vel_cols)).to(device=device, dtype=dtype)

        ctx.save_for_backward(gS.detach(), gE.detach(), gW.detach(), J_S, J_E, J_W)
        return sigma.view(1)

    @staticmethod
    def backward(ctx, grad_out):
        gS, gE, gW, J_S, J_E, J_W = ctx.saved_tensors
        dσ_dq = (gS.view(1, 3) @ J_S + gE.view(1, 3) @ J_E + gW.view(1, 3) @ J_W).squeeze(0)
        return grad_out.view(1) * dσ_dq, None, None, None, None, None

def sigma_from_q_once(sim_or_env, q_tensor, arm_qpos_addr, arm_vel_cols, sew_jids, ref_dir=(0.0,1.0,0.0)):
    return _SEWFromQ_MJ.apply(q_tensor, sim_or_env, arm_qpos_addr, arm_vel_cols, sew_jids, ref_dir)


# ========================= 原有类（尽量不动结构） =========================
class Agent:
    """Base class for adaptation"""
    def __init__(self, obs_dims, act_dims, device):
        self.obs_dim = obs_dims['obs_dim']
        self.robot_obs_dim = obs_dims['robot_obs_dim']
        self.obj_obs_dim = obs_dims['obj_obs_dim']
        self.act_dim = act_dims['act_dim']
        self.device = device
        self.batch_norm = False
        self.modules = []

        assert self.obs_dim == self.robot_obs_dim + self.obj_obs_dim

    def eval_mode(self):
        for m in self.modules:
            m.eval()

    def train_mode(self):
        for m in self.modules:
            m.train()

    def freeze(self):
        for m in self.modules:
            for p in m.parameters():
                p.requires_grad = False


class ObsAgent(Agent):
    def __init__(self, obs_dims, act_dims, device, n_layers=3, hidden_dim=256):
        super().__init__(obs_dims, act_dims, device)
        self.lat_obs_dim = obs_dims['lat_obs_dim']

        self.obs_enc = utils.build_mlp(self.robot_obs_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(self.device)
        self.obs_dec = utils.build_mlp(self.lat_obs_dim, self.robot_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(self.device)
        self.inv_dyn = utils.build_mlp(self.lat_obs_dim*2, self.act_dim-1, n_layers, hidden_dim, 
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(self.device)
        self.fwd_dyn = utils.build_mlp(self.lat_obs_dim+self.act_dim-1, self.lat_obs_dim, n_layers, hidden_dim, 
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(self.device)
        self.actor = Actor(self.lat_obs_dim+self.obj_obs_dim, self.act_dim, n_layers, hidden_dim).to(self.device)

        self.modules = [self.obs_enc, self.obs_dec, self.inv_dyn, self.fwd_dyn, self.actor]

    def save(self, model_dir):
        torch.save(self.actor.state_dict(), f'{model_dir}/actor.pt')
        torch.save(self.obs_enc.state_dict(), f'{model_dir}/obs_enc.pt')        
        torch.save(self.obs_dec.state_dict(), f'{model_dir}/obs_dec.pt')
        torch.save(self.inv_dyn.state_dict(), f'{model_dir}/inv_dyn.pt')
        torch.save(self.fwd_dyn.state_dict(), f'{model_dir}/fwd_dyn.pt')

    def load(self, model_dir):
        self.obs_enc.load_state_dict(torch.load(model_dir/'obs_enc.pt'))
        self.obs_dec.load_state_dict(torch.load(model_dir/'obs_dec.pt'))
        self.fwd_dyn.load_state_dict(torch.load(model_dir/'fwd_dyn.pt'))
        self.inv_dyn.load_state_dict(torch.load(model_dir/'inv_dyn.pt'))
        self.actor.load_state_dict(torch.load(model_dir/'actor.pt'))

    def load_actor(self, model_dir):
        self.actor.load_state_dict(torch.load(model_dir/'actor.pt'))
        for p in self.actor.parameters():
            p.requires_grad = False

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            obs = obs.unsqueeze(0)
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))

        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, self.expl_noise, size=act.shape[0])
            act = np.clip(act, -1, 1)
        return act  


class ObsActAgent(ObsAgent):
    def __init__(self, obs_dims, act_dims, device, n_layers=3, hidden_dim=256):
        super().__init__(obs_dims, act_dims, device, n_layers=n_layers, hidden_dim=hidden_dim)

        self.lat_act_dim = act_dims['lat_act_dim']

        self.act_enc = utils.build_mlp(self.robot_obs_dim+self.act_dim-1, self.lat_act_dim, n_layers, 
            hidden_dim, activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.act_dec = utils.build_mlp(self.robot_obs_dim+self.lat_act_dim, self.act_dim-1, n_layers,
            hidden_dim, activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.inv_dyn = utils.build_mlp(self.lat_obs_dim*2, self.lat_act_dim, n_layers, hidden_dim, 
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.fwd_dyn = utils.build_mlp(self.lat_obs_dim+self.lat_act_dim, self.lat_obs_dim, 
            n_layers, hidden_dim, activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.actor = Actor(self.lat_obs_dim+self.obj_obs_dim, self.lat_act_dim+1, n_layers, hidden_dim).to(device)

        self.modules += [self.act_enc, self.act_dec]

    def save(self, model_dir):
        super().save(model_dir)
        torch.save(self.act_enc.state_dict(), f'{model_dir}/act_enc.pt')        
        torch.save(self.act_dec.state_dict(), f'{model_dir}/act_dec.pt')

    def load(self, model_dir):
        super().load(model_dir)
        self.act_enc.load_state_dict(torch.load(model_dir/'act_enc.pt'))
        self.act_dec.load_state_dict(torch.load(model_dir/'act_dec.pt'))

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            obs = obs.unsqueeze(0)
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            lat_act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
            lat_act, gripper_act = lat_act[:, :-1], lat_act[:, -1].reshape(-1, 1)
            act = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
            act = torch.cat([act, gripper_act], dim=-1)
        
        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, self.expl_noise, size=act.shape[0])
            act = np.clip(act, -1, 1)

        return act


class ObsAligner:
    def __init__(
        self, 
        src_agent, 
        tgt_agent, 
        device, 
        n_layers=3, 
        hidden_dim=256,
        lr=3e-4,
        lmbd_gp=10,
        log_freq=1000,
        # === 新增（可选）：不给也能跑 ===
        src_env=None,
        tgt_env=None,
        lmbd_sew=0.0,              # 默认 0，不改变原行为；想打开 SEW 对齐就传 >0
        ref_dir=(0.0, 1.0, 0.0),
        src_qpos_slice=None,       # 例如 [start, length]；不填则默认“前 dof 个”
        tgt_qpos_slice=None,
    ):
        
        self.device = device
        self.lmbd_gp = lmbd_gp
        self.lmbd_cyc = 10
        self.lmbd_dyn = 10
        self.log_freq = log_freq

        self.src_obs_enc = src_agent.obs_enc
        self.src_obs_dec = src_agent.obs_dec
        self.tgt_obs_enc = tgt_agent.obs_enc
        self.tgt_obs_dec = tgt_agent.obs_dec
        self.fwd_dyn = src_agent.fwd_dyn
        self.inv_dyn = src_agent.inv_dyn

        assert src_agent.lat_obs_dim == tgt_agent.lat_obs_dim
        self.lat_obs_dim = src_agent.lat_obs_dim
        self.src_obs_dim = src_agent.robot_obs_dim
        self.tgt_obs_dim = tgt_agent.robot_obs_dim

        self.lat_disc = utils.build_mlp(self.lat_obs_dim, 1, n_layers, hidden_dim,
            activation='leaky_relu', output_activation='identity').to(self.device)
        self.src_disc = utils.build_mlp(self.src_obs_dim, 1, n_layers, hidden_dim,
            activation='leaky_relu', output_activation='identity').to(self.device)
        self.tgt_disc = utils.build_mlp(self.tgt_obs_dim, 1, n_layers, hidden_dim,
            activation='leaky_relu', output_activation='identity').to(self.device)

        # Optimizers
        self.tgt_obs_enc_opt = torch.optim.Adam(self.tgt_obs_enc.parameters(), lr=lr)
        self.tgt_obs_dec_opt = torch.optim.Adam(self.tgt_obs_dec.parameters(), lr=lr)
        self.lat_disc_opt = torch.optim.Adam(self.lat_disc.parameters(), lr=lr)
        self.src_disc_opt = torch.optim.Adam(self.src_disc.parameters(), lr=lr)
        self.tgt_disc_opt = torch.optim.Adam(self.tgt_disc.parameters(), lr=lr)

        # === 新增：SEW 相关配置（可选） ===
        self.lmbd_sew = float(lmbd_sew)
        self.ref_dir = tuple(ref_dir)
        self.src_sew = None
        self.tgt_sew = None
        self.src_qslice = src_qpos_slice
        self.tgt_qslice = tgt_qpos_slice
        if (src_env is not None) and (tgt_env is not None) and (self.lmbd_sew > 0.0):
            self._init_sew(src_env, tgt_env)

    def _init_sew(self, src_env, tgt_env):
        # 构建每个域的 SEW 计算配置：sim、dof、qpos/vel 索引、SEW 三点关节 id
        def _pack(env):
            robot = env.robots[0]
            dof = len(np.array(robot._ref_joint_pos_indexes).ravel())
            arm_qpos_addr = [int(i) for i in np.array(robot._ref_joint_pos_indexes).ravel().tolist()]
            arm_vel_cols = [int(i) for i in np.array(robot._ref_joint_vel_indexes).ravel().tolist()]
            joints = _joint_list_for_arm(env, robot)
            s_i, e_i, w_i = _pick_sew_indices(robot.robot_model.naming_prefix if hasattr(robot, 'robot_model') else "", joints)
            return {
                "sim_or_env": env,
                "dof": dof,
                "arm_qpos_addr": arm_qpos_addr,
                "arm_vel_cols": arm_vel_cols,
                "sew_jids": (joints[s_i]["jid"], joints[e_i]["jid"], joints[w_i]["jid"]),
            }
        self.src_sew = _pack(src_env)
        self.tgt_sew = _pack(tgt_env)

    def _extract_q(self, robot_obs_tensor, which="src"):
        """
        从 robot_obs 中抽取关节角 q（弧度）。
        - 若提供了 qpos_slice=[start, length] 就按切片取（假定就是角度）
        - 否则：
            * 若维度 >= 2*dof，按 [cos(0:dof), sin(dof:2*dof)] 还原 q=atan2(sin, cos)
            * 否则取前 dof 个元素为 q
        """
        if which == "src":
            dof = self.src_sew["dof"] if self.src_sew else self.src_obs_dim
            slc = self.src_qslice
        else:
            dof = self.tgt_sew["dof"] if self.tgt_sew else self.tgt_obs_dim
            slc = self.tgt_qslice

        if slc is not None:
            start, length = int(slc[0]), int(slc[1])
            return robot_obs_tensor[:, start:start+length]

        D = robot_obs_tensor.shape[1]
        if D >= 2 * dof:
            cos = robot_obs_tensor[:, :dof]
            sin = robot_obs_tensor[:, dof:2*dof]
            return torch.atan2(sin, cos)
        else:
            return robot_obs_tensor[:, :dof]

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = torch.rand((real_samples.size(0), 1)).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(real_samples.shape[0], 1, requires_grad=False, device=self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_disc(self, src_obs, src_act, tgt_obs, tgt_act, L, step):
        """
        Discriminator tries to separate source and target latent states and actions
        """
        
        fake_lat_obs = self.tgt_obs_enc(tgt_obs).detach()
        real_lat_obs = self.src_obs_enc(src_obs).detach()
        lat_disc_loss = self.lat_disc(fake_lat_obs).mean() - self.lat_disc(real_lat_obs).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs).detach()
        src_disc_loss = self.src_disc(fake_src_obs).mean() - self.src_disc(src_obs).mean()
        
        real_tgt_obs = self.tgt_obs_dec(real_lat_obs).detach()
        tgt_disc_loss = self.tgt_disc(real_tgt_obs).mean() - self.tgt_disc(tgt_obs).mean()

        lat_gp = self.compute_gradient_penalty(self.lat_disc, real_lat_obs, fake_lat_obs)
        src_gp = self.compute_gradient_penalty(self.src_disc, src_obs, fake_src_obs)
        tgt_gp = self.compute_gradient_penalty(self.tgt_disc, tgt_obs, real_tgt_obs)

        disc_loss = lat_disc_loss + src_disc_loss + tgt_disc_loss + \
            self.lmbd_gp * (lat_gp + src_gp + tgt_gp)

        self.lat_disc_opt.zero_grad()
        self.src_disc_opt.zero_grad()
        self.tgt_disc_opt.zero_grad()
        disc_loss.backward()
        self.lat_disc_opt.step()
        self.src_disc_opt.step()
        self.tgt_disc_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_disc/lat_disc_loss', lat_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/lat_gp', lat_gp.item(), step)
            L.add_scalar('train_lat_disc/src_disc_loss', src_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/src_gp', src_gp.item(), step)
            L.add_scalar('train_lat_disc/tgt_disc_loss', tgt_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/tgt_gp', tgt_gp.item(), step)
            L.add_scalar('train_lat_disc/real_lat_obs_sq', (real_lat_obs**2).mean().item(), step)
            L.add_scalar('train_lat_disc/fake_lat_obs_sq', (fake_lat_obs**2).mean().item(), step)

    def _sigma_batch(self, which_env, Q):
        """
        逐样本计算 sigma（支持反传）。
        which_env: "src" 或 "tgt"
        Q: [B, dof] torch（可带梯度）
        返回: [B] torch
        """
        assert which_env in ("src", "tgt")
        cfg = self.src_sew if which_env == "src" else self.tgt_sew

        # 兼容：cfg 里可能存的是 env 或 sim；sigma_from_q_once 内部会统一拆成 (m,d)
        sim_or_env   = cfg.get("sim_or_env", cfg.get("env", cfg.get("sim")))
        arm_qpos_addr = cfg["arm_qpos_addr"]
        arm_vel_cols  = cfg["arm_vel_cols"]
        sew_jids      = cfg["sew_jids"]

        sigmas = []
        # 注意：MuJoCo sim 是状态机，不支持并行，这里逐条调用
        for i in range(Q.shape[0]):
            q_i = Q[i]
            # 用 double 稳定数值（计算图仍然连通，梯度会经由 dtype cast 回传到 Q）
            q_i64 = q_i.to(torch.float64)
            sigma_i = sigma_from_q_once(
                sim_or_env, q_i64, arm_qpos_addr, arm_vel_cols, sew_jids,
                ref_dir=self.ref_dir
            )
            # 回到网络 dtype，保持一致
            sigmas.append(sigma_i.to(dtype=Q.dtype).squeeze(0))

        return torch.stack(sigmas, dim=0)

    def update_gen(self, src_obs, src_act, src_next_obs, tgt_obs, tgt_act, tgt_next_obs, L, step):
        """
        Generator outputs more realistic latent states from target samples
        """

        # Generator
        fake_lat_obs = self.tgt_obs_enc(tgt_obs)
        lat_gen_loss = -self.lat_disc(fake_lat_obs).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs)
        src_gen_loss = -self.src_disc(fake_src_obs).mean()

        real_tgt_obs = self.tgt_obs_dec(self.src_obs_enc(src_obs))
        tgt_gen_loss = -self.tgt_disc(real_tgt_obs).mean()

        gen_loss = lat_gen_loss + src_gen_loss + tgt_gen_loss

        # Cycle consistency
        pred_src_obs = self.src_obs_dec(self.tgt_obs_enc(real_tgt_obs))
        pred_tgt_obs = self.tgt_obs_dec(self.src_obs_enc(fake_src_obs))
        cycle_loss = F.l1_loss(pred_src_obs, src_obs) + F.l1_loss(pred_tgt_obs, tgt_obs)

        # Latent dynamics
        fake_lat_next_obs = self.tgt_obs_enc(tgt_next_obs)
        pred_act = self.inv_dyn(torch.cat([fake_lat_obs, fake_lat_next_obs], dim=-1))
        inv_loss = F.mse_loss(pred_act, tgt_act)
        pred_lat_next_obs = self.fwd_dyn(torch.cat([fake_lat_obs, tgt_act], dim=-1))
        fwd_loss = F.mse_loss(pred_lat_next_obs, fake_lat_next_obs)

        # === 新增：SEW 对齐损失 ===
        sew_loss = torch.tensor(0.0, device=self.device)
        if (self.lmbd_sew > 0.0) and (self.src_sew is not None) and (self.tgt_sew is not None):
            # Cross：σ( D_tgt( E_src(src) ) ) ≈ σ( src )
            q_src        = self._extract_q(src_obs, which="src")             # [B, dof_src]
            q_tgt_from_s = self._extract_q(real_tgt_obs, which="tgt")        # [B, dof_tgt]
            sigma_src        = self._sigma_batch("src", q_src)
            sigma_tgt_from_s = self._sigma_batch("tgt", q_tgt_from_s)
            L_sew_cross = F.mse_loss(sigma_tgt_from_s, sigma_src)

            # Cycle（目标域）：σ( pred_tgt_obs ) ≈ σ( tgt_obs )
            q_tgt      = self._extract_q(tgt_obs, which="tgt")
            q_tgt_pred = self._extract_q(pred_tgt_obs, which="tgt")
            sigma_tgt      = self._sigma_batch("tgt", q_tgt)
            sigma_tgt_pred = self._sigma_batch("tgt", q_tgt_pred)
            L_sew_self = F.mse_loss(sigma_tgt_pred, sigma_tgt)

            sew_loss = L_sew_cross + L_sew_self

        loss = gen_loss + self.lmbd_cyc * cycle_loss + self.lmbd_dyn * (inv_loss + fwd_loss) + self.lmbd_sew * sew_loss

        self.tgt_obs_enc_opt.zero_grad()
        self.tgt_obs_dec_opt.zero_grad()
        loss.backward()
        self.tgt_obs_enc_opt.step()
        self.tgt_obs_dec_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_gen/lat_gen_loss', lat_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/src_gen_loss', src_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/tgt_gen_loss', tgt_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/inv_loss', inv_loss.item(), step)
            L.add_scalar('train_lat_gen/fwd_loss', fwd_loss.item(), step)
            L.add_scalar('train_lat_gen/cycle_loss', cycle_loss.item(), step)
            if self.lmbd_sew > 0.0:
                L.add_scalar('train_lat_gen/sew_loss', sew_loss.item(), step)
                L.add_scalar('train_lat_gen/sew_cross', L_sew_cross.item(), step)
                L.add_scalar('train_lat_gen/sew_self', L_sew_self.item(), step)

            L.add_scalar('train_lat_gen/lat_obs_diff', 
                F.l1_loss(self.src_obs_enc(tgt_obs), fake_lat_obs), step)
            src_lat_obs = self.src_obs_enc(src_obs)
            src_lat_next_obs = self.src_obs_enc(src_next_obs)
            pred_src_lat_next_obs = self.fwd_dyn(torch.cat([src_lat_obs, src_act], dim=-1))
            pred_src_act = self.inv_dyn(torch.cat([src_lat_obs, src_lat_next_obs], dim=-1))
            L.add_scalar('train_lat_gen/src_fwd_loss', 
                F.mse_loss(src_lat_next_obs, pred_src_lat_next_obs).item(), step)
            L.add_scalar('train_lat_gen/src_inv_loss',
                F.mse_loss(src_act, pred_src_act).item(), step)


class ObsActAligner(ObsAligner):
    def __init__(
        self, 
        src_agent, 
        tgt_agent, 
        device, 
        n_layers=3, 
        hidden_dim=256,
        lr=3e-4,
        lmbd_gp=10,
        log_freq=1000,
        # === 新增参数也透传（可选）===
        src_env=None,
        tgt_env=None,
        lmbd_sew=0.0,
        ref_dir=(0.0, 1.0, 0.0),
        src_qpos_slice=None,
        tgt_qpos_slice=None,
    ):
        super().__init__(src_agent, tgt_agent, device, n_layers=n_layers, 
            hidden_dim=hidden_dim, lr=lr, lmbd_gp=lmbd_gp, log_freq=log_freq,
            src_env=src_env, tgt_env=tgt_env, lmbd_sew=lmbd_sew, ref_dir=ref_dir,
            src_qpos_slice=src_qpos_slice, tgt_qpos_slice=tgt_qpos_slice)

        assert src_agent.lat_act_dim == tgt_agent.lat_act_dim
        self.lat_act_dim = src_agent.lat_act_dim
        self.src_act_dim = src_agent.act_dim - 1
        self.tgt_act_dim = tgt_agent.act_dim - 1

        self.src_act_enc = src_agent.act_enc 
        self.src_act_dec = src_agent.act_dec
        self.tgt_act_enc = tgt_agent.act_enc 
        self.tgt_act_dec = tgt_agent.act_dec
        self.lat_disc = utils.build_mlp(self.lat_obs_dim + self.lat_act_dim, 1, n_layers, 
            hidden_dim, activation='leaky_relu', output_activation='identity').to(self.device)
        self.src_disc = utils.build_mlp(self.src_obs_dim + self.src_act_dim, 1, n_layers, 
            hidden_dim, activation='leaky_relu', output_activation='identity').to(self.device)
        self.tgt_disc = utils.build_mlp(self.tgt_obs_dim + self.tgt_act_dim, 1, n_layers, 
            hidden_dim, activation='leaky_relu', output_activation='identity').to(self.device)

        # Optimizers
        self.tgt_act_enc_opt = torch.optim.Adam(self.tgt_act_enc.parameters(), lr=lr)
        self.tgt_act_dec_opt = torch.optim.Adam(self.tgt_act_dec.parameters(), lr=lr)
        self.lat_disc_opt = torch.optim.Adam(self.lat_disc.parameters(), lr=lr)
        self.src_disc_opt = torch.optim.Adam(self.src_disc.parameters(), lr=lr)
        self.tgt_disc_opt = torch.optim.Adam(self.tgt_disc.parameters(), lr=lr)

    # 其余函数重用父类版（包括带 SEW 的 update_gen）
    
    def update_disc(self, src_obs, src_act, tgt_obs, tgt_act, L, step):
        """
        与你原始版本一致：判别器看 (lat_obs, lat_act) 拼接
        """
        # latent 编码（不求导）
        fake_lat_obs = self.tgt_obs_enc(tgt_obs).detach()
        fake_lat_act = self.tgt_act_enc(torch.cat([tgt_obs, tgt_act], dim=-1)).detach()
        real_lat_obs = self.src_obs_enc(src_obs).detach()
        real_lat_act = self.src_act_enc(torch.cat([src_obs, src_act], dim=-1)).detach()

        fake_lat_input = torch.cat([fake_lat_obs, fake_lat_act], dim=-1)
        real_lat_input = torch.cat([real_lat_obs, real_lat_act], dim=-1)
        lat_disc_loss = self.lat_disc(fake_lat_input).mean() - self.lat_disc(real_lat_input).mean()

        # 回到各自观测空间上的判别
        fake_src_obs = self.src_obs_dec(fake_lat_obs).detach()
        fake_src_act = self.src_act_dec(torch.cat([fake_src_obs, fake_lat_act], dim=-1)).detach()
        fake_src_input = torch.cat([fake_src_obs, fake_src_act], dim=-1)
        src_input = torch.cat([src_obs, src_act], dim=-1)
        src_disc_loss = self.src_disc(fake_src_input).mean() - self.src_disc(src_input).mean()

        real_tgt_obs = self.tgt_obs_dec(real_lat_obs).detach()
        real_tgt_act = self.tgt_act_dec(torch.cat([real_tgt_obs, real_lat_act], dim=-1)).detach()
        real_tgt_input = torch.cat([real_tgt_obs, real_tgt_act], dim=-1)
        tgt_input = torch.cat([tgt_obs, tgt_act], dim=-1)
        tgt_disc_loss = self.tgt_disc(real_tgt_input).mean() - self.tgt_disc(tgt_input).mean()

        # WGAN-GP
        lat_gp = self.compute_gradient_penalty(self.lat_disc, real_lat_input, fake_lat_input)
        src_gp = self.compute_gradient_penalty(self.src_disc, src_input, fake_src_input)
        tgt_gp = self.compute_gradient_penalty(self.tgt_disc, tgt_input, real_tgt_input)

        disc_loss = lat_disc_loss + src_disc_loss + tgt_disc_loss + self.lmbd_gp * (lat_gp + src_gp + tgt_gp)

        self.lat_disc_opt.zero_grad()
        self.src_disc_opt.zero_grad()
        self.tgt_disc_opt.zero_grad()
        disc_loss.backward()
        self.lat_disc_opt.step()
        self.src_disc_opt.step()
        self.tgt_disc_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_disc/lat_disc_loss', lat_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/lat_gp', lat_gp.item(), step)
            L.add_scalar('train_lat_disc/src_disc_loss', src_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/src_gp', src_gp.item(), step)
            L.add_scalar('train_lat_disc/tgt_disc_loss', tgt_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/tgt_gp', tgt_gp.item(), step)

            
    def update_gen(self, src_obs, src_act, src_next_obs, tgt_obs, tgt_act, tgt_next_obs, L, step):
        """
        与你原始 ObsActAligner.update_gen 等价 + 嵌入 SEW 对齐两项（cross/self）
        """
        # --- GAN 主损失（与原版一致）---
        fake_lat_obs = self.tgt_obs_enc(tgt_obs)
        fake_lat_act = self.tgt_act_enc(torch.cat([tgt_obs, tgt_act], dim=-1))
        real_lat_obs = self.src_obs_enc(src_obs)
        real_lat_act = self.src_act_enc(torch.cat([src_obs, src_act], dim=-1))

        fake_lat_input = torch.cat([fake_lat_obs, fake_lat_act], dim=-1)
        real_lat_input = torch.cat([real_lat_obs, real_lat_act], dim=-1)
        lat_gen_loss = -self.lat_disc(fake_lat_input).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs)
        fake_src_act = self.src_act_dec(torch.cat([fake_src_obs, fake_lat_act], dim=-1))
        fake_src_input = torch.cat([fake_src_obs, fake_src_act], dim=-1)
        src_input = torch.cat([src_obs, src_act], dim=-1)
        src_gen_loss = -self.src_disc(fake_src_input).mean()

        real_tgt_obs = self.tgt_obs_dec(real_lat_obs)
        real_tgt_act = self.tgt_act_dec(torch.cat([real_tgt_obs, real_lat_act], dim=-1))
        real_tgt_input = torch.cat([real_tgt_obs, real_tgt_act], dim=-1)
        tgt_input = torch.cat([tgt_obs, tgt_act], dim=-1)
        tgt_gen_loss = -self.tgt_disc(real_tgt_input).mean()

        gen_loss = lat_gen_loss + src_gen_loss + tgt_gen_loss

        # --- Cycle（与原版一致）---
        fake_lat_obs_1 = self.src_obs_enc(fake_src_obs)
        fake_lat_act_1 = self.src_act_enc(torch.cat([fake_src_obs, fake_src_act], dim=-1))
        pred_tgt_obs = self.tgt_obs_dec(fake_lat_obs_1)
        pred_tgt_act = self.tgt_act_dec(torch.cat([pred_tgt_obs, fake_lat_act_1], dim=-1))
        tgt_obs_cycle_loss = F.l1_loss(pred_tgt_obs, tgt_obs)
        tgt_act_cycle_loss = F.l1_loss(pred_tgt_act, tgt_act)

        real_lat_obs_1 = self.tgt_obs_enc(real_tgt_obs)
        real_lat_act_1 = self.tgt_act_enc(torch.cat([real_tgt_obs, real_tgt_act], dim=-1))
        pred_src_obs = self.src_obs_dec(real_lat_obs_1)
        pred_src_act = self.src_act_dec(torch.cat([pred_src_obs, real_lat_act_1], dim=-1))
        src_obs_cycle_loss = F.l1_loss(pred_src_obs, src_obs)
        src_act_cycle_loss = F.l1_loss(pred_src_act, src_act)
        cycle_loss = tgt_obs_cycle_loss + tgt_act_cycle_loss + src_obs_cycle_loss + src_act_cycle_loss

        # --- Dynamics（与原版一致）---
        fake_lat_next_obs = self.tgt_obs_enc(tgt_next_obs)
        pred_lat_act = self.inv_dyn(torch.cat([fake_lat_obs, fake_lat_next_obs], dim=-1))
        pred_act = self.tgt_act_dec(torch.cat([tgt_obs, pred_lat_act], dim=-1))
        inv_loss = F.mse_loss(pred_act, tgt_act)
        pred_lat_next_obs = self.fwd_dyn(torch.cat([fake_lat_obs, fake_lat_act], dim=-1))
        fwd_loss = F.mse_loss(pred_lat_next_obs, fake_lat_next_obs)

        # --- 新增：SEW 对齐（cross + self）---
        sew_loss = torch.tensor(0.0, device=self.device)
        if (self.lmbd_sew > 0.0) and (self.src_sew is not None) and (self.tgt_sew is not None):
            # Cross: σ( D_tgt(E_src(src)) ) ≈ σ(src)
            q_src        = self._extract_q(src_obs, which="src")              # 还原 q
            q_tgt_from_s = self._extract_q(real_tgt_obs, which="tgt")
            sigma_src        = self._sigma_batch("src", q_src)                # [B]
            sigma_tgt_from_s = self._sigma_batch("tgt", q_tgt_from_s)         # [B]
            L_sew_cross = F.mse_loss(sigma_tgt_from_s, sigma_src)

            # Self (目标域cycle): σ(pred_tgt_obs) ≈ σ(tgt_obs)
            q_tgt      = self._extract_q(tgt_obs, which="tgt")
            q_tgt_pred = self._extract_q(pred_tgt_obs, which="tgt")
            sigma_tgt      = self._sigma_batch("tgt", q_tgt)
            sigma_tgt_pred = self._sigma_batch("tgt", q_tgt_pred)
            L_sew_self = F.mse_loss(sigma_tgt_pred, sigma_tgt)

            sew_loss = L_sew_cross + L_sew_self

        loss = gen_loss + self.lmbd_cyc * cycle_loss + self.lmbd_dyn * (inv_loss + fwd_loss) + self.lmbd_sew * sew_loss

        self.tgt_obs_enc_opt.zero_grad()
        self.tgt_obs_dec_opt.zero_grad()
        self.tgt_act_enc_opt.zero_grad()
        self.tgt_act_dec_opt.zero_grad()
        loss.backward()
        self.tgt_obs_enc_opt.step()
        self.tgt_obs_dec_opt.step()
        self.tgt_act_enc_opt.step()
        self.tgt_act_dec_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_gen/lat_gen_loss', lat_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/src_gen_loss', src_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/tgt_gen_loss', tgt_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/inv_loss', inv_loss.item(), step)
            L.add_scalar('train_lat_gen/fwd_loss', fwd_loss.item(), step)
            L.add_scalar('train_lat_gen/cycle_loss', cycle_loss.item(), step)
            if self.lmbd_sew > 0.0:
                L.add_scalar('train_lat_gen/sew_loss', sew_loss.item(), step)