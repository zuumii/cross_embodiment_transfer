import os
import argparse
import numpy as np
import torch
import robosuite as suite
import mujoco

os.environ.setdefault("MUJOCO_GL", "osmesa")

def unwrap_mj(sim):
    m = getattr(sim.model, "_model", sim.model)
    d = getattr(sim.data, "_data", sim.data)
    return m, d

# -------- 纯 Torch 的 SEW 几何（可微） --------
def _normalize(v, eps=1e-9):
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def sigma_from_SEW_torch(S, E, W, ref_dir=(0.0, 1.0, 0.0)):
    u = _normalize(W - S)  # SW 方向单位向量（臂法向）
    I = torch.eye(3, device=S.device, dtype=S.dtype).view(1,3,3).expand(S.shape[:-1]+(3,3))
    P = I - u.unsqueeze(-1) @ u.unsqueeze(-2)  # 投影到与 u 垂直的平面
    v = E - S
    v_perp = (P @ v.unsqueeze(-1)).squeeze(-1)
    v_hat = _normalize(v_perp)

    r = torch.tensor(ref_dir, device=S.device, dtype=S.dtype).view(1,3).expand_as(S)
    e1 = (P @ r.unsqueeze(-1)).squeeze(-1)
    tiny = (e1.norm(dim=-1, keepdim=True) < 1e-6)
    alt = torch.tensor([1.0,0.0,0.0], device=S.device, dtype=S.dtype).view(1,3).expand_as(S)
    e1 = torch.where(tiny, (P @ alt.unsqueeze(-1)).squeeze(-1), e1)
    e1 = _normalize(e1)
    e2 = _normalize(torch.cross(u, e1, dim=-1))

    x = (v_hat * e1).sum(-1); y = (v_hat * e2).sum(-1)
    theta = torch.atan2(y, x)
    sigma = torch.tan(0.5 * theta)
    return sigma, theta

# -------- “子 body 局部点” 的世界坐标（用于前向和校验）--------
def joint_point_world(sim, joint_id):
    m_raw, d_raw = unwrap_mj(sim)
    jid = int(joint_id)
    bid = int(m_raw.jnt_bodyid[jid])
    p_local = np.array(m_raw.jnt_pos[jid], dtype=np.float64).reshape(3,)
    R = d_raw.xmat[bid].reshape(3,3); t = d_raw.xpos[bid]
    world = t + R @ p_local
    return world

# --------（新）MuJoCo 解析雅可比：任意 body 上某世界点的位置雅可比 --------
def analytic_point_jac(sim, joint_id, arm_vel_cols):
    """
    返回关节锚点(世界坐标)对 qvel 的位置雅可比，并切到机械臂的列。
    维度：返回 3 x dof_arm
    说明：mj_jac 给的是速度雅可比 Jp(q) 使得 p_dot = Jp * qvel。
         在固定位姿的小扰动下 δp ≈ Jp(q) δq（把 Jp 当作 ∂p/∂q 的线性化即可）。
    """
    m, d = unwrap_mj(sim)
    bid = int(m.jnt_bodyid[joint_id])
    p_world = joint_point_world(sim, joint_id).astype(np.float64)

    Jp = np.zeros((3, m.nv), dtype=np.float64)
    Jr = np.zeros((3, m.nv), dtype=np.float64)
    mujoco.mj_jac(m, d, Jp, Jr, p_world, bid)

    # 只保留手臂相关的 qvel 列（robosuite 已提供）
    return Jp[:, arm_vel_cols]  # shape: 3 x dof_arm

def joint_list_for_arm(env, robot):
    m = env.sim.model
    idx_raw = robot._ref_joint_pos_indexes
    arm_qpos_set = set(int(i) for i in np.array(idx_raw).ravel())
    items = []
    for jid in range(m.njnt):
        qadr = int(m.jnt_qposadr[jid])
        if qadr in arm_qpos_set:
            items.append({"name": m.joint_id2name(jid),
                          "jid": jid,
                          "qpos_adr": qadr,
                          "body_id": int(m.jnt_bodyid[jid])})
    items.sort(key=lambda it: it["qpos_adr"])
    return items

def pick_sew_indices(robot_name, arm_joints):
    if "panda" in robot_name.lower():
        return 1, 2, 3   # S=joint2, E=joint3, W=joint4（从0数起）
    dof = len(arm_joints)
    if dof >= 7: return 1, 2, 3
    if dof == 6: return 1, 2, 4
    mid = max(2, dof // 2); return 1, mid, max(dof-2, 1)

# -------- 自定义 autograd：q -> σ （解析 J）--------
class SEWFromQ_MJ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, sim, arm_qpos_addr, arm_vel_cols, sew_jids, ref_dir=(0.0,1.0,0.0)):
        """
        q: torch tensor [dof], requires_grad=True
        sim: robosuite 的 sim（MuJoCo）
        arm_qpos_addr: 该臂在 qpos 中的列索引（用于写入姿态）
        arm_vel_cols:  该臂在 qvel 中的列索引（用于切雅可比）
        sew_jids: (jid_S, jid_E, jid_W)
        """
        device, dtype = q.device, q.dtype

        # 1) 把 q 写回 MuJoCo 并前向
        q_np = q.detach().cpu().numpy().astype(np.float64)
        sim.data.qpos[arm_qpos_addr] = q_np
        sim.forward()

        # 2) 取三点世界坐标
        jid_S, jid_E, jid_W = sew_jids
        S_np = joint_point_world(sim, jid_S)
        E_np = joint_point_world(sim, jid_E)
        W_np = joint_point_world(sim, jid_W)

        # 3) Torch 几何前向：σ(S,E,W)，并拿几何端梯度 gS,gE,gW
        with torch.enable_grad():
            S = torch.tensor(S_np, device=device, dtype=dtype, requires_grad=True)
            E = torch.tensor(E_np, device=device, dtype=dtype, requires_grad=True)
            W = torch.tensor(W_np, device=device, dtype=dtype, requires_grad=True)
            sigma, _ = sigma_from_SEW_torch(S.view(1,3), E.view(1,3), W.view(1,3), ref_dir=ref_dir)
            sigma = sigma.squeeze(0)
            gS, gE, gW = torch.autograd.grad(sigma, [S, E, W], retain_graph=False, create_graph=False)

        # 4) 用 MuJoCo 解析雅可比：J_S, J_E, J_W（3 x dof_arm）
        J_S_np = analytic_point_jac(sim, jid_S, arm_vel_cols)
        J_E_np = analytic_point_jac(sim, jid_E, arm_vel_cols)
        J_W_np = analytic_point_jac(sim, jid_W, arm_vel_cols)

        # 转成 torch
        J_S = torch.from_numpy(J_S_np).to(device=device, dtype=dtype)
        J_E = torch.from_numpy(J_E_np).to(device=device, dtype=dtype)
        J_W = torch.from_numpy(J_W_np).to(device=device, dtype=dtype)

        # 保存供 backward 使用
        ctx.save_for_backward(gS.detach(), gE.detach(), gW.detach(), J_S, J_E, J_W)
        return sigma.view(1)

    @staticmethod
    def backward(ctx, grad_out):
        gS, gE, gW, J_S, J_E, J_W = ctx.saved_tensors
        dσ_dq = (gS.view(1,3) @ J_S + gE.view(1,3) @ J_E + gW.view(1,3) @ J_W).squeeze(0)
        return grad_out.view(1) * dσ_dq, None, None, None, None, None

def sigma_from_q_once(sim, q_tensor, arm_qpos_addr, arm_vel_cols, sew_jids, ref_dir=(0.0,1.0,0.0)):
    return SEWFromQ_MJ.apply(q_tensor, sim, arm_qpos_addr, arm_vel_cols, sew_jids, ref_dir)

# -------- 主程序（含 σ 直接 FD 校验）--------
def main(robot="Panda", seed=0, set_demo=True, eps=1e-5):
    np.random.seed(seed); torch.manual_seed(seed)
    env = suite.make(env_name="Lift", robots=robot, has_renderer=False, use_camera_obs=False, control_freq=20)
    env.reset()
    robot_obj = env.robots[0]; sim = env.sim

    # 关节索引（qpos / qvel）
    arm_qpos_addr = [int(i) for i in np.array(robot_obj._ref_joint_pos_indexes).ravel().tolist()]
    arm_vel_cols  = np.array(robot_obj._ref_joint_vel_indexes).ravel().astype(int)  # 解析雅可比用它切列
    dof = len(arm_qpos_addr); assert dof > 0

    # 关节列表与 S/E/W 选择
    arm_joints = joint_list_for_arm(env, robot_obj)
    s_i, e_i, w_i = pick_sew_indices(robot, arm_joints)
    jid_S = arm_joints[s_i]["jid"]; jid_E = arm_joints[e_i]["jid"]; jid_W = arm_joints[w_i]["jid"]
    print(f"[{robot}] dof={dof} | S={env.sim.model.joint_id2name(jid_S)}, "
          f"E={env.sim.model.joint_id2name(jid_E)}, W={env.sim.model.joint_id2name(jid_W)}")

    # 初始 q
    if set_demo:
        q0 = np.linspace(0.2, -0.4, dof).astype(np.float64)
    else:
        q0 = (np.random.randn(dof) * 0.1).astype(np.float64)

    # Torch 张量（用 double 精度，减少数值误差）
    q = torch.tensor(q0, dtype=torch.float64, requires_grad=True)

    # 前向 + 反传（用解析 J）
    sigma = sigma_from_q_once(sim, q, arm_qpos_addr, arm_vel_cols, (jid_S, jid_E, jid_W), ref_dir=(0.0,1.0,0.0))
    print(f"sigma(q0) = {sigma.item():.6f}")
    q.grad = None; sigma.backward()
    grad_auto = q.grad.detach().cpu().numpy().astype(np.float64)
    print(f"||dσ/dq||={np.linalg.norm(grad_auto):.6f}, dσ/dq = {grad_auto}")

    # —— 端到端中心差分（验证用，保持你的逻辑不变）——
    fd = np.zeros_like(q0, dtype=np.float64)
    for i in range(dof):
        qp = q0.copy(); qm = q0.copy()
        qp[i] += eps; qm[i] -= eps
        sim.data.qpos[arm_qpos_addr] = qp; sim.forward()
        Sp = joint_point_world(sim, jid_S); Ep = joint_point_world(sim, jid_E); Wp = joint_point_world(sim, jid_W)
        sp,_ = sigma_from_SEW_torch(torch.tensor(Sp, dtype=torch.float64).view(1,3),
                                    torch.tensor(Ep, dtype=torch.float64).view(1,3),
                                    torch.tensor(Wp, dtype=torch.float64).view(1,3))
        sim.data.qpos[arm_qpos_addr] = qm; sim.forward()
        Sm = joint_point_world(sim, jid_S); Em = joint_point_world(sim, jid_E); Wm = joint_point_world(sim, jid_W)
        sm,_ = sigma_from_SEW_torch(torch.tensor(Sm, dtype=torch.float64).view(1,3),
                                    torch.tensor(Em, dtype=torch.float64).view(1,3),
                                    torch.tensor(Wm, dtype=torch.float64).view(1,3))
        fd[i] = float(sp.item() - sm.item()) / (2.0 * eps)

    sim.data.qpos[arm_qpos_addr] = q0; sim.forward()
    print("finite-diff dσ/dq ≈", fd)
    cos_sim = float(np.dot(fd, grad_auto) / (np.linalg.norm(fd) * np.linalg.norm(grad_auto) + 1e-12))
    print(f"cos(angle(fd, autograd)) = {cos_sim:.6f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_set_demo_q", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()
    main(robot=args.robot, seed=args.seed, set_demo=(not args.no_set_demo_q), eps=args.eps)