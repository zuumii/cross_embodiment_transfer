import os
import time
import numpy as np
import imageio.v2 as imageio
import robosuite as suite

# 建议在无显示环境使用 osmesa；如有 EGL 可改为 "egl"
os.environ.setdefault("MUJOCO_GL", "osmesa")

# -------- 相机相关：获取模型中全部相机名 --------
def list_all_camera_names(sim):
    names = []
    try:
        import mujoco
        model = sim.model
        # 遍历模型里所有 camera
        for cam_id in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            if name is not None:
                names.append(name)
    except Exception:
        # 兜底：一些常见相机名，逐个尝试
        names = ["agentview", "frontview", "sideview", "birdview", "topview"]
    # 去重并保持顺序
    seen = set()
    uniq = []
    for n in names:
        if n and n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq

# -------- 你的关节世界坐标计算（保持不变） --------
def joint_world_positions_for_arm(env, robot):
    m, d = env.sim.model, env.sim.data
    idx_raw = robot._ref_joint_pos_indexes
    if isinstance(idx_raw, (list, tuple)):
        arm_qpos_idx = set(int(i) for i in idx_raw)
    else:
        arm_qpos_idx = set(int(i) for i in np.array(idx_raw).ravel())

    pairs = []
    for jid in range(m.njnt):
        qpos_adr = int(m.jnt_qposadr[jid])
        if qpos_adr in arm_qpos_idx:
            name = m.joint_id2name(jid)
            bid = int(m.jnt_bodyid[jid])
            jnt_pos_local = m.jnt_pos[jid]
            body_xpos = d.xpos[bid]
            body_xmat = d.xmat[bid].reshape(3, 3)
            world = body_xpos + body_xmat @ jnt_pos_local
            pairs.append((name, world.copy()))
    pairs.sort(key=lambda t: int(m.jnt_qposadr[m.joint_name2id(t[0])]))
    return pairs

# -------- 渲染并保存单张 PNG --------
def save_rgb(env, out_path, cam_name="agentview", width=960, height=720):
    rgb = env.sim.render(width=width, height=height, camera_name=cam_name, depth=False)
    rgb = np.flipud(rgb)  # MuJoCo 返回的是自下而上
    imageio.imwrite(out_path, rgb)
    return out_path

# -------- 渲染并保存所有相机视角 --------
def save_all_cameras(env, out_dir="renders", width=960, height=720):
    os.makedirs(out_dir, exist_ok=True)
    cams = list_all_camera_names(env.sim)
    if not cams:
        cams = ["agentview", "frontview", "sideview", "birdview", "topview"]
    print("Cameras found / tried:", cams)

    saved = []
    ts = time.strftime("%Y%m%d-%H%M%S")
    for cam in cams:
        out_path = os.path.join(out_dir, f"{cam}_{ts}.png")
        try:
            save_rgb(env, out_path, cam_name=cam, width=width, height=height)
            print(f"[OK] saved {cam} -> {out_path}")
            saved.append(out_path)
        except Exception as e:
            print(f"[SKIP] {cam}: {e}")

    # 如果一个都没保存成功，再用兜底列表试一次
    if not saved:
        for cam in ["agentview", "frontview", "sideview", "birdview", "topview"]:
            out_path = os.path.join(out_dir, f"{cam}_{ts}_fallback.png")
            try:
                save_rgb(env, out_path, cam_name=cam, width=width, height=height)
                print(f"[OK-fallback] saved {cam} -> {out_path}")
                saved.append(out_path)
            except:
                pass
    return saved

if __name__ == "__main__":
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    robot = env.robots[0]

    # 仅给“手臂关节”赋值（7 维）
    arm_idx = np.array(robot._ref_joint_pos_indexes, dtype=int)
    q_arm = np.linspace(0, 0, len(arm_idx))  # 示例角度
    env.sim.data.qpos[arm_idx] = q_arm
    env.sim.forward()

    print("q_arm =", np.array2string(q_arm, precision=3))
    pairs = joint_world_positions_for_arm(env, robot)
    print(f"Panda arm DOF: {len(arm_idx)}")
    for name, xyz in pairs:
        print(f"{name:25s} -> [{xyz[0]: .6f}, {xyz[1]: .6f}, {xyz[2]: .6f}]")

    # EEF 坐标（可选打印）
    hp = robot._hand_pos
    if isinstance(hp, dict):
        for k, pos in hp.items():
            print(f"EEF({k})                   -> [{pos[0]: .6f}, {pos[1]: .6f}, {pos[2]: .6f}]")
    else:
        print(f"EEF                        -> [{hp[0]: .6f}, {hp[1]: .6f}, {hp[2]: .6f}]")

    # 渲染并保存所有相机的图片
    saved_paths = save_all_cameras(env, out_dir="renders", width=960, height=720)
    if not saved_paths:
        print("未能保存任何相机图片，请检查 MUJOCO_GL 或相机名称是否存在。")

    env.close()