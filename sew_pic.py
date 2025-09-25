# run_robot_sew_and_render.py
import os
import math
import time
import argparse
import numpy as np
import imageio.v2 as imageio
import robosuite as suite

# 建议在无显示环境使用 osmesa；如有 EGL 可改为 "egl"
os.environ.setdefault("MUJOCO_GL", "osmesa")


# ------------------ 通用向量工具 ------------------
def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def project_to_plane(v, axis):
    axis = normalize(axis)
    return v - np.dot(v, axis) * axis


# ------------------ SEW 角计算 ------------------
def signed_sew_angle(S, E, W, ref_dir_world=np.array([0.0, 1.0, 0.0])):
    """
    计算球面 SEW 角（肘点绕肩–腕轴的有符号角）。
    - S,E,W: 3D np.array
    - ref_dir_world: 世界参考方向（将会投影到垂直于 u_SW 的平面作为0角方向）
    返回: (theta_rad, theta_deg, sigma) 其中 sigma = tan(theta/2)
    """
    S = np.asarray(S, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    # 肩->腕轴
    u_sw = normalize(W - S)

    # 肘向量投影到垂直平面
    v = E - S
    v_perp = project_to_plane(v, u_sw)
    v_perp = normalize(v_perp)

    # 参考方向也投到同一平面；若与轴几乎共线则切换参考
    e1 = project_to_plane(ref_dir_world, u_sw)
    if np.linalg.norm(e1) < 1e-6:
        e1 = project_to_plane(np.array([1.0, 0.0, 0.0]), u_sw)
    e1 = normalize(e1)
    e2 = normalize(np.cross(u_sw, e1))  # 右手系，给符号

    # 平面内有符号角
    x = float(np.dot(v_perp, e1))
    y = float(np.dot(v_perp, e2))
    theta = math.atan2(y, x)  # (-pi, pi]
    sigma = math.tan(theta / 2.0)  # stereographic 参数（一维版）

    return theta, math.degrees(theta), sigma


# ------------------ 关节世界坐标（沿用你的逻辑） ------------------
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


def pick_sew_indices_by_dof(arm_dof):
    """
    简单且通用的规则：
      - 7DoF：S/E/W 取第2/4/6个关节锚点（0-based: 1,3,5）
      - 6DoF：S/E/W 取第2/3/5个关节锚点（0-based: 1,2,4）
      - 其他：S=1, E=中位, W=倒数第二（可按需细化）
    """
    if arm_dof >= 7:
        return 1, 3, 5
    elif arm_dof == 6:
        return 1, 2, 4
    else:
        mid = max(2, arm_dof // 2)
        return 1, mid, max(arm_dof - 2, 1)


def get_eef_pos(robot):
    hp = robot._hand_pos
    if isinstance(hp, dict):
        k = list(hp.keys())[0]
        return np.array(hp[k], dtype=np.float64)
    return np.array(hp, dtype=np.float64)


# ------------------ 相机渲染（沿用 + 小封装） ------------------
def list_all_camera_names(sim):
    names = []
    try:
        import mujoco
        model = sim.model
        for cam_id in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            if name is not None:
                names.append(name)
    except Exception:
        names = ["agentview", "frontview", "sideview", "birdview", "topview"]

    seen, uniq = set(), []
    for n in names:
        if n and n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq


def save_rgb(env, out_path, cam_name="agentview", width=960, height=720):
    rgb = env.sim.render(width=width, height=height, camera_name=cam_name, depth=False)
    rgb = np.flipud(rgb)  # MuJoCo 返回自下而上的图像，需要翻转
    imageio.imwrite(out_path, rgb)
    return out_path


def save_all_cameras(env, out_dir="renders", width=960, height=720, filename_suffix=""):
    os.makedirs(out_dir, exist_ok=True)
    cams = list_all_camera_names(env.sim)
    if not cams:
        cams = ["agentview", "frontview", "sideview", "birdview", "topview"]
    print("Cameras found / tried:", cams)

    saved, ts = [], time.strftime("%Y%m%d-%H%M%S")
    for cam in cams:
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        out_path = os.path.join(out_dir, f"{cam}_{ts}{suffix}.png")
        try:
            save_rgb(env, out_path, cam_name=cam, width=width, height=height)
            print(f"[OK] saved {cam} -> {out_path}")
            saved.append(out_path)
        except Exception as e:
            print(f"[SKIP] {cam}: {e}")

    if not saved:
        # 兜底
        for cam in ["agentview", "frontview", "sideview", "birdview", "topview"]:
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            out_path = os.path.join(out_dir, f"{cam}_{ts}_fallback{suffix}.png")
            try:
                save_rgb(env, out_path, cam_name=cam, width=width, height=height)
                print(f"[OK-fallback] saved {cam} -> {out_path}")
                saved.append(out_path)
            except:
                pass
    return saved


# ------------------ 主流程 ------------------
def main(robot_name="Panda", control_freq=20, set_demo_q=True, out_dir="renders", img_w=960, img_h=720):
    env = suite.make(
        env_name="Lift",
        robots=robot_name,
        has_renderer=False,
        use_camera_obs=False,
        control_freq=control_freq,
    )
    env.reset()
    robot = env.robots[0]

    # 为了有个确定的姿态（也可从你的数据集/策略里读）
    arm_idx = np.array(robot._ref_joint_pos_indexes, dtype=int)
    arm_dof = len(arm_idx)
    if set_demo_q:
        demo_q = np.linspace(1, 1, arm_dof)  # 一组较自然的角度
        env.sim.data.qpos[arm_idx] = demo_q
        env.sim.forward()

    # 打印关节世界坐标
    pairs = joint_world_positions_for_arm(env, robot)
    print(f"Robot: {robot_name} | Arm DOF: {arm_dof}")
    print("Joint world positions (arm only):")
    for name, xyz in pairs:
        print(f"{name:25s} -> [{xyz[0]: .6f}, {xyz[1]: .6f}, {xyz[2]: .6f}]")

    # 选 S/E/W/T
    s_i, e_i, w_i = pick_sew_indices_by_dof(arm_dof)
    S = pairs[s_i][1]
    E = pairs[e_i][1]
    W = pairs[w_i][1]
    T = get_eef_pos(robot)

    print("\nPicked keypoints (by simple rule):")
    print(f"S (joint #{s_i+1:02d}) -> [{S[0]: .6f}, {S[1]: .6f}, {S[2]: .6f}]")
    print(f"E (joint #{e_i+1:02d}) -> [{E[0]: .6f}, {E[1]: .6f}, {E[2]: .6f}]")
    print(f"W (joint #{w_i+1:02d}) -> [{W[0]: .6f}, {W[1]: .6f}, {W[2]: .6f}]")
    print(f"T (EEF)            -> [{T[0]: .6f}, {T[1]: .6f}, {T[2]: .6f}]")

    # 计算 SEW 角（以及 stereographic 参数）
    theta_rad, theta_deg, sigma = signed_sew_angle(S, E, W, ref_dir_world=np.array([0.0, 1.0, 0.0]))
    print("\nSEW angle (about shoulder–wrist axis):")
    print(f"theta = {theta_rad:.6f} rad  |  {theta_deg:.3f} deg")
    print(f"stereographic sigma = {sigma:.6f}")

    # 渲染并保存所有相机的图片（文件名附上角度，便于检索）
    # 角度放整数或一位小数，避免文件名过长
    angle_tag = f"SEW_{theta_deg:+.1f}deg"
    saved_paths = save_all_cameras(env, out_dir=out_dir, width=img_w, height=img_h, filename_suffix=angle_tag)
    if not saved_paths:
        print("未能保存任何相机图片，请检查 MUJOCO_GL 或相机名称是否存在。")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SEW angle and render all cameras in robosuite.")
    parser.add_argument("--robot", type=str, default="Panda", help="Robot name in robosuite (e.g., Panda, Sawyer)")
    parser.add_argument("--no_set_demo_q", action="store_true", help="Do not set demo joint angles (use env default)")
    parser.add_argument("--out_dir", type=str, default="renders", help="Directory to save camera renders")
    parser.add_argument("--width", type=int, default=960, help="Image width")
    parser.add_argument("--height", type=int, default=720, help="Image height")
    args = parser.parse_args()

    main(
        robot_name=args.robot,
        set_demo_q=(not args.no_set_demo_q),
        out_dir=args.out_dir,
        img_w=args.width,
        img_h=args.height,
    )