# compute_sew.py
import os
import math
import argparse
import numpy as np
import robosuite as suite

# 建议在无显示环境用 osmesa（或你本机有 EGL 就用 "egl"）
os.environ.setdefault("MUJOCO_GL", "osmesa")


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def project_to_plane(v, axis, eps=1e-9):
    """
    把向量 v 投影到垂直于 axis 的平面：v_perp = v - (v·axis) axis
    axis 需为单位向量
    """
    axis = normalize(axis)
    return v - np.dot(v, axis) * axis


def signed_sew_angle(S, E, W, ref_dir_world=np.array([0.0, 1.0, 0.0])):
    """
    计算球面 SEW 角（肘点绕肩–腕轴的有符号角）。
    - S,E,W: 3D np.array
    - ref_dir_world: 世界参考方向（将会投影到垂直于 u_SW 的平面作为0角方向）
    返回: (theta_rad, theta_deg)
    """
    S = np.asarray(S, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    u_sw = normalize(W - S)
    # 肘向量（也可用 E-((S+W)/2)，这里用 E-S 即可）
    v = E - S
    v_perp = project_to_plane(v, u_sw)

    # 参考方向投到同一平面；若退化则换备选
    e1 = project_to_plane(ref_dir_world, u_sw)
    if np.linalg.norm(e1) < 1e-6:
        # 参考方向与轴几乎共线，切换到 X 轴
        e1 = project_to_plane(np.array([1.0, 0.0, 0.0]), u_sw)
    e1 = normalize(e1)
    e2 = normalize(np.cross(u_sw, e1))  # 右手定则保证符号

    vp = normalize(v_perp)
    x = float(np.dot(vp, e1))
    y = float(np.dot(vp, e2))
    theta = math.atan2(y, x)  # (-pi, pi]
    return theta, math.degrees(theta)


def joint_world_positions_for_arm(env, robot):
    """
    返回 [(joint_name, world_xyz)]，仅包含“手臂关节”（通过 robot._ref_joint_pos_indexes 过滤）
    world_xyz = body_xpos + body_xmat @ jnt_pos_local
    """
    m, d = env.sim.model, env.sim.data

    # 兼容 list / np.array
    idx_raw = robot._ref_joint_pos_indexes
    if isinstance(idx_raw, (list, tuple)):
        arm_qpos_idx = set(int(i) for i in idx_raw)
    else:
        arm_qpos_idx = set(int(i) for i in np.array(idx_raw).ravel())

    pairs = []
    for jid in range(m.njnt):
        qpos_adr = int(m.jnt_qposadr[jid])  # 该关节在 qpos 中的起始索引
        if qpos_adr in arm_qpos_idx:
            name = m.joint_id2name(jid)
            bid = int(m.jnt_bodyid[jid])
            jnt_pos_local = m.jnt_pos[jid]                # (3,)
            body_xpos = d.xpos[bid]                       # (3,)
            body_xmat = d.xmat[bid].reshape(3, 3)         # (3,3)
            world = body_xpos + body_xmat @ jnt_pos_local # (3,)
            pairs.append((name, world.copy()))

    # 按 qpos 顺序排序，便于阅读
    pairs.sort(key=lambda t: int(m.jnt_qposadr[m.joint_name2id(t[0])]))
    return pairs


def pick_sew_indices_by_dof(arm_dof):
    """
    简单且通用的规则：7DoF 取第2/4/6个关节锚点作 S/E/W（0-based: 1,3,5）
    6DoF 取第2/3/5个关节锚点作 S/E/W（0-based: 1,2,4）
    其他 DoF 给出一个合理猜测（可以按需调整）
    """
    if arm_dof >= 7:
        return 1, 3, 5
    elif arm_dof == 6:
        return 1, 2, 4
    else:
        # 粗略猜测：S 取 1，E 取中间，W 取倒数第二
        mid = max(2, arm_dof // 2)
        return 1, mid, max(arm_dof - 2, 1)


def get_eef_pos(robot):
    hp = robot._hand_pos
    if isinstance(hp, dict):
        # 单臂通常只有一个键，取第一个
        k = list(hp.keys())[0]
        return np.array(hp[k], dtype=np.float64)
    return np.array(hp, dtype=np.float64)


def demo(robot_name="Panda", control_freq=20, set_demo_q=True):
    env = suite.make(
        env_name="Lift",
        robots=robot_name,
        has_renderer=False,
        use_camera_obs=False,
        control_freq=control_freq,
    )
    env.reset()
    robot = env.robots[0]

    # 仅给“手臂关节”赋值（为了有个确定姿态；你也可以读策略/数据集）
    arm_idx = np.array(robot._ref_joint_pos_indexes, dtype=int)
    arm_dof = len(arm_idx)
    if set_demo_q:
        # 来一组较为自然的关节角（单位按模型，一般为弧度）
        demo_q = np.linspace(1, 1, arm_dof)
        env.sim.data.qpos[arm_idx] = demo_q
        env.sim.forward()

    pairs = joint_world_positions_for_arm(env, robot)

    print(f"Robot: {robot_name} | Arm DOF: {arm_dof}")
    for name, xyz in pairs:
        print(f"{name:25s} -> [{xyz[0]: .6f}, {xyz[1]: .6f}, {xyz[2]: .6f}]")

    # 选择 S/E/W
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

    # 计算球面 SEW 角（参考方向默认世界 +Y，必要时自动切换）
    theta_rad, theta_deg = signed_sew_angle(S, E, W, ref_dir_world=np.array([0.0, 1.0, 0.0]))
    print("\nSEW angle (about shoulder–wrist axis):")
    print(f"theta = {theta_rad:.6f} rad  |  {theta_deg:.3f} deg")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute spherical SEW angle for a robosuite robot.")
    parser.add_argument("--robot", type=str, default="Panda", help="Robot name in robosuite (e.g., Panda, Sawyer)")
    parser.add_argument("--no_set_demo_q", action="store_true", help="Do not set demo joint angles (use env default)")
    args = parser.parse_args()

    demo(robot_name=args.robot, set_demo_q=(not args.no_set_demo_q))