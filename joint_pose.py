import os
import numpy as np
import robosuite as suite

# 建议在无显示环境用 osmesa，或者你本机有 EGL 就留着默认
os.environ.setdefault("MUJOCO_GL", "osmesa")  # or "egl"

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

    # 仅给“手臂关节”赋值
    arm_idx = robot._ref_joint_pos_indexes
    arm_dof = len(arm_idx)
    q_arm = np.linspace(1, 1, arm_dof)  # 示例角度
    env.sim.data.qpos[np.array(arm_idx, dtype=int)] = q_arm
    env.sim.forward()

    pairs = joint_world_positions_for_arm(env, robot)
    print(f"Panda arm DOF: {arm_dof}")
    for name, xyz in pairs:
        print(f"{name:25s} -> [{xyz[0]: .6f}, {xyz[1]: .6f}, {xyz[2]: .6f}]")

    # （可选）打印 EEF 坐标
    hp = robot._hand_pos
    if isinstance(hp, dict):
        for k, pos in hp.items():
            print(f"EEF({k})                   -> [{pos[0]: .6f}, {pos[1]: .6f}, {pos[2]: .6f}]")
    else:
        print(f"EEF                        -> [{hp[0]: .6f}, {hp[1]: .6f}, {hp[2]: .6f}]")

    env.close()