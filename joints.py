# joint_pose_sawyer.py
import os
import numpy as np
import robosuite as suite

os.environ.setdefault("MUJOCO_GL", "osmesa")  # or "egl"

def arm_joint_names_and_indices(env, robot):
    m = env.sim.model
    idx_raw = robot._ref_joint_pos_indexes
    arm_qpos_idx = set(int(i) for i in (idx_raw if isinstance(idx_raw, (list, tuple)) else np.array(idx_raw).ravel()))
    triples = []
    for jid in range(m.njnt):
        adr = int(m.jnt_qposadr[jid])
        if adr in arm_qpos_idx:
            name = m.joint_id2name(jid)
            triples.append((name, adr, jid))
    triples.sort(key=lambda t: t[1])
    return triples

def joint_anchor_world(env, jid):
    m, d = env.sim.model, env.sim.data
    bid = int(m.jnt_bodyid[jid])
    jnt_pos_local = m.jnt_pos[jid].copy()
    body_xpos = d.xpos[bid].copy()
    body_xmat = d.xmat[bid].reshape(3, 3).copy()
    world_anchor = body_xpos + body_xmat @ jnt_pos_local

    axis_world = None
    jtype = int(m.jnt_type[jid])  # 0:FREE, 1:BALL, 2:SLIDE, 3:HINGE
    if jtype in (2, 3):
        axis_world = body_xmat @ m.jnt_axis[jid].copy()

    body_world = body_xpos
    body_name = m.body_id2name(bid)
    return world_anchor, axis_world, body_world, body_name

if __name__ == "__main__":
    env = suite.make(
        env_name="Lift",        # 任务随意，主要是加载机器人
        robots="Sawyer",        # 切换为 Sawyer
        has_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    robot = env.robots[0]

    # 给“手臂关节”赋一组角度（长度自动匹配）
    arm_idx = robot._ref_joint_pos_indexes
    arm_dof = len(arm_idx)
    q_arm = np.linspace(-0.6, 0.6, arm_dof)  # 示例角度
    env.sim.data.qpos[np.array(arm_idx, dtype=int)] = q_arm
    env.sim.forward()

    # 打印每个手臂关节的笛卡尔（锚点）坐标 + 轴 + 所属 body 坐标
    triples = arm_joint_names_and_indices(env, robot)
    print(f"Sawyer arm DOF: {arm_dof}")
    for name, adr, jid in triples:
        anchor_w, axis_w, body_w, body_nm = joint_anchor_world(env, jid)
        axis_str = "axis: N/A" if axis_w is None else f"axis:[{axis_w[0]: .6f}, {axis_w[1]: .6f}, {axis_w[2]: .6f}]"
        print(f"{name:20s} qpos@{adr:2d}  "
              f"anchor:[{anchor_w[0]: .6f}, {anchor_w[1]: .6f}, {anchor_w[2]: .6f}]  "
              f"{axis_str}  "
              f"body({body_nm}):[{body_w[0]: .6f}, {body_w[1]: .6f}, {body_w[2]: .6f}]")

    # （可选）打印 EEF 坐标
    hp = robot._hand_pos
    if isinstance(hp, dict):
        for k, pos in hp.items():
            print(f"EEF({k}):[{pos[0]: .6f}, {pos[1]: .6f}, {pos[2]: .6f}]")
    else:
        print(f"EEF      :[{hp[0]: .6f}, {hp[1]: .6f}, {hp[2]: .6f}]")

    env.close()