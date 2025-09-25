# joint_pose_panda.py
import os
import numpy as np
import robosuite as suite

# 降噪：无渲染时用 osmesa；你也可以改成 "egl" 或删掉这行
os.environ.setdefault("MUJOCO_GL", "osmesa")

def arm_joint_names_and_indices(env, robot):
    """
    用 qpos 映射筛出“手臂关节”（不含夹爪）；
    返回 [(joint_name, qpos_adr, jid)]，按 qpos 顺序排序
    """
    m = env.sim.model
    idx_raw = robot._ref_joint_pos_indexes
    arm_qpos_idx = set(int(i) for i in (idx_raw if isinstance(idx_raw, (list, tuple)) else np.array(idx_raw).ravel()))
    triples = []
    for jid in range(m.njnt):
        adr = int(m.jnt_qposadr[jid])   # 该 joint 在 qpos 的起始地址
        if adr in arm_qpos_idx:
            name = m.joint_id2name(jid)
            triples.append((name, adr, jid))
    triples.sort(key=lambda t: t[1])
    return triples

def joint_anchor_world(env, jid):
    """
    返回：
      - world_anchor: 关节锚点世界坐标
      - axis_world:   关节轴世界方向（仅对 SLIDE/HINGE 有意义，否则 None）
      - body_world:   关节所属 body 的世界坐标
      - body_name:    该 body 名称
    计算：world_anchor = body_xpos + body_xmat @ jnt_pos_local
          axis_world   = body_xmat @ jnt_axis_local   (仅 type ∈ {2,3})
    """
    m, d = env.sim.model, env.sim.data
    bid = int(m.jnt_bodyid[jid])
    jnt_pos_local = m.jnt_pos[jid].copy()            # (3,)
    body_xpos = d.xpos[bid].copy()                   # (3,)
    body_xmat = d.xmat[bid].reshape(3, 3).copy()     # (3,3)
    world_anchor = body_xpos + body_xmat @ jnt_pos_local

    axis_world = None
    jtype = int(m.jnt_type[jid])  # 0:FREE, 1:BALL, 2:SLIDE, 3:HINGE
    if jtype in (2, 3):
        axis_world = body_xmat @ m.jnt_axis[jid].copy()

    body_world = body_xpos
    body_name = m.body_id2name(bid)
    return world_anchor, axis_world, body_world, body_name

if __name__ == "__main__":
    # 1) 建 Panda 环境
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset() 
    robot = env.robots[0]

    # 2) 给“手臂关节”赋一组角度（长度必须匹配 7）
    arm_idx = robot._ref_joint_pos_indexes
    arm_dof = len(arm_idx)
    q_arm = np.linspace(2, 2, arm_dof)  # 示例角度
    env.sim.data.qpos[np.array(arm_idx, dtype=int)] = q_arm
    env.sim.forward()

    # 3) 打印每个手臂关节的笛卡尔（锚点）坐标 + 轴 + 所属 body 坐标
    triples = arm_joint_names_and_indices(env, robot)
    print(f"Panda arm DOF: {arm_dof}")
    for name, adr, jid in triples:
        anchor_w, axis_w, body_w, body_nm = joint_anchor_world(env, jid)
        axis_str = "axis: N/A" if axis_w is None else f"axis:[{axis_w[0]: .6f}, {axis_w[1]: .6f}, {axis_w[2]: .6f}]"
        print(f"{name:20s} qpos@{adr:2d}  "
              f"anchor:[{anchor_w[0]: .6f}, {anchor_w[1]: .6f}, {anchor_w[2]: .6f}]  "
              f"{axis_str}  "
              f"body({body_nm}):[{body_w[0]: .6f}, {body_w[1]: .6f}, {body_w[2]: .6f}]")

    # 4) （可选）打印 EEF 坐标
    hp = robot._hand_pos
    if isinstance(hp, dict):
        for k, pos in hp.items():
            print(f"EEF({k}):[{pos[0]: .6f}, {pos[1]: .6f}, {pos[2]: .6f}]")
    else:
        print(f"EEF      :[{hp[0]: .6f}, {hp[1]: .6f}, {hp[2]: .6f}]")

    env.close()