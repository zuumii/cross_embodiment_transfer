import pathlib
import argparse
import time
from ruamel.yaml import YAML

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import utils
import replay_buffer
from td3 import TD3Agent, TD3ObsAgent, TD3ObsActAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    yaml = YAML(typ='safe')
    params = yaml.load(open(args.config))

    ##################################
    # CREATE DIRECTORY FOR LOGGING
    ##################################
    demo_dir = None
    if params.get('expert_folder') is not None:
        demo_dir = (pathlib.Path(params['expert_folder']) /
                    params['env_name'] / params['robots'] / params['controller_type']).resolve()

    if params.get('logdir_prefix') is None:
        logdir_prefix = pathlib.Path(__file__).parent
    else:
        logdir_prefix = pathlib.Path(params['logdir_prefix'])
    data_path = logdir_prefix / 'logs' / time.strftime("%m.%d.%Y")
    logdir = '_'.join([
        time.strftime("%H-%M-%S"),
        params['env_name'],
        params['robots'],
        params['controller_type'],
        params.get('suffix', '')
    ])
    logdir = (data_path / logdir).resolve()
    params['logdir'] = str(logdir)
    print("======== Train Params ========")
    print(params)

    # dump params
    logdir.mkdir(parents=True, exist_ok=True)
    import yaml as pyyaml
    with open(logdir / 'params.yml', 'w') as fp:
        pyyaml.safe_dump(params, fp, sort_keys=False)

    model_dir = (logdir / 'models').resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    # optional: buffer save dir
    replay_buffer_dir = (logdir / 'replay_buffer').resolve()
    if params.get('save_buffer', False):
        replay_buffer_dir.mkdir(parents=True, exist_ok=True)

    ##################################
    # SETUP ENV, AGENT
    ##################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 用 robosuite env 拉一次原生观测，拿到 robot/obj 切分尺寸
    env_probe = utils.make_robosuite_env(
        params['env_name'],
        robots=params['robots'],
        controller_type=params['controller_type'],
        **params['env_kwargs']
    )
    obs0 = env_probe.reset()
    robot_obs_shape = np.concatenate([obs0[k] for k in params['robot_obs_keys']]).shape
    obj_obs_shape = np.concatenate([obs0[k] for k in params['obj_obs_keys']]).shape

    params['obs_keys'] = params['robot_obs_keys'] + params['obj_obs_keys']

    # 训练用 env（已按 obs_keys 打包）
    env = utils.make(
        params['env_name'],
        robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=params['obs_keys'],
        seed=params['seed'],
        **params['env_kwargs'],
    )
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    print(f"Environment observation space shape {obs_shape}")
    print(f"Environment action space shape {act_shape}")

    eval_env = utils.make(
        params['env_name'],
        robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=params['obs_keys'],
        seed=params['seed'] + 100,
        **params['env_kwargs'],
    )

    logger = SummaryWriter(log_dir=params['logdir'])

    # ==== 选择 Agent ====
    # 示例：使用 TD3ObsActAgent（和你原来一致）
    agent_cls = TD3ObsActAgent
    obs_dims = {
        'obs_dim': obs_shape[0],
        'robot_obs_dim': robot_obs_shape[0],
        'obj_obs_dim': obj_obs_shape[0],
        'lat_obs_dim': params['lat_obs_dim'],
    }
    act_dims = {
        'act_dim': act_shape[0],
        'lat_act_dim': params['lat_act_dim'],
    }
    agent = agent_cls(obs_dims, act_dims, device)

    # ==== 创建 Replay Buffer 并灌入 demonstrations ====
    agent_replay_buffer = replay_buffer.ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=act_shape,
        capacity=params.get('replay_capacity', 2_000_000),
        batch_size=params['batch_size'],
        device=device
    )
    if demo_dir is not None:
        demo_paths = utils.load_episodes(demo_dir, params['obs_keys'])
        agent_replay_buffer.add_rollouts(demo_paths)
        print(f"Loaded demos from {demo_dir} into replay buffer.")

    # ==== 主训练循环 ====
    episode, episode_reward, done = 0, 0.0, True
    start_time = time.time()
    obs = env.reset()

    for step in range(params['total_timesteps']):
        # 评估 & 保存
        if step % params['evaluation']['interval'] == 0:
            print(f"[Step {step}] Evaluating...")
            logger.add_scalar('eval/episode', episode, step)
            utils.evaluate(eval_env, agent, 4, logger, step)

        if step % params['evaluation']['save_interval'] == 0:
            print(f"[Step {step}] Saving model...")
            step_dir = model_dir / f"step_{step:07d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            agent.save(step_dir)

            if params.get('save_buffer', False):
                agent_replay_buffer.save(replay_buffer_dir)

        # 采样动作与环境交互
        action = agent.sample_action(obs)  # 默认带探索噪声

        # 环境一步
        next_obs, reward, done, info = env.step(action)

        # --- 关键修复：正确构造 not_done（只在“真正终止”时为 0） ---
        timeout = False
        if isinstance(info, dict):
            # 兼容 gym / gymnasium / robosuite 的不同命名
            timeout = bool(
                info.get("TimeLimit.truncated", False) or
                info.get("TimeLimit.truncation", False) or
                info.get("truncated", False)
            )
        not_done = 0.0 if (done and not timeout) else 1.0
        # -----------------------------------------------

        # 累计奖励
        episode_reward += float(reward)

        # 写入 buffer
        agent_replay_buffer.add(obs, action, reward, next_obs, not_done)

        # 用“最新数据”更新
        agent.update(agent_replay_buffer, logger, step)

        # 处理 episode 结束
        if done:
            # 记录一次 episode 指标
            logger.add_scalar('train/episode_reward', episode_reward, step)
            logger.add_scalar('train/duration', time.time() - start_time, step)
            start_time = time.time()
            logger.add_scalar('train/episode', episode, step)
            logger.flush()

            # reset
            obs = env.reset()
            episode_reward = 0.0
            episode += 1
        else:
            obs = next_obs

    logger.close()


if __name__ == '__main__':
    main()