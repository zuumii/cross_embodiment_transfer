# -*- coding: utf-8 -*-
# scripts/eval_dual.py — headless 录像 & 单/双机器人评测（目标机器人仅用编解码器，可共享源策略）

import os
import pathlib
import argparse
from ruamel.yaml import YAML

import numpy as np
import torch
import imageio.v3 as iio

from align import ObsActAgent as Agent
import utils


def _select_mujoco_gl(force: str | None = None):
    if force:
        os.environ["MUJOCO_GL"] = force
        return
    if "MUJOCO_GL" in os.environ:
        return
    if "DISPLAY" not in os.environ:
        try:
            import pynvml  # noqa: F401
            os.environ["MUJOCO_GL"] = "egl"
        except Exception:
            os.environ["MUJOCO_GL"] = "osmesa"


def render_frame(env, camera: str, width: int, height: int):
    try:
        frame = env.render(mode="rgb_array", camera_name=camera, width=width, height=height)
        if frame is not None:
            return frame
    except TypeError:
        try:
            frame = env.render(mode="rgb_array")
            if frame is not None:
                return frame
        except Exception:
            pass
    except Exception:
        pass

    if hasattr(env, "sim"):
        sim = env.sim
        try:
            return sim.render(camera_name=camera, width=width, height=height)
        except Exception:
            try:
                return sim.render_camera(camera_name=camera, width=width, height=height)
            except Exception as e:
                raise RuntimeError(f"无法从相机 {camera} 渲染帧: {e}")
    raise RuntimeError("未找到可用的渲染路径。")


class VideoSink:
    def __init__(self, out_path: pathlib.Path, fps: int):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path = out_path
        self.fps = fps
        self.frames = []

    def append(self, frame: np.ndarray):
        self.frames.append(frame)

    def close(self):
        if len(self.frames) == 0:
            return
        iio.imwrite(self.out_path.as_posix(), np.stack(self.frames, axis=0), fps=self.fps)


def load_agent(model_dir: pathlib.Path, obs_dims: dict, act_dims: dict, device: torch.device,
               load_actor: bool = True) -> Agent:
    agent = Agent(obs_dims, act_dims, device)
    # 依赖 align.ObsActAgent.load 的新参数 load_actor
    agent.load(model_dir, load_actor=load_actor)
    if hasattr(agent, "eval_mode"):
        agent.eval_mode()
    return agent


def adopt_policy(dst_agent: Agent, src_agent: Agent):
    if not hasattr(src_agent, "actor") or not hasattr(dst_agent, "actor"):
        raise RuntimeError("Agent 缺少 actor 属性，请按说明修改 align.py")
    dst_agent.adopt_actor_from(src_agent)


def make_env_block(base_params: dict, block: dict, offscreen: bool, cam_name: str, cam_w: int, cam_h: int):
    rs_env = utils.make_robosuite_env(
        block["env_name"],
        robots=block["robots"],
        controller_type=block["controller_type"],
        **block.get("env_kwargs", {}),
    )
    obs = rs_env.reset()
    robot_obs = np.concatenate([obs[k] for k in block["robot_obs_keys"]], axis=0)
    obj_obs = np.concatenate([obs[k] for k in block["obj_obs_keys"]], axis=0)

    env = utils.make(
        block["env_name"],
        robots=block["robots"],
        controller_type=block["controller_type"],
        obs_keys=block["robot_obs_keys"] + block["obj_obs_keys"],
        seed=block.get("seed", base_params.get("seed", 0)),
        render=not offscreen,
        render_offscreen=True,           # 如果你的 utils.make 不支持这些参数，就删掉这三行
        camera=cam_name,                 # 并仅保留 render=not offscreen
        camera_width=cam_w,
        camera_height=cam_h,
        **block.get("env_kwargs", {}),
    )

    obs_dims = {
        "robot_obs_dim": robot_obs.shape[0],
        "obs_dim": robot_obs.shape[0] + obj_obs.shape[0],
        "lat_obs_dim": block["lat_obs_dim"],
        "obj_obs_dim": obj_obs.shape[0],
    }
    act_dims = {
        "act_dim": env.action_space.shape[0],
        "lat_act_dim": block["lat_act_dim"],
    }
    return env, obs_dims, act_dims


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config path")
    return p.parse_args()


def main():
    _select_mujoco_gl()
    args = parse_args()

    yaml = YAML(typ="safe")
    params = yaml.load(open(args.config, "r", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    headless = bool(params.get("headless", True))
    vid_cfg = params.get("video", {})
    save_video = bool(vid_cfg.get("enabled", True))
    cam = vid_cfg.get("camera", "agentview")
    W = int(vid_cfg.get("width", 640))
    H = int(vid_cfg.get("height", 480))
    FPS = int(vid_cfg.get("fps", 20))
    out_dir = pathlib.Path(vid_cfg.get("out_dir", "videos/eval"))

    dual = params.get("dual_eval", {}).get("enabled", False)

    if not dual:
        block = {
            "env_name": params["env_name"],
            "robots": params["robots"],
            "controller_type": params["controller_type"],
            "robot_obs_keys": params["robot_obs_keys"],
            "obj_obs_keys": params["obj_obs_keys"],
            "lat_obs_dim": params["lat_obs_dim"],
            "lat_act_dim": params["lat_act_dim"],
            "env_kwargs": params.get("env_kwargs", {}),
            "seed": params.get("seed", 0),
        }
        env, obs_dims, act_dims = make_env_block(params, block, headless, cam, W, H)
        agent = load_agent(pathlib.Path(params["model_dir"]), obs_dims, act_dims, device, load_actor=True)

        num_eps = int(params.get("num_episodes", 10))
        for ep in range(num_eps):
            obs, done, ep_ret, steps = env.reset(), False, 0.0, 0
            sink = VideoSink(out_dir / f"{ep:03d}_{block['env_name']}_{block['robots']}.mp4", FPS) if save_video else None
            while not done:
                action = agent.sample_action(obs, deterministic=True)
                obs, rew, done, _ = env.step(action)
                ep_ret += rew
                if sink:
                    sink.append(render_frame(env, cam, W, H))
                steps += 1
            if sink:
                sink.close()
            print(f"[Single] Episode {ep}: return={ep_ret:.3f}, steps={steps}")

    else:
        src = params["dual_eval"]["src"]
        tgt = params["dual_eval"]["tgt"]

        env_src, obs_src, act_src = make_env_block(params, src, headless, cam, W, H)
        env_tgt, obs_tgt, act_tgt = make_env_block(params, tgt, headless, cam, W, H)

        agent_src = load_agent(pathlib.Path(src["model_dir"]), obs_src, act_src, device, load_actor=True)
        agent_tgt = load_agent(pathlib.Path(tgt.get("model_dir", src["model_dir"])), obs_tgt, act_tgt, device,
                               load_actor=bool(tgt.get("load_policy", False)))
        if not tgt.get("load_policy", False):
            adopt_policy(agent_tgt, agent_src)

        num_eps = int(params.get("num_episodes", 10))
        max_horizon = int(max(src.get("env_kwargs", {}).get("horizon", 200),
                              tgt.get("env_kwargs", {}).get("horizon", 200)))

        for ep in range(num_eps):
            obsA, doneA, retA = env_src.reset(), False, 0.0
            obsB, doneB, retB = env_tgt.reset(), False, 0.0
            t = 0

            sinkA = VideoSink(out_dir / f"{ep:03d}_SRC_{src['robots']}.mp4", FPS) if save_video else None
            sinkB = VideoSink(out_dir / f"{ep:03d}_TGT_{tgt['robots']}.mp4", FPS) if save_video else None

            while (not doneA or not doneB) and t < max_horizon:
                if not doneA:
                    actA = agent_src.sample_action(obsA, deterministic=True)
                    obsA, rewA, doneA, _ = env_src.step(actA)
                    retA += rewA
                    if sinkA: sinkA.append(render_frame(env_src, cam, W, H))
                else:
                    if sinkA: sinkA.append(render_frame(env_src, cam, W, H))

                if not doneB:
                    actB = agent_tgt.sample_action(obsB, deterministic=True)
                    obsB, rewB, doneB, _ = env_tgt.step(actB)
                    retB += rewB
                    if sinkB: sinkB.append(render_frame(env_tgt, cam, W, H))
                else:
                    if sinkB: sinkB.append(render_frame(env_tgt, cam, W, H))

                t += 1

            if sinkA: sinkA.close()
            if sinkB: sinkB.close()
            print(f"[Dual] Episode {ep}: SRC({src['robots']}) ret={retA:.3f} | TGT({tgt['robots']}) ret={retB:.3f} | steps={t}")

    print("Done.")


if __name__ == "__main__":
    main()