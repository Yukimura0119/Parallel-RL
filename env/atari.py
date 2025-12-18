from collections import deque

import cv2
import gymnasium as gym
import numpy as np
from mpi4py import MPI

gym.register_envs(ale_py)


class AtariPreprocessor:
    """
    將 Atari 的 RGB 畫面轉為 84x84 灰階並進行 Frame Stacking
    """

    def __init__(self, env, frame_stack=4):
        self.env = env
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def _process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._process_frame(obs)
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self._process_frame(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info


def create_env(env_name):
    return AtariPreprocessor(gym.make(env_name, render_mode="rgb_array"))


def run_worker(rank, env_name, base_seed):
    """
    Worker Process: 負責維護一個環境，接收 Master 指令並回傳結果
    """
    comm = MPI.COMM_WORLD
    seed = base_seed + rank

    # 建立環境
    env = create_env(env_name)
    env.reset(seed=seed)

    try:
        while True:
            # 接收指令 (Blocking)
            cmd, data = comm.recv(source=0)

            if cmd == "step":
                action = data
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                real_next_state = next_state

                if done:
                    reset_state, _ = env.reset()
                    # 回傳: (next_state, reward, done, reset_state, info)
                    comm.send(
                        (real_next_state, reward, done, reset_state, info), dest=0
                    )
                else:
                    comm.send((real_next_state, reward, done, None, info), dest=0)

            elif cmd == "reset":
                state, info = env.reset()
                comm.send((state, info), dest=0)

            elif cmd == "close":
                env.env.close()
                break
    except Exception as e:
        print(f"Worker {rank} error: {e}")
    finally:
        if hasattr(env, "env"):
            env.env.close()
