import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from agent.buffer import NStepBuffer, PrioritizedReplayBuffer
from env.atari import create_env
from models.DQN import DQN
from mpi4py import MPI


class MPIVectorEnv:
    """
    管理多個 MPI Worker 的介面
    """

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()

        if self.size < num_envs + 1:
            raise RuntimeError(
                f"MPI Size {self.size} is not enough for 1 Master + {num_envs} Workers."
            )

    def reset(self):
        # 發送 Reset 指令給所有 Worker (Rank 1 ~ num_envs)
        for i in range(self.num_envs):
            self.comm.send(("reset", None), dest=i + 1)

        results = []
        for i in range(self.num_envs):
            results.append(self.comm.recv(source=i + 1))

        states, infos = zip(*results)
        return np.stack(states), infos

    def step(self, actions):
        # 回退到 Blocking Communication 以避免記憶體錯誤 (free(): invalid next size)
        # Python 物件的 Non-blocking 傳輸 (isend/irecv) 需要非常小心地管理物件生命週期
        # 簡單的 isend/irecv 在高頻率呼叫下容易導致 GC 與 MPI 底層衝突

        # 發送 Step 指令與 Action
        for i, action in enumerate(actions):
            self.comm.send(("step", action), dest=i + 1)

        results = []
        for i in range(self.num_envs):
            results.append(self.comm.recv(source=i + 1))

        # results: list of (next_state, reward, done, reset_state, info)
        next_states, rewards, dones, reset_states, infos = zip(*results)

        next_states = np.stack(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # 處理 Auto-Reset 的狀態回傳
        current_obs_for_agent = np.array(
            [
                reset_states[i] if dones[i] else next_states[i]
                for i in range(self.num_envs)
            ]
        )

        return next_states, rewards, dones, current_obs_for_agent, infos

    def close(self):
        for i in range(self.num_envs):
            self.comm.send(("close", None), dest=i + 1)


# ==========================================
# 4. 資料結構 (Buffer & N-Step)
# ==========================================


# ==========================================
# 5. Agent 實作 (Parallel Support)
# ==========================================
class MultiStepDQNAgent:
    def __init__(self, env_name, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.num_envs = args.num_envs
        self.env_name = env_name
        # 初始化平行環境 (MPI)
        self.envs = MPIVectorEnv(self.num_envs)

        # 建立評估用的環境 (單一環境)
        self.eval_env = create_env(env_name)

        # 取得 Action Space (假設所有環境一樣)
        temp_env = gym.make(env_name)
        self.num_actions = temp_env.action_space.n
        temp_env.close()

        self.q_net = DQN(self.num_actions).to(self.device)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.replay_start_size = args.replay_start_size
        self.gamma = args.discount_factor
        self.n_step = args.n_step
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.eval_freq = args.eval_freq

        self.memory = PrioritizedReplayBuffer(args.memory_size)
        # 為每個環境建立獨立的 N-Step Buffer
        self.n_step_buffers = [
            NStepBuffer(n_step=self.n_step, gamma=self.gamma)
            for _ in range(self.num_envs)
        ]

        self.total_steps = 0
        self.train_count = 0
        self.target_update_freq = args.target_update_frequency
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.best_reward = -float("inf")

    def select_actions(self, states):
        """
        Batch Action Selection
        states: numpy array of shape (num_envs, 4, 84, 84)
        """
        if random.random() < self.epsilon:
            return [
                random.randint(0, self.num_actions - 1) for _ in range(self.num_envs)
            ]

        state_tensor = torch.from_numpy(states).float().to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            actions = q_values.argmax(dim=1).cpu().numpy()
        return actions

    def evaluate(self, num_episodes=3):
        """
        獨立評估函數：不訓練、低 Epsilon (Greedy)
        """
        total_reward = 0
        for _ in range(num_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                state_tensor = (
                    torch.from_numpy(np.array([state])).float().to(self.device)
                )
                with torch.no_grad():
                    # Evaluation 時不使用 Epsilon-Greedy，直接選最大 Q
                    q_values = self.q_net(state_tensor)
                    action = q_values.argmax(dim=1).item()

                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward

        avg_reward = total_reward / num_episodes
        print(
            f"Evaluation Result: Avg Reward = {avg_reward:.2f} at Step {self.total_steps}"
        )
        wandb.log({"Evaluation Reward": avg_reward, "Step": self.total_steps})
        return avg_reward

    def run(self, max_steps=2000000):
        # 初始狀態
        states, _ = self.envs.reset()
        episode_rewards = [0] * self.num_envs
        episode_counts = 0

        print(f"Started training with {self.num_envs} parallel environments...")

        while self.total_steps < max_steps:
            # 1. 選擇動作 (Batch)
            actions = self.select_actions(states)

            # 2. 環境互動 (Parallel)
            # terminal_states: 該步結束時的狀態 (用於計算 TD Target)
            # next_states_for_agent: 如果 done，這是 reset 後的狀態；否則等於 terminal_states
            terminal_states, rewards, dones, next_states_for_agent, _ = self.envs.step(
                actions
            )

            # 3. 處理每個環境的回傳資料
            for i in range(self.num_envs):
                # 加入 N-Step Buffer
                self.n_step_buffers[i].add(
                    (states[i], actions[i], rewards[i], terminal_states[i], dones[i])
                )
                episode_rewards[i] += rewards[i]

                # 從 N-Step Buffer 取出資料存入 Replay Memory
                transition = self.n_step_buffers[i].get_transition()
                if transition:
                    self.memory.add(transition)

                # 如果該環境結束 (Done)
                if dones[i]:
                    # 清空該環境剩餘的 buffer 資料 (Flush)
                    while len(self.n_step_buffers[i].buffer) > 0:
                        transition = self.n_step_buffers[i].get_transition()
                        if transition:
                            self.memory.add(transition)
                        if len(self.n_step_buffers[i].buffer) > 0:
                            self.n_step_buffers[i].buffer.popleft()

                    # 紀錄 Log
                    episode_counts += 1
                    wandb.log(
                        {
                            "Episode Reward": episode_rewards[i],
                            "Epsilon": self.epsilon,
                            "Total Steps": self.total_steps,
                        }
                    )
                    print(
                        f"Ep Done. Reward: {episode_rewards[i]:.2f}, Steps: {self.total_steps}, Eps: {self.epsilon:.3f}"
                    )
                    episode_rewards[i] = 0

            # 更新狀態
            states = next_states_for_agent
            self.total_steps += self.num_envs  # 每次互動增加了 num_envs 步

            # 4. 訓練模型
            # 隨著資料收集速度變快，我們可能需要每個 step 訓練多次，或者每幾個 step 訓練一次
            if len(self.memory) > self.replay_start_size:
                self.train()

            # 定期評估與儲存
            if self.total_steps % self.eval_freq < self.num_envs:
                self.evaluate()
                self.save_model("latest_model.pt")

    def train(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        mini_batch, indices = self.memory.sample(self.batch_size)
        b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*mini_batch)

        b_states = torch.from_numpy(np.array(b_states)).float().to(self.device)
        b_next_states = (
            torch.from_numpy(np.array(b_next_states)).float().to(self.device)
        )
        b_actions = torch.tensor(b_actions, dtype=torch.int64).to(self.device)
        b_rewards = torch.tensor(b_rewards, dtype=torch.float32).to(self.device)
        b_dones = torch.tensor(b_dones, dtype=torch.float32).to(self.device)

        # Q(s, a)
        current_q = self.q_net(b_states).gather(1, b_actions.unsqueeze(1)).squeeze(1)

        # Double DQN Target
        with torch.no_grad():
            next_actions = self.q_net(b_next_states).max(1)[1]
            next_q = (
                self.target_net(b_next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q = b_rewards + (self.gamma**self.n_step) * next_q * (1 - b_dones)

        # Huber Loss
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)  # 防止梯度爆炸
        self.optimizer.step()

        # 更新 Priorities
        td_errors = (target_q - current_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, np.abs(td_errors))

        if self.train_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            wandb.log({"Loss": loss.item(), "Q Value": current_q.mean().item()})

    def save_model(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")
