import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
from functools import partial
import time
from mpi4py import MPI

gym.register_envs(ale_py)

# ==========================================
# 1. 模型架構修正 (Nature DQN Architecture)
# ==========================================
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # 使用標準的 DeepMind Atari 架構: Strided Convolutions, 無 BN/Pool
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 關鍵修正：將 uint8 (0-255) 正規化為 float (0-1)
        x = x / 255.0
        return self.network(x)

# ==========================================
# 2. 預處理與環境包裝 (Preprocessing)
# ==========================================
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

# ==========================================
# 3. 平行化環境架構 (MPI)
# ==========================================
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
            
            if cmd == 'step':
                action = data
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                real_next_state = next_state
                
                if done:
                    reset_state, _ = env.reset()
                    # 回傳: (next_state, reward, done, reset_state, info)
                    comm.send((real_next_state, reward, done, reset_state, info), dest=0)
                else:
                    comm.send((real_next_state, reward, done, None, info), dest=0)
            
            elif cmd == 'reset':
                state, info = env.reset()
                comm.send((state, info), dest=0)
            
            elif cmd == 'close':
                env.env.close()
                break
    except Exception as e:
        print(f"Worker {rank} error: {e}")
    finally:
        if hasattr(env, 'env'):
            env.env.close()

class MPIVectorEnv:
    """
    管理多個 MPI Worker 的介面
    """
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        
        if self.size < num_envs + 1:
            raise RuntimeError(f"MPI Size {self.size} is not enough for 1 Master + {num_envs} Workers.")

    def reset(self):
        # 發送 Reset 指令給所有 Worker (Rank 1 ~ num_envs)
        for i in range(self.num_envs):
            self.comm.send(('reset', None), dest=i+1)
        
        results = []
        for i in range(self.num_envs):
            results.append(self.comm.recv(source=i+1))
            
        states, infos = zip(*results)
        return np.stack(states), infos

    def step(self, actions):
        # 回退到 Blocking Communication 以避免記憶體錯誤 (free(): invalid next size)
        # Python 物件的 Non-blocking 傳輸 (isend/irecv) 需要非常小心地管理物件生命週期
        # 簡單的 isend/irecv 在高頻率呼叫下容易導致 GC 與 MPI 底層衝突
        
        # 發送 Step 指令與 Action
        for i, action in enumerate(actions):
            self.comm.send(('step', action), dest=i+1)
        
        results = []
        for i in range(self.num_envs):
            results.append(self.comm.recv(source=i+1))
        
        # results: list of (next_state, reward, done, reset_state, info)
        next_states, rewards, dones, reset_states, infos = zip(*results)
        
        next_states = np.stack(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # 處理 Auto-Reset 的狀態回傳
        current_obs_for_agent = np.array([
            reset_states[i] if dones[i] else next_states[i] 
            for i in range(self.num_envs)
        ])
        
        return next_states, rewards, dones, current_obs_for_agent, infos

    def close(self):
        for i in range(self.num_envs):
            self.comm.send(('close', None), dest=i+1)

# ==========================================
# 4. 資料結構 (Buffer & N-Step)
# ==========================================
class NStepBuffer:
    def __init__(self, n_step=3, gamma=0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
    
    def add(self, transition):
        self.buffer.append(transition)
    
    def get_transition(self):
        if len(self.buffer) < self.n_step:
            return None
        
        state, action, _, _, _ = self.buffer[0]
        n_step_return = 0
        
        # 計算 N-Step Return
        for i, (_, _, reward, _, done) in enumerate(self.buffer):
            n_step_return += (self.gamma ** i) * reward
            if done:
                # 如果中間就結束了，Next State 是該步的 Next State
                next_state, _, _, _, final_done = self.buffer[i]
                return (state, action, n_step_return, next_state, final_done)
        
        # 正常 N 步
        _, _, _, next_state, done = self.buffer[-1]
        return (state, action, n_step_return, next_state, done)
    
    def reset(self):
        self.buffer.clear()

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0  # 初始最大優先權

    def __len__(self):
        return len(self.buffer)

    def add(self, transition):
        # 新樣本直接給予最大優先權，避免在收集時計算 TD Error
        priority = self.max_priority
        
        # 優化記憶體：確保儲存的是 uint8 且是獨立的 copy (避免 view 參照到大 array)
        state, action, reward, next_state, done = transition
        state = np.array(state, dtype=np.uint8, copy=True)
        next_state = np.array(next_state, dtype=np.uint8, copy=True)
        transition = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        probs = self.priorities[:buffer_size] / self.priorities[:buffer_size].sum()
        indices = np.random.choice(buffer_size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

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
        self.gamma = args.discount_factor
        self.n_step = args.n_step
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.eval_freq = args.eval_freq

        self.memory = PrioritizedReplayBuffer(args.memory_size)
        # 為每個環境建立獨立的 N-Step Buffer
        self.n_step_buffers = [NStepBuffer(n_step=self.n_step, gamma=self.gamma) for _ in range(self.num_envs)]

        self.total_steps = 0
        self.train_count = 0
        self.target_update_freq = args.target_update_frequency
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.best_reward = -float('inf')

    def select_actions(self, states):
        """
        Batch Action Selection
        states: numpy array of shape (num_envs, 4, 84, 84)
        """
        if random.random() < self.epsilon:
            return [random.randint(0, self.num_actions - 1) for _ in range(self.num_envs)]
        
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
                state_tensor = torch.from_numpy(np.array([state])).float().to(self.device)
                with torch.no_grad():
                    # Evaluation 時不使用 Epsilon-Greedy，直接選最大 Q
                    q_values = self.q_net(state_tensor)
                    action = q_values.argmax(dim=1).item()
                
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
        
        avg_reward = total_reward / num_episodes
        print(f"Evaluation Result: Avg Reward = {avg_reward:.2f} at Step {self.total_steps}")
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
            terminal_states, rewards, dones, next_states_for_agent, _ = self.envs.step(actions)
            
            # 3. 處理每個環境的回傳資料
            for i in range(self.num_envs):
                # 加入 N-Step Buffer
                self.n_step_buffers[i].add((states[i], actions[i], rewards[i], terminal_states[i], dones[i]))
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
                    wandb.log({
                        "Episode Reward": episode_rewards[i],
                        "Epsilon": self.epsilon,
                        "Total Steps": self.total_steps
                    })
                    print(f"Ep Done. Reward: {episode_rewards[i]:.2f}, Steps: {self.total_steps}, Eps: {self.epsilon:.3f}")
                    episode_rewards[i] = 0

            # 更新狀態
            states = next_states_for_agent
            self.total_steps += self.num_envs # 每次互動增加了 num_envs 步

            # 4. 訓練模型
            # 隨著資料收集速度變快，我們可能需要每個 step 訓練多次，或者每幾個 step 訓練一次
            if len(self.memory) > args.replay_start_size:
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
        b_next_states = torch.from_numpy(np.array(b_next_states)).float().to(self.device)
        b_actions = torch.tensor(b_actions, dtype=torch.int64).to(self.device)
        b_rewards = torch.tensor(b_rewards, dtype=torch.float32).to(self.device)
        b_dones = torch.tensor(b_dones, dtype=torch.float32).to(self.device)

        # Q(s, a)
        current_q = self.q_net(b_states).gather(1, b_actions.unsqueeze(1)).squeeze(1)

        # Double DQN Target
        with torch.no_grad():
            next_actions = self.q_net(b_next_states).max(1)[1]
            next_q = self.target_net(b_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = b_rewards + (self.gamma ** self.n_step) * next_q * (1 - b_dones)

        # Huber Loss
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0) # 防止梯度爆炸
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

if __name__ == "__main__":
    # 限制 CPU 使用率
    torch.set_num_threads(4)
    cv2.setNumThreads(1)

    # MPI 初始化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results_parallel")
    parser.add_argument("--wandb-run-name", type=str, default="parallel_dqn_pong")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=8)  # 平行環境數量
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99998)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--target-update-frequency", type=int, default=2000)
    parser.add_argument("--replay-start-size", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=20000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=100000)
    args = parser.parse_args()

    if rank == 0:
        # Master Process
        print(f"Master process started. MPI Size: {size}")
        if size < args.num_envs + 1:
            print(f"Error: MPI size ({size}) is too small for {args.num_envs} environments + 1 master.")
            print(f"Please run with: mpirun -n {args.num_envs + 1} python main_mpi.py")
            comm.Abort()
        
        wandb.init(project="DLP-Lab5-Parallel", name=args.wandb_run_name, config=args)
        
        agent = MultiStepDQNAgent(env_name="ALE/Pong-v5", args=args)
        try:
            agent.run(max_steps=args.max_steps)
        except KeyboardInterrupt:
            print("Stopping training...")
        finally:
            agent.envs.close()
            
    else:
        # Worker Process
        # Rank 1 對應 env_index 0, Rank 2 對應 env_index 1, ...
        # 確保只運行 num_envs 個 Worker
        if rank <= args.num_envs:
            run_worker(rank, "ALE/Pong-v5", args.seed)
        else:
            print(f"Rank {rank} is idle (extra process).")