from collections import deque

import numpy as np


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
            n_step_return += (self.gamma**i) * reward
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
