import torch
import torch.nn as nn


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
