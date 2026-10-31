"""教学起点，尚未验证收敛；先用 beta=0 跑通普通 DQN，再加 bonus。"""

from dataclasses import dataclass


@dataclass
class Config:
    env_id: str = "MountainCar-v0"
    seed: int = 0
    total_steps: int = 100_000
    capacity: int = 50_000
    batch_size: int = 128
    learning_starts: int = 2_000
    gamma: float = 0.99
    lr: float = 3e-4
    target_update_interval: int = 1_000  # 按环境步数做硬更新
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 30_000  # 线性衰减
    n_bits: int = 32
    beta: float = 0.1
    ext_reward_scale: float = 1.0
    append_constant: bool = False  # 进阶对照：给 hash 特征追加常数 1
