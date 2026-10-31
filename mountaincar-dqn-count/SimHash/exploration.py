"""练习 1：固定特征 -> 固定随机投影 -> 哈希桶 -> 跨回合访问计数。

Hash 不依赖正在训练的 Q 网络，否则同一状态的桶会随网络变化，旧计数失效。
本教程只对交互得到的 next_obs 更新一次计数；reset 状态不额外计数。
"""

from collections import defaultdict

import numpy as np


class SimHashCounter:
    def __init__(self, n_bits=32, seed=0, append_constant=False):
        if n_bits <= 0:
            raise ValueError("n_bits 必须为正")
        self.append_constant = append_constant
        feature_dim = 3 if append_constant else 2
        # A: [k, d]；只初始化一次，不参与梯度更新。
        self.projection = np.random.default_rng(seed).standard_normal((n_bits, feature_dim))
        self.counts = defaultdict(int)

        self.offset = np.array([-0.3, 0], dtype=np.float32)
        self.scale = np.array([0.9, 0.07], dtype=np.float32)

    def features(self, obs):
        """TODO 1a：原始 [position, velocity] -> float32 特征 [d]。

        先用 (obs - [-0.3, 0]) / [0.9, 0.07] 缩放到约 [-1, 1]。
        append_constant=True 时再追加一个 1，形成三维特征。
        不使用运行均值方差：hash 坐标系必须保持固定。
        验收：[-0.3, 0] 在默认模式下应接近 [0, 0]。
        """
        # raise NotImplementedError("TODO 1a: 固定特征缩放")
        obs = np.array(obs, dtype=np.float32)
        normalized = (obs - self.offset) / self.scale
        if self.append_constant:
            return np.concatenate([normalized, np.array([1.0], dtype=np.float32)])
        return normalized

    def key(self, obs):
        """TODO 1b：返回可作为字典 key 的 tuple，长度 k，每项为 0 或 1。

        数据流：[k,d] @ [d] -> [k] -> 比较 >= 0 -> tuple。
        这里约定投影恰好为 0 时取 1；不要调用 Python 内置 hash(obs)。
        不要在本方法中增加 counts；查询哈希不等于发生了一次访问。
        """
        # raise NotImplementedError("TODO 1b: sign(A @ features)")
        feat = self.features(obs)
        proj = self.projection @ feat
        bits = (proj >= 0).astype(int)
        return tuple(bits.tolist())

    def observe(self, next_obs):
        """TODO 1c：先将对应桶的 N 加 1，再返回 (1/sqrt(N), N)。

        返回的 bonus 尚未乘 beta，由训练循环统一混合奖励。
        同一状态连续三次应返回 bonus=1, 1/sqrt(2), 1/sqrt(3)。
        counts 跨回合保留；经验回放采样、梯度更新、评估都不调用它。
        """
        # raise NotImplementedError("TODO 1c: 更新访问次数并计算 bonus")
        key = self.key(next_obs)
        self.counts[key] += 1
        n = self.counts[key]
        bonus = 1.0 / float(np.sqrt(n))
        return bonus, n
