"""分阶段自检；这里只给可观察行为，不提供参考实现。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from dqn import compute_loss, select_action
from exploration import SimHashCounter
from model import ResidualQNetwork
from replay import ReplayBuffer


def check_model():
    model = ResidualQNetwork()
    obs = torch.tensor([[-0.5, 0.0], [-0.4, 0.02]])
    q = model(obs)
    assert q.shape == (2, 3) and torch.isfinite(q).all()
    q.square().mean().backward()
    assert model.q_head[-1].weight.grad is not None
    clone = ResidualQNetwork(**model.model_config)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone(obs), q.detach())


def check_hash():
    obs = np.array([-0.5, 0.01], dtype=np.float32)
    counter = SimHashCounter(seed=7)
    np.testing.assert_allclose(counter.features([-0.3, 0]), [0, 0], atol=1e-6)
    other = SimHashCounter(seed=7)
    key = counter.key(obs)
    assert isinstance(key, tuple) and len(key) == 32 and set(key) <= {0, 1}
    assert key == other.key(obs)
    assert len(counter.counts) == 0, "key() 不应修改访问计数"
    for n in range(1, 4):
        bonus, count = counter.observe(obs)
        assert count == n and np.isclose(bonus, 1 / np.sqrt(n))
    assert len(counter.counts) == 1
    augmented = SimHashCounter(append_constant=True)
    assert augmented.features(obs).shape == (3,)
    assert augmented.features(obs)[-1] == 1
    assert len(augmented.key(obs)) == 32
    # 人为替换成可手算投影，避免仅检查重复性却漏掉投影公式。
    counter.projection = np.eye(2)
    assert counter.key([0.6, -0.07]) == (1, 0)


def check_replay():
    replay = ReplayBuffer(2, seed=0)
    obs = np.array([-0.5, 0], dtype=np.float32)
    next_obs = np.array([-0.49, 0.01], dtype=np.float32)
    replay.push(obs, 2, -0.9, next_obs, False)
    obs[:] = 99
    next_obs[:] = 99
    batch = replay.sample(1, torch.device("cpu"))
    torch.testing.assert_close(batch["obs"], torch.tensor([[-0.5, 0.0]]))
    torch.testing.assert_close(batch["next_obs"], torch.tensor([[-0.49, 0.01]]))
    assert batch["actions"].shape == (1, 1) and batch["actions"].dtype == torch.int64
    assert batch["rewards"].shape == (1,) and batch["rewards"].dtype == torch.float32
    assert batch["terminated"].shape == (1,) and batch["terminated"].dtype == torch.bool
    replay.push(np.zeros(2), 0, -1, np.ones(2), True)
    replay.push(np.zeros(2), 1, -2, np.ones(2), False)
    assert len(replay) == 2
    assert set(replay.sample(2, "cpu")["actions"].flatten().tolist()) == {0, 1}


def check_dqn():
    # 用常量 Q 构造可手算场景，不依赖网络是否已学会 MountainCar。
    policy = nn.Linear(2, 3)
    target = nn.Linear(2, 3)
    with torch.no_grad():
        policy.weight.zero_()
        policy.bias.copy_(torch.tensor([1., 2., 3.]))
        target.weight.zero_()
        target.bias.copy_(torch.tensor([2., 4., 6.]))
    batch = {"obs": torch.zeros(2, 2), "next_obs": torch.zeros(2, 2),
             "actions": torch.tensor([[0], [2]]), "rewards": torch.tensor([-1., -1.]),
             "terminated": torch.tensor([True, False])}
    # gamma=.5: y=[-1,2]，Q(s,a)=[1,3]；Huber=(1.5+.5)/2=1。
    loss = compute_loss(policy, target, batch, gamma=0.5)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert policy.bias.grad is not None
    assert all(p.grad is None for p in target.parameters()), "target 应位于 no_grad 内"
    rng = np.random.default_rng(0)
    assert select_action(policy, np.zeros(2), 0., rng, 3, "cpu") == 2
    draws = [select_action(policy, np.zeros(2), 1., rng, 3, "cpu") for _ in range(100)]
    assert all(isinstance(a, int) for a in draws) and set(draws) == {0, 1, 2}
    # 全部 terminated 也应该能更新，避免 cat([]) 一类边界问题。
    batch["terminated"][:] = True
    torch.testing.assert_close(compute_loss(policy, target, batch, 0.5), torch.tensor(2.5))


if __name__ == "__main__":
    checks = {"model": check_model, "hash": check_hash,
              "replay": check_replay, "dqn": check_dqn}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*checks, "all"])
    args = parser.parse_args()
    selected = checks if args.stage == "all" else {args.stage: checks[args.stage]}
    for name, check in selected.items():
        try:
            check()
        except NotImplementedError as error:
            print(f"待填空 [{name}]：{error}")
            sys.exit(2)
        print(f"PASS [{name}]")
