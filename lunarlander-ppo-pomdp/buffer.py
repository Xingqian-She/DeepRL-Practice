"""rollout 按 (时间 T, 环境 N) 保存，训练时切成含预热前缀的连续序列。"""

import torch

from sensors import FAILURE_INDICES, OBSERVATION_DIM, SENSOR_NAMES


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, device, prefix, warmup_length):
        self.num_steps, self.num_envs = num_steps, num_envs
        self.device, self.warmup_length = device, warmup_length
        self.prefix = [token.clone() for token in prefix]
        self.observations = torch.zeros(num_steps, num_envs, OBSERVATION_DIM, device=device)
        self.sampling_logits = torch.zeros(num_steps, num_envs, 4, device=device)
        shape = (num_steps, num_envs)
        for name in ("previous_actions", "position_ids", "actions"):
            setattr(self, name, torch.zeros(shape, dtype=torch.long, device=device))
        for name in ("episode_starts", "terminated", "truncated"):
            setattr(self, name, torch.zeros(shape, dtype=torch.bool, device=device))
        for name in ("old_log_probs", "values", "rewards", "next_values", "raw_advantages", "advantages", "returns"):
            setattr(self, name, torch.zeros(shape, device=device))
        self.step = 0
        self.episodes = []
        self.sensor_probability_start = 0.0
        self.sensor_probability_end = 0.0

    def add(self, token, actions, log_probs, values, rewards, terminated, truncated, *, logits):
        if self.step >= self.num_steps:
            raise RuntimeError("rollout 已满")
        for name in ("observations", "previous_actions", "episode_starts", "position_ids"):
            getattr(self, name)[self.step].copy_(getattr(token, name))
        for name, value in (("actions", actions), ("old_log_probs", log_probs), ("values", values),
                            ("rewards", rewards), ("terminated", terminated), ("truncated", truncated)):
            getattr(self, name)[self.step].copy_(value.detach())
        self.sampling_logits[self.step].copy_(logits.detach())
        self.step += 1

    @torch.no_grad()
    def compute_returns_and_advantages(self, gamma, gae_lambda, normalize_advantage=True):
        if self.step != self.num_steps:
            raise RuntimeError("先收满 rollout，再计算 GAE")
        next_advantage = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            bootstrap_mask = (~self.terminated[t]).float()
            trace_mask = (~(self.terminated[t] | self.truncated[t])).float()
            delta = self.rewards[t] + gamma * bootstrap_mask * self.next_values[t] - self.values[t]
            advantage = delta + gamma * gae_lambda * trace_mask * next_advantage
            self.raw_advantages[t] = advantage
            self.returns[t] = advantage + self.values[t]
            next_advantage = advantage
        # 只标准化策略优势，不改变 critic 的回报目标；整轮只做一次。
        self.advantages.copy_(self.raw_advantages)
        if normalize_advantage:
            self.advantages.copy_((self.advantages - self.advantages.mean()) /
                                  (self.advantages.std(unbiased=False) + 1e-8))

    def training_statistics(self):
        result = {}
        for name, tensor in (("raw_advantage", self.raw_advantages), ("advantage", self.advantages),
                             ("value_prediction", self.values), ("return_target", self.returns),
                             ("reward", self.rewards)):
            data = tensor[:self.step].double()
            result.update({f"{name}_{key}": value.item() for key, value in (
                ("mean", data.mean()), ("std", data.std(unbiased=False)),
                ("min", data.min()), ("max", data.max()))})
        return result

    def sensor_metrics(self):
        result = {"sensor_failure_probability_start": self.sensor_probability_start,
                  "sensor_failure_probability_end": self.sensor_probability_end}
        for index in FAILURE_INDICES:
            result[f"sensor_missing_rate_{SENSOR_NAMES[index]}"] = (
                1 - self.observations[:self.step, :, 8 + index].mean().item()
            )
        return result

    def behavior_metrics(self):
        distribution = torch.distributions.Categorical(logits=self.sampling_logits[:self.step])
        result = {f"action_fraction_{i}": (self.actions[:self.step] == i).float().mean().item()
                  for i in range(4)}
        result.update(main_engine_probability=distribution.probs[..., 2].mean().item(),
                      behavior_entropy=distribution.entropy().mean().item())
        return result

    def sequence_specs(self, sequence_length):
        return [(env, begin, min(begin + sequence_length, self.num_steps))
                for env in range(self.num_envs)
                for begin in range(0, self.num_steps, sequence_length)]

    def collate(self, specs):
        """将不同长度的序列右侧补零，loss_mask 只选本轮真实学习区。

        预热最多来自两个地方：rollout 前保留的原始历史、当前 rollout 的更早时间。
        一条样本不会读取 end 之后的观测，也不会从别的环境借历史。
        """
        rows = []
        for env, begin, end in specs:
            prefix_length = len(self.prefix)
            left = max(0, prefix_length + begin - self.warmup_length)
            row = {}
            for name in ("observations", "previous_actions", "episode_starts", "position_ids"):
                current = getattr(self, name)[:end, env]
                if self.prefix:
                    preceding = torch.stack([getattr(token, name)[env] for token in self.prefix])
                    current = torch.cat((preceding, current), dim=0)
                row[name] = current[left:]
            row["loss_begin"] = prefix_length + begin - left
            row["source"] = (env, begin, end)
            rows.append(row)

        batch_size = len(rows)
        length = max(len(row["observations"]) for row in rows)
        batch = {"observations": torch.zeros(batch_size, length, OBSERVATION_DIM, device=self.device)}
        for name in ("previous_actions", "position_ids", "actions"):
            fill = -1 if name == "previous_actions" else 0
            batch[name] = torch.full((batch_size, length), fill, dtype=torch.long, device=self.device)
        for name in ("episode_starts", "valid_mask", "loss_mask"):
            batch[name] = torch.zeros(batch_size, length, dtype=torch.bool, device=self.device)
        batch["sample_ids"] = torch.full((batch_size, length), -1, dtype=torch.long, device=self.device)
        for name in ("old_log_probs", "old_values", "advantages", "returns"):
            batch[name] = torch.zeros(batch_size, length, device=self.device)
        for i, row in enumerate(rows):
            count, loss_begin = len(row["observations"]), row["loss_begin"]
            for name in ("observations", "previous_actions", "episode_starts", "position_ids"):
                batch[name][i, :count] = row[name]
            batch["valid_mask"][i, :count] = True
            batch["loss_mask"][i, loss_begin:count] = True
            env, begin, end = row["source"]
            batch["sample_ids"][i, loss_begin:count] = torch.arange(begin, end, device=self.device) * self.num_envs + env
            for name in ("actions", "old_log_probs", "old_values", "advantages", "returns"):
                source = "values" if name == "old_values" else name
                batch[name][i, loss_begin:count] = getattr(self, source)[begin:end, env]
        return batch

    def batches(self, sequence_length, sequences_per_minibatch, *, shuffle):
        specs = self.sequence_specs(sequence_length)
        order = torch.randperm(len(specs)).tolist() if shuffle else list(range(len(specs)))
        for begin in range(0, len(order), sequences_per_minibatch):
            yield self.collate([specs[i] for i in order[begin:begin + sequences_per_minibatch]])
