"""PPO：实际采样概率 → 更新前一致性检查 → GAE → 连续序列 minibatch 更新。"""

import math
import time

import torch
from torch import nn


class ConsistencyError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def ppo_losses(ratio, advantages, values, returns, entropy, *, clip_coef, vf_coef, ent_coef,
               old_values=None, value_clip_coef=0.0):
    """输入均已排除预热和 padding；value loss 定义为半 MSE。"""
    surrogate = ratio * advantages
    clipped = ratio.clamp(1 - clip_coef, 1 + clip_coef) * advantages
    policy_loss = -torch.minimum(surrogate, clipped).mean()
    value_errors = (values - returns).square()
    if value_clip_coef:
        if old_values is None:
            raise ValueError("价值裁剪需要采样时冻结的 old_values")
        old_values = old_values.detach()
        clipped_values = old_values + (values - old_values).clamp(-value_clip_coef, value_clip_coef)
        value_errors = torch.maximum(value_errors, (clipped_values - returns).square())
    value_loss = 0.5 * value_errors.mean()
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()
    return policy_loss, value_loss, total_loss


@torch.no_grad()
def check_consistency(agent, buffer, config):
    """只能在本轮第一次 optimizer.step 前运行，之后概率变化是 PPO 的正常行为。"""
    maximum, ratio_error, kl_sum, clipped, count = 0.0, 0.0, 0.0, 0, 0
    finite = True
    worst = {f"consistency_worst_{key}": -1 for key in ("env", "time", "action", "position")}
    specs = buffer.sequence_specs(config.sequence_length)
    for batch_index, batch in enumerate(buffer.batches(config.sequence_length, config.sequences_per_minibatch,
                                                      shuffle=False)):
        try:
            new_log_probs, _, _ = agent.evaluate_actions(batch)
        except (ValueError, RuntimeError) as error:
            # Categorical validates logits before it can return NaN log-probs.
            raise ConsistencyError(f"更新前概率检查失败：{error}", {
                "consistency_max_logprob_error": None, "consistency_max_ratio_error": None,
                "consistency_approx_kl": None, "consistency_clip_fraction": 0.0,
                "consistency_finite": 0, "consistency_warning": 1, **worst,
            }) from error
        selected = batch["loss_mask"]
        # 诊断统计在 float64 中累加，网络参数/输入/缓存仍统一为 FP32。
        delta = new_log_probs[selected].double() - batch["old_log_probs"][selected].double()
        if not torch.isfinite(delta).all():
            finite = False
        magnitude = torch.nan_to_num(delta.abs(), nan=float("inf"), posinf=float("inf"))
        batch_max = magnitude.max().item()
        if batch_max > maximum:
            row, col = selected.nonzero()[magnitude.argmax()].tolist()
            env, begin, _ = specs[batch_index * config.sequences_per_minibatch + row]
            first = batch["loss_mask"][row].nonzero()[0].item()
            worst = {"consistency_worst_env": env, "consistency_worst_time": begin + col - first,
                     "consistency_worst_action": batch["actions"][row, col].item(),
                     "consistency_worst_position": batch["position_ids"][row, col].item()}
        maximum = max(maximum, batch_max)
        deviation = torch.expm1(delta)
        finite = finite and bool(torch.isfinite(deviation).all())
        ratio_error = max(ratio_error, deviation.abs().max().item())
        kl_sum += (deviation - delta).sum().item()
        clipped += (deviation.abs() > config.clip_coef).sum().item()
        count += len(delta)
    result = {"consistency_max_logprob_error": maximum if finite else None,
              "consistency_max_ratio_error": ratio_error if finite else None,
              "consistency_approx_kl": kl_sum / count if finite else None,
              "consistency_clip_fraction": clipped / count,
              "consistency_finite": int(finite),
              "consistency_warning": int(not finite or maximum > config.consistency_warning_tolerance),
              **worst}
    if not finite or maximum > config.consistency_tolerance or clipped:
        raise ConsistencyError(
            f"更新前采样/并行概率不一致：max |delta logp|={maximum:.3g}，"
            f"阈值={config.consistency_tolerance:.3g}。请检查动作对齐、预热、位置与参数版本。"
            + (" 检查出现 NaN/Inf。" if not finite else ""), result,
        )
    return result


class PPOTrainer:
    def __init__(self, agent, config):
        self.agent, self.config = agent, config
        # 所有参数（包括残差门控 alpha）正常学习，但不做权重衰减。
        self.optimizer = torch.optim.Adam(
            agent.parameters(), lr=config.learning_rate, eps=1e-5, weight_decay=0.0,
        )
        self.completed_updates = 0
        self.last_diagnostics = None

    def schedule(self, update):
        if not 1 <= update <= self.config.num_updates:
            raise ValueError("update 必须在本次训练的完整 rollout 更新范围内")
        fraction = (update - 1) / (self.config.num_updates - 1) if self.config.num_updates > 1 else 0.0
        def coefficient(initial, final, mode):
            return initial + fraction * (final - initial) if mode == "linear" else initial
        learning_rate = coefficient(self.config.learning_rate, self.config.learning_rate_final, self.config.lr_schedule)
        entropy = coefficient(self.config.ent_coef, self.config.ent_coef_final, self.config.ent_coef_schedule)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate, entropy

    def update(self, buffer, on_consistency_warning=None, *, update_index=None):
        config = self.config
        update_index = self.completed_updates + 1 if update_index is None else update_index
        learning_rate, ent_coef = self.schedule(update_index)
        self.last_diagnostics = None
        # 不覆盖 old_log_probs；它们必须始终是行为策略实际采样时保存的概率。
        metrics = check_consistency(self.agent, buffer, config)
        if metrics["consistency_warning"] and on_consistency_warning is not None:
            on_consistency_warning(metrics)
        buffer.compute_returns_and_advantages(config.gamma, config.gae_lambda, config.normalize_advantage)
        if not torch.isfinite(buffer.advantages).all() or not torch.isfinite(buffer.returns).all():
            raise RuntimeError("GAE 得到了非有限的优势或回报")
        variance = buffer.returns.var(unbiased=False).item()
        metrics["explained_variance"] = (1 - (buffer.returns - buffer.values).var(unbiased=False).item() / variance
                                         if variance > 1e-12 else None)
        metrics.update(buffer.behavior_metrics())
        metrics.update(buffer.training_statistics())
        metrics.update(learning_rate=learning_rate, ent_coef=ent_coef, diagnostic_seconds=0.0)
        probe = None
        if config.diagnostic_interval and (update_index == 1 or update_index == config.num_updates
                                           or update_index % config.diagnostic_interval == 0):
            from training_diagnostics import GradientProbe
            diagnostic_started = time.perf_counter()
            batch = next(buffer.batches(config.sequence_length, config.sequences_per_minibatch, shuffle=False))
            probe = GradientProbe(self.agent, batch, config, ent_coef)
            metrics["diagnostic_seconds"] = time.perf_counter() - diagnostic_started
        self.agent.train()
        totals = {key: 0.0 for key in ("policy_loss", "value_loss", "total_loss", "entropy",
                                      "clip_fraction", "approx_kl", "grad_norm", "gradient_clip_fraction",
                                      "value_clip_fraction")}
        sample_count = 0
        visited = torch.zeros(buffer.num_steps * buffer.num_envs, dtype=torch.bool, device=buffer.device)
        optimizer_steps, early_stop, max_kl, clipped_steps = 0, False, 0.0, 0
        for _ in range(config.update_epochs):
            for batch in buffer.batches(config.sequence_length, config.sequences_per_minibatch, shuffle=True):
                log_probs, entropy, values = self.agent.evaluate_actions(batch)
                selected = batch["loss_mask"]
                log_probs, entropy, values = log_probs[selected], entropy[selected], values[selected]
                old_log_probs = batch["old_log_probs"][selected]
                with torch.no_grad():
                    delta = log_probs.double() - old_log_probs.double()
                    minibatch_kl = (torch.expm1(delta) - delta).mean().item()
                if not math.isfinite(minibatch_kl):
                    raise RuntimeError("PPO minibatch KL 出现 NaN/Inf")
                max_kl = max(max_kl, minibatch_kl)
                if config.target_kl and minibatch_kl > 1.5 * config.target_kl:
                    early_stop = True
                    break
                ratio = torch.exp(log_probs - old_log_probs)
                policy_loss, value_loss, loss = ppo_losses(
                    ratio, batch["advantages"][selected], values, batch["returns"][selected], entropy,
                    clip_coef=config.clip_coef, vf_coef=config.vf_coef, ent_coef=ent_coef,
                    old_values=batch["old_values"][selected], value_clip_coef=config.value_clip_coef,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("PPO 总损失不是有限标量")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.agent.parameters(), config.max_grad_norm, error_if_nonfinite=True,
                )
                self.optimizer.step()
                optimizer_steps += 1
                clipped_steps += int(grad_norm > config.max_grad_norm)
                with torch.no_grad():
                    delta = (log_probs - old_log_probs).double()
                    stats = {
                        "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                        "total_loss": loss.item(), "entropy": entropy.mean().item(),
                        "clip_fraction": ((ratio - 1).abs() > config.clip_coef).float().mean().item(),
                        "approx_kl": (torch.expm1(delta) - delta).mean().item(),
                        "grad_norm": float(grad_norm),
                        "gradient_clip_fraction": float(grad_norm > config.max_grad_norm),
                        "value_clip_fraction": ((values - batch["old_values"][selected]).abs() > config.value_clip_coef
                                                ).float().mean().item() if config.value_clip_coef else 0.0,
                    }
                count = int(selected.sum().item())
                for name, value in stats.items():
                    if not math.isfinite(value):
                        raise RuntimeError(f"训练指标 {name} 出现 NaN/Inf")
                    totals[name] += value * count
                sample_count += count
                visited[batch["sample_ids"][selected]] = True
            if early_stop:
                break
        metrics.update({name: value / sample_count if sample_count else 0.0 for name, value in totals.items()})
        metrics.update(optimizer_steps=optimizer_steps, ppo_early_stop=int(early_stop),
                       max_minibatch_kl=max_kl,
                       planned_optimizer_steps=config.update_epochs * math.ceil(
                           len(buffer.sequence_specs(config.sequence_length)) / config.sequences_per_minibatch),
                       accepted_sample_visits=sample_count,
                       planned_sample_visits=config.update_epochs * visited.numel(),
                       sample_reuse_count=sample_count / visited.numel(),
                       sample_visit_fraction=sample_count / (config.update_epochs * visited.numel()),
                       unique_sample_coverage=visited.float().mean().item())
        metrics["gradient_clip_fraction"] = clipped_steps / optimizer_steps if optimizer_steps else 0.0
        if probe is not None:
            diagnostic_started = time.perf_counter()
            self.last_diagnostics = probe.finish(self.agent)
            metrics["diagnostic_seconds"] += time.perf_counter() - diagnostic_started
        self.completed_updates = update_index
        return metrics
