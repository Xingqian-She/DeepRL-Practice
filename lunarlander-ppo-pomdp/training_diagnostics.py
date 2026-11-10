"""Periodic eager gradient probes. No random sampling, .grad writes, or persisted tensors."""

import math

import torch

from ppo import ppo_losses


class GradientProbe:
    def __init__(self, agent, batch, config, ent_coef):
        self.before = {name: parameter.detach().clone() for name, parameter in agent.named_parameters()}
        policy_ids = {id(p) for p in agent.policy_parameters()}
        value_ids = {id(p) for p in agent.value_parameters()}
        self.groups = {name: ("shared" if id(p) in policy_ids & value_ids
                              else "policy_only" if id(p) in policy_ids else "value_only")
                       for name, p in agent.named_parameters()}
        parameters = list(agent.parameters())
        # forward_sequence explicitly bypasses the compiled PPO path.
        logits, values = agent.forward_sequence(
            batch["observations"], batch["previous_actions"], episode_starts=batch["episode_starts"],
            valid_mask=batch["valid_mask"], position_ids=batch["position_ids"],
        )
        distribution = torch.distributions.Categorical(logits=logits)
        selected = batch["loss_mask"]
        ratio = (distribution.log_prob(batch["actions"])[selected] - batch["old_log_probs"][selected]).exp()
        entropy = distribution.entropy()[selected]
        policy, value, _ = ppo_losses(
            ratio, batch["advantages"][selected], values[selected], batch["returns"][selected], entropy,
            clip_coef=config.clip_coef, vf_coef=config.vf_coef, ent_coef=ent_coef,
            old_values=batch["old_values"][selected], value_clip_coef=config.value_clip_coef,
        )
        pg = torch.autograd.grad(policy - ent_coef * entropy.mean(), parameters, allow_unused=True, retain_graph=True)
        vg = torch.autograd.grad(config.vf_coef * value, parameters, allow_unused=True)
        def square_norm(gradients):
            return sum(g.double().square().sum().item() for g in gradients if g is not None)
        shared = [index for index, p in enumerate(parameters) if id(p) in policy_ids & value_ids]
        left = [pg[i] for i in shared]
        right = [vg[i] for i in shared]
        denominator = math.sqrt(square_norm(left) * square_norm(right))
        dot = sum((pg[i].double() * vg[i].double()).sum().item()
                  for i in shared if pg[i] is not None and vg[i] is not None)
        self.result = {"policy_objective_grad_norm": math.sqrt(square_norm(pg)),
                       "value_objective_grad_norm": math.sqrt(square_norm(vg)),
                       "shared_gradient_cosine": max(-1.0, min(1.0, dot / denominator)) if denominator else None,
                       "probe_samples": int(selected.sum().item())}

    @torch.no_grad()
    def finish(self, agent):
        totals = {group: [0.0, 0.0] for group in ("total", "shared", "policy_only", "value_only")}
        for name, parameter in agent.named_parameters():
            before = self.before[name].double()
            delta = (parameter.detach().double() - before).square().sum().item()
            norm = before.square().sum().item()
            for group in ("total", self.groups[name]):
                totals[group][0] += delta
                totals[group][1] += norm
        for group, (delta, norm) in totals.items():
            self.result[f"{group}_parameter_delta_norm"] = math.sqrt(delta)
            self.result[f"{group}_relative_parameter_delta"] = math.sqrt(delta / norm) if norm else None
        return self.result
