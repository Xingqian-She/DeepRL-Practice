"""Run after installing ./shared; synthetic features, not an RL learning result."""

import argparse

import torch
from torch import nn

from rl_shared import TransformerBackbone, TransformerConfig, init_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    config = TransformerConfig(hidden_dim=128, context_length=8)
    backbone = TransformerBackbone(config).to(device)
    actor = init_head(nn.Linear(128, 6), gain=0.01).to(device)
    critic = init_head(nn.Linear(128, 1), gain=1.0).to(device)

    encoded = torch.randn(3, 20, 128, device=device)
    starts = torch.zeros(3, 20, dtype=torch.bool, device=device)
    starts[1, 7] = True
    starts[2, 11] = True
    valid = torch.arange(20, device=device)[None, :] < torch.tensor([20, 17, 20], device=device)[:, None]
    hidden = backbone(encoded, episode_starts=starts, valid_mask=valid)
    # A synthetic scalar loss only illustrates how heads stay external.
    loss = critic(hidden)[valid].square().mean() + actor(hidden)[valid].square().mean()
    loss.backward()
    nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
    backbone.zero_grad(set_to_none=True)
    print("parallel shape:", tuple(hidden.shape), "synthetic loss:", round(loss.item(), 6))

    cache = backbone.create_cache(3)
    with torch.no_grad():
        stream = torch.cat([backbone.step(encoded[:, t:t + 1], cache, episode_starts=starts[:, t])
                            for t in range(encoded.shape[1])], dim=1)
        reference = backbone(encoded, episode_starts=starts)
    torch.testing.assert_close(stream, reference, atol=1e-5, rtol=2e-5)
    print("stream matches parallel; next positions:", cache.next_positions.tolist())

    current = torch.randn(3, 1, 128, device=device, requires_grad=True)
    with torch.no_grad():  # step's local flag re-enables the backbone graph.
        current_hidden = backbone.step(current, cache, enable_grad=True)
    current_loss = critic(current_hidden).square().mean()
    # Later interaction/reset cannot invalidate current_hidden's saved K/V.
    backbone.step(torch.randn_like(current), cache)
    cache.reset(torch.tensor([True, False, False], device=device))
    current_loss.backward()
    print("single-step input gradient norm:", current.grad.norm().item())

    optimizer = torch.optim.AdamW(backbone.parameters(), lr=3e-4)
    nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
    optimizer.step()
    cache.reset()  # Parameters changed: rebuild context before the next rollout if needed.
    print("cache reset after parameter update; bytes:", cache.nbytes)


if __name__ == "__main__":
    main()
