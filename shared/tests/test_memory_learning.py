"""Small supervised memory task; this is not an RL convergence benchmark."""

import json
import unittest

import torch
from torch import nn

from rl_shared import TransformerBackbone, TransformerConfig, init_head


def memory_batch(generator, n=64):
    """Only the first token reveals the label; the last token asks for it."""
    device = generator.device
    labels = torch.randint(2, (n,), generator=generator, device=device)
    observations = torch.zeros(n, 16, 4, device=device)
    observations[:, :, 3] = torch.randn(n, 16, generator=generator, device=device)
    observations[:, 0, 0] = labels.float() * 2 - 1
    observations[:, 0, 1] = 1  # cue marker
    observations[:, -1, 2] = 1  # query marker
    return observations, labels


class MemoryLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def check_learning(self, device, dtype, seed):
        torch.manual_seed(seed)
        encoder = nn.Linear(4, 32).to(device)
        config = TransformerConfig(hidden_dim=32, num_heads=2, num_layers=2,
                                   context_length=16, ffn_dim=64)
        backbone = TransformerBackbone(config).to(device)
        head = init_head(nn.Linear(32, 2), gain=0.01).to(device)
        model = nn.Sequential(encoder, backbone, head)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler(device, enabled=dtype == torch.float16, init_scale=1024.0)
        train_rng = torch.Generator(device=device).manual_seed(1000 + seed)
        eval_rng = torch.Generator(device=device).manual_seed(9000 + seed)

        def amp():
            return torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32)

        for update in range(300):
            observations, labels = memory_batch(train_rng)
            optimizer.zero_grad(set_to_none=True)
            with amp():
                loss = nn.functional.cross_entropy(model(observations)[:, -1], labels)
            self.assertTrue(loss.isfinite())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if update == 0:
                self.assertTrue(all(p.grad is not None for p in model.parameters()))
                self.assertGreater(backbone.blocks[0].attention.qkv.weight.grad.norm().item(), 0)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        with torch.no_grad(), amp():
            observations, labels = memory_batch(eval_rng, n=1024)
            logits = model(observations)[:, -1]
            accuracy = (logits.argmax(-1) == labels).float().mean().item()
            # Same trained network, but no access to the cue or other history.
            current_logits = model(observations[:, -1:])[:, -1]
            current_accuracy = (current_logits.argmax(-1) == labels).float().mean().item()
            cache = backbone.create_cache(64, dtype=dtype)
            for t in range(16):
                hidden = backbone.step(encoder(observations[:64, t:t + 1]), cache)
            stream_logits = head(hidden[:, 0])
            half = dtype != torch.float32
            torch.testing.assert_close(stream_logits, logits[:64], rtol=2e-2 if half else 2e-5,
                                       atol=2e-2 if half else 3e-6)

        result = dict(device=device, dtype=str(dtype), seed=seed, updates=300,
                      loss=loss.item(), accuracy=accuracy, current_only_accuracy=current_accuracy)
        print("memory_learning " + json.dumps(result), flush=True)
        self.assertGreaterEqual(accuracy, 0.95, result)
        self.assertGreaterEqual(current_accuracy, 0.40, result)
        self.assertLessEqual(current_accuracy, 0.60, result)
        self.assertTrue(all(p.isfinite().all() for p in model.parameters()))

    def test_cpu_default_gates_learn_memory(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                self.check_learning("cpu", torch.float32, seed)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_default_gates_learn_memory(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for seed in (0, 1, 2):
                with self.subTest(dtype=dtype, seed=seed):
                    self.check_learning("cuda", dtype, seed)


if __name__ == "__main__":
    unittest.main()
