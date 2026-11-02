import unittest

import torch

from rl_shared import TransformerBackbone, TransformerConfig


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CUDATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(321)

    def test_autocast_streaming_gradients_and_float_masks(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                config = TransformerConfig(hidden_dim=64, num_heads=2, context_length=5, alpha_init=0.5)
                model = TransformerBackbone(config).cuda()
                x = torch.randn(3, 17, 64, device="cuda")
                starts = torch.zeros(3, 17, device="cuda", dtype=torch.bool)
                starts[1, 7] = True
                cache = model.create_cache(3, dtype=dtype)
                with torch.autocast("cuda", dtype=dtype):
                    parallel = model(x, episode_starts=starts)
                    outputs = [model.step(x[:, t:t + 1], cache, episode_starts=starts[:, t])
                               for t in range(x.shape[1])]
                    # Tensor cores may accumulate GEMM and SDPA in different
                    # orders for T=1 and T>1; require dtype-appropriate agreement.
                    torch.testing.assert_close(torch.cat(outputs, 1), parallel, rtol=2e-2, atol=2e-2)
                    float_mask = torch.zeros(17, 17, device="cuda")
                    float_mask[:, 0] = -torch.inf
                    trained = model(x, attn_mask=float_mask)
                    loss = trained.square().mean()
                loss.backward()
                self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
                current = torch.randn(3, 1, 64, device="cuda", requires_grad=True)
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype):
                    step_loss = model.step(current, cache, enable_grad=True).square().mean()
                    model.step(torch.randn_like(current), cache)
                    cache.reset()
                step_loss.backward()
                self.assertTrue(current.grad.isfinite().all())

    def test_half_model_rotary_and_cache_dtype_checks(self):
        for dtype in (torch.bfloat16, torch.float16):
            model = TransformerBackbone(TransformerConfig(hidden_dim=64, alpha_init=0.4)).cuda().to(dtype)
            x = torch.randn(2, 4, 64, device="cuda", dtype=dtype)
            cache = model.create_cache(2)
            torch.testing.assert_close(torch.cat([model.step(x[:, i:i + 1], cache) for i in range(4)], 1),
                                       model(x), rtol=3e-2, atol=3e-2)
        model = TransformerBackbone(TransformerConfig(hidden_dim=64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16), self.assertRaisesRegex(ValueError, "projected K/V"):
            model.step(x.float()[:, :1], model.create_cache(2))
        with self.assertRaises(ValueError):
            model.step(x.float()[:, :1], model.create_cache(2, device="cpu"))

    def test_compile_fullgraph_forward_backward_with_packed_window_mask(self):
        config = TransformerConfig(hidden_dim=32, num_heads=2, num_layers=1, context_length=4, alpha_init=0.5)
        model = TransformerBackbone(config).cuda()
        x = torch.randn(2, 9, 32, device="cuda", requires_grad=True)
        starts = torch.zeros(2, 9, dtype=torch.bool, device="cuda")
        starts[0, 4] = True
        valid = torch.arange(9, device="cuda")[None, :] < torch.tensor([9, 7], device="cuda")[:, None]
        eager = model(x, episode_starts=starts, valid_mask=valid)
        eager.square().mean().backward()
        reference_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        x.grad = None
        compiled = torch.compile(model, fullgraph=True)
        actual = compiled(x, episode_starts=starts, valid_mask=valid)
        actual.square().mean().backward()
        torch.testing.assert_close(actual, eager, rtol=2e-4, atol=3e-5)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, reference_grads[name], rtol=5e-4, atol=5e-5)


if __name__ == "__main__":
    unittest.main()
