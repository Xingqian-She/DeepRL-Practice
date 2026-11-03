"""Float bias offsets must not overflow or change relative attention weights."""

import unittest

import torch

from rl_shared import TransformerBackbone, TransformerConfig


class FloatMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_case(self, device, *, model_dtype=torch.float32):
        torch.manual_seed(321)
        config = TransformerConfig(hidden_dim=32, num_heads=2, num_layers=1,
                                   context_length=4, ffn_dim=64, alpha_init=0.5)
        model = TransformerBackbone(config).to(device=device, dtype=model_dtype)
        x = torch.randn(3, 9, 32, device=device, dtype=model_dtype)
        starts = torch.zeros(3, 9, dtype=torch.bool, device=device)
        starts[0, 4] = True
        starts[1, 5] = True
        valid = torch.arange(9, device=device)[None, :] < torch.tensor([9, 7, 0], device=device)[:, None]
        x[~valid] = float("nan")
        x.requires_grad_()
        bias = (torch.arange(9, device=device, dtype=model_dtype) * 0.25).expand(9, 9).clone()
        bias[:, 1] = -torch.inf
        bias[3] = -torch.inf
        return model, x, dict(episode_starts=starts, valid_mask=valid), bias

    def evaluate(self, model, x, kwargs, bias, dtype):
        model.zero_grad(set_to_none=True)
        x.grad = None
        with torch.autocast(x.device.type, dtype=dtype, enabled=dtype in (torch.float16, torch.bfloat16)):
            out = model(x, attn_mask=bias, **kwargs)
            loss = out.square().mean()
        loss.backward()
        self.assertTrue(out.isfinite().all())
        self.assertEqual(out[~kwargs["valid_mask"]].count_nonzero().item(), 0)
        grads = [x.grad] + [p.grad for p in model.parameters()]
        self.assertTrue(all(g is not None and g.isfinite().all() for g in grads))
        self.assertEqual(x.grad[~kwargs["valid_mask"]].count_nonzero().item(), 0)
        return out.detach(), [g.clone() for g in grads]

    def assert_same_result(self, actual, expected, *, rtol=2e-5, atol=3e-6):
        torch.testing.assert_close(actual[0], expected[0], rtol=rtol, atol=atol)
        for grad, reference in zip(actual[1], expected[1]):
            torch.testing.assert_close(grad, reference, rtol=rtol, atol=atol)

    def check_offsets(self, device, dtype):
        model_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        model, x, kwargs, base = self.make_case(device, model_dtype=model_dtype)
        for per_head in (False, True):
            with self.subTest(per_head=per_head):
                bias = base.expand(3, 2, 9, 9).clone() if per_head else base.clone()
                if per_head:
                    bias[0, 1, 5] = -torch.inf  # only this query/head is fully blocked
                reference = self.evaluate(model, x, kwargs, bias, dtype)
                # Quarter-integer biases remain exactly representable in FP32
                # after these offsets; differences must survive a half cast.
                for offset in (100000.0, -100000.0):
                    row_offsets = (torch.arange(9, device=device, dtype=model_dtype) + 1)[:, None] * offset
                    if per_head:
                        row_offsets = row_offsets + torch.arange(6, device=device).view(3, 2, 1, 1) * 4096
                    shifted = bias + row_offsets
                    if per_head:
                        shifted[0, :, 2, 8] = 1e9  # future
                        shifted[0, :, 5, 3] = 1e9  # previous episode
                        shifted[0, :, 8, 4] = 1e9  # outside the window
                        shifted[1, :, 6, 8] = 1e9  # padding
                    else:
                        shifted[2, 8] = 1e9
                        shifted[5, 3] = 1e9
                        shifted[8, 4] = 1e9
                    with self.subTest(offset=offset):
                        self.assert_same_result(self.evaluate(model, x, kwargs, shifted, dtype), reference)

                # These FP64 masks would overflow even an intermediate FP32
                # conversion. Compare constant finite entries with a bool mask.
                allowed = torch.isfinite(bias)
                reference = self.evaluate(model, x, kwargs, allowed, dtype)
                for offset in (1e300, -1e300):
                    huge = torch.full(bias.shape, offset, dtype=torch.float64, device=device)
                    huge.masked_fill_(~allowed, -torch.inf)
                    self.assert_same_result(self.evaluate(model, x, kwargs, huge, dtype), reference)

    def test_cpu_float32_and_float64_offsets(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self.check_offsets("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_float32_and_autocast_offsets(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                self.check_offsets("cuda", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_compile_fullgraph_float_bias_forward_and_backward(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                model, x, kwargs, base = self.make_case("cuda")
                bias = (base + 100000).expand(3, 2, 9, 9).clone()
                bias[0, 1, 5] = -torch.inf
                bias[0, :, 2, 8] = 1e9
                eager = self.evaluate(model, x, kwargs, bias, dtype)
                compiled = torch.compile(model, fullgraph=True)
                # A previous batch shape can make Dynamo symbolize B. Exercise
                # that path even when this test runs alone in a fresh process.
                warm_x = x[:2].detach().clone().requires_grad_()
                warm_kwargs = {name: value[:2] for name, value in kwargs.items()}
                self.evaluate(compiled, warm_x, warm_kwargs, None, dtype)
                actual = self.evaluate(compiled, x, kwargs, bias, dtype)
                half = dtype in (torch.float16, torch.bfloat16)
                self.assert_same_result(actual, eager, rtol=3e-2 if half else 5e-4,
                                        atol=3e-3 if half else 5e-5)


if __name__ == "__main__":
    unittest.main()
