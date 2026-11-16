"""Component equations, independence, masks and streaming for configurable models."""

from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

from rl_shared import EncoderDecoderBackbone, EncoderDecoderConfig, TransformerBackbone, TransformerConfig
from rl_shared.layers import DynamicTanh, RMSNorm, Snake
from rl_shared.models.transformer.feedforward import SwiSnakeGLU


class VariantTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_component_equations_channel_broadcast_and_gradients(self):
        x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        dyt = DynamicTanh(8).double()
        snake = Snake(8).double()
        with torch.no_grad():
            snake.log_alpha.copy_(torch.linspace(-2, 2, 8))
            dyt.weight.copy_(torch.linspace(.5, 1.5, 8))
            dyt.bias.copy_(torch.linspace(-.2, .2, 8))
        torch.testing.assert_close(dyt(x), torch.tanh(.5 * x) * dyt.weight + dyt.bias)
        a = snake.log_alpha.exp()
        torch.testing.assert_close(snake(x), x + torch.sin(a * x).square() / (a + 1e-6))
        (dyt(x).square().sum() + snake(x).square().sum()).backward()
        for parameter in (*dyt.parameters(), *snake.parameters()):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0)
        tiny = Snake(8, alpha_init=1e-30)
        y = tiny(x.detach().float())
        y.sum().backward()
        torch.testing.assert_close(y, x.detach().float())
        self.assertTrue(torch.isfinite(tiny.log_alpha.grad).all())

    def test_snake_only_on_value_branch(self):
        config = TransformerConfig(hidden_dim=16, ffn_dim=32, ffn_type="swi_snake_glu")
        ffn = SwiSnakeGLU(config)
        x = torch.randn(2, 7, 16)
        gate, value = F.linear(x, ffn.gate_up.weight).chunk(2, -1)
        expected = F.linear(F.silu(gate) * (value + value.sin().square() / (1 + config.snake_eps)), ffn.down.weight)
        torch.testing.assert_close(ffn(x), expected)
        self.assertEqual(ffn.value_activation.log_alpha.shape, (32,))

    def test_variants_stream_masks_resets_padding_and_gradients(self):
        for norm in ("rmsnorm", "dyt"):
            for ffn in ("swiglu", "swi_snake_glu"):
                for ed in (False, True):
                    with self.subTest(norm=norm, ffn=ffn, ed=ed):
                        kind = EncoderDecoderConfig if ed else TransformerConfig
                        config = kind(hidden_dim=16, num_layers=2, num_heads=2, context_length=3,
                                      ffn_dim=32, alpha_init=.7, norm_type=norm, ffn_type=ffn, final_norm=True,
                                      **({"decoder_layers": 2} if ed else {}))
                        model = (EncoderDecoderBackbone if ed else TransformerBackbone)(config)
                        x = torch.randn(2, 15, 16)
                        y = torch.randn_like(x)
                        starts = torch.zeros(2, 15, dtype=torch.bool)
                        starts[:, 0] = True
                        starts[0, 5] = True
                        starts[1, 9] = True
                        inputs = (x, y) if ed else (x,)
                        outputs = model(*inputs, episode_starts=starts)
                        expected = outputs if ed else (outputs,)
                        cache = model.create_cache(2)
                        actual = [[] for _ in expected]
                        for t in range(15):
                            result = model.step(*(v[:, t:t+1] for v in inputs), cache, episode_starts=starts[:, t])
                            for tokens, token in zip(actual, result if ed else (result,)):
                                tokens.append(token)
                        for reference, tokens in zip(expected, actual):
                            torch.testing.assert_close(torch.cat(tokens, 1), reference, rtol=3e-5, atol=4e-6)
                        valid = torch.ones_like(starts)
                        valid[0, -2:] = False
                        padded = [v.masked_fill(~valid[..., None], float("nan")).requires_grad_() for v in inputs]
                        result = model(*padded, episode_starts=starts, valid_mask=valid)
                        result = result if ed else (result,)
                        self.assertTrue(all(v[~valid].count_nonzero() == 0 for v in result))
                        sum(v.square().sum() for v in result).backward()
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                        if norm == "dyt":
                            self.assertFalse(any(isinstance(m, RMSNorm) for m in model.modules()))
                            attention = model.encoder.blocks[0].attention if ed else model.blocks[0].attention
                            self.assertIsInstance(attention.q_norm, DynamicTanh)
                            self.assertEqual(attention.q_norm.weight.shape, (config.head_dim,))

    def test_causality_episode_isolation_and_disabled_qk(self):
        model = EncoderDecoderBackbone(EncoderDecoderConfig(hidden_dim=16, num_heads=2, num_layers=2,
            decoder_layers=2, context_length=3, norm_type="dyt", ffn_type="swi_snake_glu",
            qk_norm=False, alpha_init=.7))
        x, y = torch.randn(1, 10, 16), torch.randn(1, 10, 16)
        starts = torch.zeros(1, 10, dtype=torch.bool)
        starts[:, 5] = True
        reference = model(x, y, episode_starts=starts)
        xx, yy = x.clone(), y.clone()
        xx[:, 7:] += 100
        yy[:, 7:] -= 100
        for left, right in zip(reference, model(xx, yy, episode_starts=starts)):
            torch.testing.assert_close(left[:, :7], right[:, :7], rtol=0, atol=0)
        for left, right in zip(reference, model(x[:, 5:], y[:, 5:])):
            torch.testing.assert_close(left[:, 5:], right, rtol=3e-5, atol=4e-6)

    def test_config_validation_and_compatibility_imports(self):
        from rl_shared.config import TransformerConfig as old
        from rl_shared.models.transformer import TransformerConfig as new
        self.assertIs(old, new)
        for options in ({"norm_type": "unknown"}, {"ffn_type": "unknown"},
                        {"snake_eps": 0}, {"snake_alpha_init": float("inf")},
                        {"dyt_alpha_init": True}, {"snake_alpha_init": -1}):
            with self.assertRaises(ValueError):
                TransformerConfig(hidden_dim=16, **options)
        original = TransformerBackbone(TransformerConfig(hidden_dim=16))
        self.assertFalse(any("log_alpha" in name or ".bias" in name for name in original.state_dict()))
        config = EncoderDecoderConfig(hidden_dim=16, norm_type="dyt", ffn_type="swi_snake_glu")
        self.assertEqual(config.encoder_config().norm_type, "dyt")
        self.assertEqual(config.cache_config().ffn_type, "swi_snake_glu")
        self.assertNotEqual(config, replace(config, norm_type="rmsnorm"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class VariantCUDATests(unittest.TestCase):
    def test_dyt_snake_inductor_outputs_and_gradients(self):
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        config = EncoderDecoderConfig(hidden_dim=16, num_heads=2, num_layers=1, decoder_layers=1,
            context_length=3, ffn_dim=32, norm_type="dyt", ffn_type="swi_snake_glu", alpha_init=.7)
        model = EncoderDecoderBackbone(config).cuda()
        x, y = torch.randn(2, 9, 16, device="cuda"), torch.randn(2, 9, 16, device="cuda")
        starts = torch.zeros(2, 9, dtype=torch.bool, device="cuda")
        starts[0, 4] = True
        expected = model(x, y, episode_starts=starts)
        sum(v.square().mean() for v in expected).backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        compiled = torch.compile(model, fullgraph=True)
        actual = compiled(x, y, episode_starts=starts)
        sum(v.square().mean() for v in actual).backward()
        for left, right in zip((*actual, *(p.grad for p in model.parameters())), (*expected, *gradients)):
            torch.testing.assert_close(left, right, rtol=5e-4, atol=5e-5)


if __name__ == "__main__":
    unittest.main()
