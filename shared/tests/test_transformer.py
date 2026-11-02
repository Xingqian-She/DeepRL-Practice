"""Behavioral tests; no environment training or third-party test runner required."""

import io
import unittest

import torch
from torch import nn

from rl_shared import KVCache, TransformerBackbone, TransformerConfig, init_head


def make_model(**kwargs):
    options = dict(hidden_dim=16, num_heads=2, num_layers=2, context_length=4,
                   ffn_dim=32, alpha_init=0.7)
    options.update(kwargs)
    return TransformerBackbone(TransformerConfig(**options))


class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)

    def test_stream_matches_parallel_across_wraps_and_episode_resets(self):
        for qk_norm in (True, False):
            for final_norm in (True, False):
                with self.subTest(qk_norm=qk_norm, final_norm=final_norm):
                    model = make_model(qk_norm=qk_norm, final_norm=final_norm)
                    x = torch.randn(3, 23, 16)
                    starts = torch.zeros(3, 23, dtype=torch.bool)
                    starts[:, 0] = True
                    starts[0, 5] = True
                    starts[2, 7:9] = True
                    starts[:, 10] = True
                    starts[1, 17] = True
                    reference = model(x, episode_starts=starts)
                    cache = model.create_cache(3)
                    outputs = []
                    expected_positions = torch.zeros(3, dtype=torch.long)
                    key_storage = cache.keys.data_ptr()
                    for t in range(x.shape[1]):
                        outputs.append(model.step(x[:, t:t + 1], cache, episode_starts=starts[:, t]))
                        expected_positions.masked_fill_(starts[:, t], 0).add_(1)
                        torch.testing.assert_close(cache.next_positions, expected_positions)
                        torch.testing.assert_close(cache.lengths, expected_positions.clamp(max=4))
                        self.assertEqual(cache.keys.data_ptr(), key_storage)
                    torch.testing.assert_close(torch.cat(outputs, dim=1), reference, rtol=2e-5, atol=3e-6)

    def test_partial_reset_matches_fresh_environment_and_preserves_other_slots(self):
        model = make_model()
        cache = model.create_cache(3)
        for _ in range(7):
            model.step(torch.randn(3, 1, 16), cache)
        old_keys, old_values = cache.keys.clone(), cache.values.clone()
        cache.reset(torch.tensor([False, True, False]))
        torch.testing.assert_close(cache.next_positions, torch.tensor([7, 0, 7]))
        torch.testing.assert_close(cache.keys[:, [0, 2]], old_keys[:, [0, 2]])
        torch.testing.assert_close(cache.values[:, [0, 2]], old_values[:, [0, 2]])
        # Large stale but finite values must be completely invisible after reset.
        cache.keys[:, 1].fill_(1e4)
        cache.values[:, 1].fill_(-1e4)
        token = torch.randn(3, 1, 16)
        result = model.step(token, cache)
        fresh = model.step(token[1:2], model.create_cache(1))
        torch.testing.assert_close(result[1:2], fresh, rtol=2e-5, atol=3e-6)
        cache.reset(torch.zeros(3, dtype=torch.bool))
        torch.testing.assert_close(cache.next_positions, torch.tensor([8, 1, 8]))
        cache.reset()
        self.assertEqual(cache.next_positions.count_nonzero().item(), 0)
        self.assertEqual(cache.lengths.count_nonzero().item(), 0)

    def test_future_and_other_episodes_cannot_leak(self):
        model = make_model()
        x = torch.randn(2, 12, 16)
        starts = torch.zeros(2, 12, dtype=torch.bool)
        starts[:, 6] = True
        original = model(x, episode_starts=starts)
        changed = x.clone()
        changed[:, 9:] = torch.randn_like(changed[:, 9:]) * 20
        torch.testing.assert_close(model(changed, episode_starts=starts)[:, :9], original[:, :9])
        changed = x.clone()
        changed[:, :6] = torch.randn_like(changed[:, :6]) * 20
        torch.testing.assert_close(model(changed, episode_starts=starts)[:, 6:], original[:, 6:])
        # Removing episode separation must actually change the boundary token.
        self.assertGreater((model(x)[:, 6] - original[:, 6]).abs().max().item(), 1e-3)

    def test_padding_is_zero_and_cannot_contaminate_valid_outputs_or_gradients(self):
        model = make_model(final_norm=True)
        x = torch.randn(3, 7, 16)
        valid = torch.arange(7)[None, :] < torch.tensor([7, 3, 0])[:, None]
        x[~valid] = float("nan")
        x.requires_grad_()
        out = model(x, valid_mask=valid)
        self.assertTrue(out.isfinite().all())
        self.assertEqual(out[~valid].count_nonzero().item(), 0)
        reference = model(x[1:2, :3].detach())
        torch.testing.assert_close(out[1:2, :3], reference)
        out.square().sum().backward()
        self.assertTrue(x.grad.isfinite().all())
        self.assertEqual(x.grad[~valid].count_nonzero().item(), 0)

    def test_custom_boolean_float_per_head_and_empty_masks(self):
        model = make_model()
        x = torch.randn(2, 7, 16)
        mask = torch.ones(2, 2, 7, 7, dtype=torch.bool)
        mask[:, :, :, 1] = False
        mask[0, 1, 4] = False  # one fully blocked query/head
        float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf)
        out = model(x, attn_mask=mask)
        torch.testing.assert_close(out, model(x, attn_mask=float_mask))
        # A blocked token cannot influence other tokens, at any layer.
        altered = x.clone()
        altered[:, 1] *= 30
        indices = torch.tensor([0, 2, 3, 4, 5, 6])
        torch.testing.assert_close(out[:, indices], model(altered, attn_mask=mask)[:, indices])
        for mask2 in (torch.zeros(7, 7, dtype=torch.bool), torch.full((7, 7), -torch.inf)):
            empty = model(x, attn_mask=mask2)
            self.assertTrue(empty.isfinite().all())
            # Attention vanishes, leaving token-wise residual + FFN behavior.
            torch.testing.assert_close(empty, torch.cat([model(x[:, i:i + 1], attn_mask=mask2[:1, :1])
                                                        for i in range(7)], dim=1))
        bias = torch.linspace(-2, 2, 7).expand(7, 7)
        self.assertGreater((model(x, attn_mask=bias) - model(x)).abs().max().item(), 1e-3)

    def test_noncausal_removes_window_but_retains_episode_masks(self):
        model = make_model(context_length=1)
        x = torch.randn(1, 8, 16)
        starts = torch.zeros(1, 8, dtype=torch.bool)
        starts[:, 4] = True
        original = model(x, episode_starts=starts, is_causal=False)
        changed = x.clone()
        changed[:, 3] *= 10
        result = model(changed, episode_starts=starts, is_causal=False)
        self.assertGreater((result[:, 0] - original[:, 0]).abs().max().item(), 1e-3)
        torch.testing.assert_close(result[:, 4:], original[:, 4:])

    def test_explicit_positions_rope_relative_shift_and_dtype_conversion(self):
        model = make_model()
        x = torch.randn(2, 7, 16)
        positions = torch.arange(7).expand(2, -1)
        torch.testing.assert_close(model(x), model(x, position_ids=positions))
        torch.testing.assert_close(model(x), model(x, position_ids=positions + 123), rtol=3e-5, atol=4e-6)
        self.assertGreater((model(x, position_ids=positions * 3) - model(x)).abs().max().item(), 1e-3)
        phase_before = model.rope(positions)[0]
        model.bfloat16()
        torch.testing.assert_close(model.rope(positions)[0], phase_before, rtol=0, atol=0)

    def test_zero_gate_identity_and_expected_gradient_startup(self):
        model = make_model(alpha_init=0.0)
        x = torch.randn(2, 6, 16, requires_grad=True)
        torch.testing.assert_close(model(x), x, rtol=0, atol=0)
        model(x).square().mean().backward()
        for block in model.blocks:
            self.assertGreater(block.alpha_attn.grad.abs().item(), 0)
            self.assertEqual(block.attention.qkv.weight.grad.count_nonzero().item(), 0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        self.assertGreater(model.blocks[0].attention.qkv.weight.grad.norm().item(), 0)
        small = make_model(alpha_init=1e-4)
        small(x).square().mean().backward()
        self.assertGreater(small.blocks[0].attention.qkv.weight.grad.norm().item(), 0)

    def test_short_parameter_updates_are_finite_and_change_features(self):
        model = make_model(alpha_init=1e-4)
        x, target = torch.randn(3, 7, 16), torch.randn(3, 7, 16)
        before = model(x).detach()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - target).square().mean()
            loss.backward()
            self.assertTrue(loss.isfinite())
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        self.assertGreater((model(x).detach() - before).abs().max().item(), 1e-4)

    def test_step_default_detaches_and_train_eval_does_not_select_path(self):
        model = make_model()
        x = torch.randn(2, 1, 16, requires_grad=True)
        self.assertFalse(model.step(x, model.create_cache(2)).requires_grad)
        model.eval()
        self.assertTrue(model(x).requires_grad)
        self.assertTrue(model.step(x, model.create_cache(2), enable_grad=True).requires_grad)

    def test_step_current_gradient_matches_full_input_gradient_without_history_grad(self):
        model = make_model()
        history = torch.randn(2, 3, 16, requires_grad=True)
        current = torch.randn(2, 1, 16, requires_grad=True)
        cache = model.create_cache(2)
        for t in range(3):
            model.step(history[:, t:t + 1], cache, enable_grad=True)
        with torch.no_grad():
            out = model.step(current, cache, enable_grad=True)
        out.square().sum().backward()
        current_grad = current.grad.clone()
        self.assertIsNone(history.grad)
        for part in model.blocks[0].attention.qkv.weight.grad.chunk(3):
            self.assertGreater(part.norm().item(), 0)
        self.assertFalse(cache.keys.requires_grad)
        self.assertIsNone(cache.keys.grad_fn)
        self.assertIsNone(cache.values.grad_fn)
        full = torch.cat((history.detach(), current.detach()), dim=1).requires_grad_()
        full_out = model(full)[:, -1:]
        torch.testing.assert_close(out, full_out, rtol=2e-5, atol=3e-6)
        full_out.square().sum().backward()
        torch.testing.assert_close(current_grad, full.grad[:, -1:], rtol=3e-5, atol=4e-6)

    def test_delayed_backward_survives_later_grad_no_grad_wrap_and_reset(self):
        model = make_model(context_length=2)
        cache = model.create_cache(2)
        first = torch.randn(2, 1, 16, requires_grad=True)
        second = torch.randn(2, 1, 16, requires_grad=True)
        a = model.step(first, cache, enable_grad=True)
        b = model.step(second, cache, enable_grad=True)
        for _ in range(5):
            model.step(torch.randn(2, 1, 16), cache)
        cache.reset(torch.tensor([True, False]))
        model.step(torch.randn(2, 1, 16), cache)
        cache.reset()
        # b must not retain a cross-time edge to first.
        b.square().sum().backward()
        self.assertIsNone(first.grad)
        self.assertTrue(second.grad.isfinite().all())
        a.square().sum().backward()
        self.assertTrue(first.grad.isfinite().all())

    def test_inference_mode_rejection_is_nonmutating_and_cache_can_be_reused(self):
        model = make_model()
        x = torch.randn(1, 1, 16, requires_grad=True)
        with torch.inference_mode():
            cache = model.create_cache(1)
            with self.assertRaisesRegex(RuntimeError, "leaving torch.inference_mode"):
                model.step(x, cache, enable_grad=True)
            self.assertEqual(cache.next_positions.item(), 0)
            model.step(x, cache)
        model.step(x, cache, enable_grad=True).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_cache_is_independent_and_excluded_from_serialization(self):
        model = make_model()
        a, b = model.create_cache(1), model.create_cache(1)
        x = torch.randn(1, 1, 16)
        model.step(x, a)
        self.assertEqual(b.next_positions.item(), 0)
        self.assertFalse(any("cache" in key for key in model.state_dict()))
        serialized = io.BytesIO()
        torch.save(model.state_dict(), serialized)
        serialized.seek(0)
        other = TransformerBackbone(model.config)
        other.load_state_dict(torch.load(serialized, weights_only=True))
        torch.testing.assert_close(other(x), model(x))
        torch.testing.assert_close(model.step(x, b), other.step(x, other.create_cache(1)))

    def test_config_input_and_cache_validation(self):
        for options in ({"hidden_dim": 0}, {"hidden_dim": 18}, {"num_layers": 0},
                        {"context_length": -1}, {"ffn_dim": 0}, {"alpha_init": -1},
                        {"rms_eps": 0}, {"rope_theta": float("nan")}, {"num_heads": True}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                TransformerConfig(**(dict(hidden_dim=16) | options))
        self.assertEqual(TransformerConfig(hidden_dim=128).ffn_dim, 384)
        model = make_model()
        x = torch.randn(2, 1, 16)
        for bad in (torch.randn(2, 16), torch.randn(2, 1, 15), torch.empty(0, 1, 16), torch.ones(2, 1, 16, dtype=torch.long)):
            with self.subTest(shape=bad.shape), self.assertRaises((ValueError, TypeError)):
                model(bad)
        for bad_cache in (model.create_cache(1), model.create_cache(2, dtype=torch.float64),
                          make_model(context_length=8).create_cache(2)):
            with self.assertRaises(ValueError):
                model.step(x, bad_cache)
        for kw in ({"episode_starts": torch.zeros(2, 1)}, {"valid_mask": torch.ones(2, 2, dtype=torch.bool)},
                   {"position_ids": torch.zeros(2, 1)}, {"attn_mask": torch.ones(2, 1, 1)},
                   {"attn_mask": torch.ones(3, 3, dtype=torch.bool)}, {"is_causal": 1}):
            with self.subTest(kw=kw.keys()), self.assertRaises((ValueError, TypeError)):
                model(x, **kw)
        cache = model.create_cache(2)
        with self.assertRaises(TypeError):
            cache.reset(torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            cache.reset(torch.tensor([True]))
        with self.assertRaises(ValueError):
            model.step(x.expand(2, 3, 16), cache)
        with self.assertRaises(TypeError):
            model.step(x, cache, enable_grad=1)
        with self.assertRaises(ValueError):
            model.create_cache(0)
        with self.assertRaises(TypeError):
            model.create_cache(2, dtype=torch.long)
        self.assertIsInstance(cache, KVCache)
        self.assertGreater(cache.nbytes, 0)

    def test_head_initialization_and_bias_free_backbone(self):
        for shape in ((3, 16), (20, 8)):
            for gain in (0.01, 1.0):
                head = init_head(nn.Linear(shape[1], shape[0]), gain=gain, bias=0.25)
                weight = head.weight.detach()
                gram = weight @ weight.T if shape[0] <= shape[1] else weight.T @ weight
                torch.testing.assert_close(gram, torch.eye(min(shape)) * gain ** 2, rtol=2e-5, atol=1e-7)
                torch.testing.assert_close(head.bias, torch.full_like(head.bias, 0.25))
        init_head(nn.Linear(4, 2, bias=False))
        with self.assertRaises(TypeError):
            init_head(nn.Sequential(nn.Linear(4, 2)))
        with self.assertRaises(ValueError):
            init_head(nn.Linear(4, 2), gain=float("nan"))
        self.assertTrue(all(m.bias is None for m in make_model().modules() if isinstance(m, nn.Linear)))
        self.assertFalse(any(isinstance(m, nn.Dropout) for m in make_model().modules()))


if __name__ == "__main__":
    unittest.main()
