"""Aligned causal cross-memory, ring lifetime and differentiable history."""

import copy
import unittest

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from rl_shared import EncoderDecoderBackbone, EncoderDecoderCache, EncoderDecoderConfig


def make_model(**changes):
    options = dict(hidden_dim=16, num_layers=2, decoder_layers=3, num_heads=2,
                   context_length=3, ffn_dim=32, alpha_init=0.7)
    options.update(changes)
    return EncoderDecoderBackbone(EncoderDecoderConfig(**options))


class EncoderDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(47)

    def test_parallel_step_wrap_reset_and_incremental_cross_projections(self):
        for encoder_layers, decoder_layers, window in ((1, 3, 1), (2, 3, 3), (3, 1, 4)):
            with self.subTest(depths=(encoder_layers, decoder_layers), window=window):
                model = make_model(num_layers=encoder_layers, decoder_layers=decoder_layers,
                                   context_length=window, final_norm=True)
                encoder, decoder = torch.randn(3, 29, 16), torch.randn(3, 29, 16)
                starts = torch.zeros(3, 29, dtype=torch.bool)
                starts[:, 0] = True
                starts[0, 7:9] = True
                starts[1, 13] = True
                starts[2, 19] = True
                reference = model(encoder, decoder, episode_starts=starts)
                seen = [[] for _ in model.decoder]
                handles = [block.cross_attention.kv.register_forward_pre_hook(
                    lambda module, args, bucket=bucket: bucket.append(tuple(args[0].shape)))
                    for block, bucket in zip(model.decoder, seen)]
                cache = model.create_cache(3)
                self.assertIsInstance(cache, EncoderDecoderCache)
                pointers = [cache.keys.data_ptr(), cache.values.data_ptr()]
                outputs = [[], []]
                positions = torch.zeros(3, dtype=torch.long)
                for t in range(29):
                    result = model.step(encoder[:, t:t + 1], decoder[:, t:t + 1], cache,
                                        episode_starts=starts[:, t])
                    for collected, item in zip(outputs, result):
                        collected.append(item)
                    positions.masked_fill_(starts[:, t], 0).add_(1)
                    torch.testing.assert_close(cache.next_positions, positions)
                    torch.testing.assert_close(cache.lengths, positions.clamp(max=window))
                    self.assertEqual([cache.keys.data_ptr(), cache.values.data_ptr()], pointers)
                for actual, expected in zip(outputs, reference):
                    torch.testing.assert_close(torch.cat(actual, 1), expected, rtol=2e-5, atol=3e-6)
                for handle in handles:
                    handle.remove()
                self.assertEqual(seen, [[(3, 1, 16)] * 29 for _ in model.decoder])
                self.assertEqual(cache.keys.shape[0], encoder_layers + 2 * decoder_layers)
                self.assertGreater(cache.nbytes, 2 * cache.keys.numel() * cache.keys.element_size())

    def test_future_episode_environment_and_padding_isolation(self):
        model = make_model()
        encoder, decoder = torch.randn(3, 18, 16), torch.randn(3, 18, 16)
        starts = torch.zeros(3, 18, dtype=torch.bool)
        starts[:, 8] = True
        original = model(encoder, decoder, episode_starts=starts)
        for source in (0, 1):
            changed = [encoder.clone(), decoder.clone()]
            changed[source][:, 13:] *= 50
            result = model(*changed, episode_starts=starts)
            for left, right in zip(result, original):
                torch.testing.assert_close(left[:, :13], right[:, :13])
            changed[source][:, :8] *= 50
            result = model(*changed, episode_starts=starts)
            for left, right in zip(result, original):
                torch.testing.assert_close(left[:, 8:13], right[:, 8:13])
            changed[source][0] *= 20
            result = model(*changed, episode_starts=starts)
            for left, right in zip(result, original):
                torch.testing.assert_close(left[1:, 8:13], right[1:, 8:13])
        valid = torch.arange(18)[None] < torch.tensor([18, 7, 0])[:, None]
        inputs = [x.clone().masked_fill(~valid[..., None], float("nan")).requires_grad_()
                  for x in (encoder, decoder)]
        outputs = model(*inputs, valid_mask=valid)
        sum(x.square().sum() for x in outputs).backward()
        for output in outputs:
            self.assertTrue(output.isfinite().all())
            self.assertEqual(output[~valid].count_nonzero().item(), 0)
        for x in inputs:
            self.assertTrue(x.grad.isfinite().all())
            self.assertEqual(x.grad[~valid].count_nonzero().item(), 0)

    def test_local_cross_window_includes_current_and_excludes_distance_w(self):
        model = make_model().double()
        block = model.decoder[0]
        memory = torch.randn(1, 7, 16, dtype=torch.float64, requires_grad=True)
        queries = torch.randn_like(memory)
        positions = torch.arange(7)[None]
        cos, sin = model.encoder.rope(positions)
        distance = positions[0, :, None] - positions[0, None, :]
        mask = ((distance >= 0) & (distance < 3))[None, None]
        q = block.cross_attention.project_query(block.cross_norm(queries), cos, sin)
        k, v = block.cross_attention.project_memory(block.memory_norm(memory), cos, sin)
        out = block.cross_attention.attend(q, k, v, mask, False)
        grad, = torch.autograd.grad(out[:, 4].square().sum(), memory)
        self.assertEqual(grad[:, :2].count_nonzero().item(), 0)  # distance >= W
        self.assertEqual(grad[:, 5:].count_nonzero().item(), 0)  # future
        self.assertTrue((grad[:, 2:5].norm(dim=-1) > 0).all())  # includes H_t

    def test_full_history_gradients_and_receptive_field_boundaries(self):
        model = make_model().double()
        encoder = torch.randn(1, 17, 16, dtype=torch.float64, requires_grad=True)
        decoder = torch.randn_like(encoder, requires_grad=True)
        history, policy = model(encoder, decoder)
        encoder_grad, decoder_grad = torch.autograd.grad(policy[:, -1].square().sum(), (encoder, decoder),
                                                        retain_graph=True)
        # E=2, D=3, W=3 -> total reach 10 previous tokens, critic reach 4.
        self.assertEqual(encoder_grad[:, :6].count_nonzero().item(), 0)
        self.assertGreater(encoder_grad[:, 6].norm().item(), 1e-12)
        self.assertEqual(decoder_grad[:, :10].count_nonzero().item(), 0)
        self.assertGreater(decoder_grad[:, 10].norm().item(), 1e-12)
        encoder_grad, = torch.autograd.grad(history[:, -1].square().sum(), encoder)
        self.assertEqual(encoder_grad[:, :12].count_nonzero().item(), 0)
        self.assertGreater(encoder_grad[:, 12].norm().item(), 1e-12)
        default = make_model(alpha_init=1e-4)
        x = torch.randn(1, 4, 16, requires_grad=True)
        default(x, torch.randn_like(x))[1][:, -1].square().sum().backward()
        self.assertGreater(x.grad[:, :-1].norm().item(), 0)

    def test_step_current_gradients_survive_mutation_without_cross_time_graph(self):
        model = make_model(context_length=2)
        cache = model.create_cache(2)
        old = torch.randn(2, 1, 16, requires_grad=True)
        model.step(old, torch.randn_like(old), cache, enable_grad=True)
        encoder = torch.randn_like(old, requires_grad=True)
        decoder = torch.randn_like(old, requires_grad=True)
        with torch.no_grad():
            history, policy = model.step(encoder, decoder, cache, enable_grad=True)
        for _ in range(5):
            model.step(torch.randn_like(old), torch.randn_like(old), cache)
        cache.reset(torch.tensor([True, False]))
        model.step(torch.randn_like(old), torch.randn_like(old), cache)
        cache.reset()
        (history.square().sum() + policy.square().sum()).backward()
        self.assertIsNone(old.grad)
        self.assertGreater(encoder.grad.norm().item(), 0)
        self.assertGreater(decoder.grad.norm().item(), 0)
        self.assertFalse(cache.keys.requires_grad)
        self.assertIsNone(cache.values.grad_fn)
        self.assertFalse(model.step(encoder, decoder, cache)[1].requires_grad)
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, "leaving torch.inference_mode"):
                model.step(encoder, decoder, cache, enable_grad=True)

    def test_reset_invalidates_all_three_caches_and_preserves_other_environments(self):
        model = make_model()
        cache = model.create_cache(3)
        for _ in range(8):
            model.step(torch.randn(3, 1, 16), torch.randn(3, 1, 16), cache)
        old = {name: tensor.clone() for name, tensor in cache.tensors().items()}
        cache.reset(torch.tensor([False, True, False]))
        torch.testing.assert_close(cache.keys[:, [0, 2]], old["keys"][:, [0, 2]])
        cache.keys[:, 1].fill_(1e4)
        cache.values[:, 1].fill_(-1e4)
        x, y = torch.randn(3, 1, 16), torch.randn(3, 1, 16)
        for result, fresh in zip(model.step(x, y, cache), model.step(x[1:2], y[1:2], model.create_cache(1))):
            torch.testing.assert_close(result[1:2], fresh, rtol=2e-5, atol=3e-6)

    def test_validation_defaults_and_serialization(self):
        config = EncoderDecoderConfig(hidden_dim=16)
        self.assertEqual((config.num_layers, config.decoder_layers), (4, 4))
        self.assertEqual(EncoderDecoderConfig(hidden_dim=256, context_length=32).warmup_length, 249)
        self.assertEqual(EncoderDecoderConfig(hidden_dim=256, context_length=32).value_warmup_length, 125)
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                EncoderDecoderConfig(hidden_dim=16, decoder_layers=invalid)
        model = make_model()
        cache = model.create_cache(2)
        x = torch.randn(2, 1, 16)
        with self.assertRaises(ValueError):
            model.step(x, x[:, :, :8], cache)
        with self.assertRaises(ValueError):
            model.step(x, x, model.create_cache(1))
        with self.assertRaises(TypeError):
            model.step(x, x, model.encoder.create_cache(2))
        with self.assertRaises(ValueError):
            model.step(x, x, cache, episode_starts=torch.zeros(1, dtype=torch.bool))
        self.assertFalse(any("cache" in key for key in model.state_dict()))
        other = copy.deepcopy(model)
        for actual, expected in zip(other(x, x), model(x, x)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class EncoderDecoderCUDATests(unittest.TestCase):
    def test_fused_forward_backward_and_streaming_match_math(self):
        torch.manual_seed(47)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = make_model(hidden_dim=32).cuda()
        encoder, decoder = torch.randn(2, 17, 32, device="cuda"), torch.randn(2, 17, 32, device="cuda")
        starts = torch.zeros(2, 17, dtype=torch.bool, device="cuda")
        starts[0, 7] = True
        valid = torch.ones_like(starts)
        with sdpa_kernel(SDPBackend.MATH):
            expected = model(encoder, decoder, episode_starts=starts, valid_mask=valid)
            sum(x.square().mean() for x in expected).backward()
            gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        # Force a fused kernel so this check cannot silently exercise math again.
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            actual = model(encoder, decoder, episode_starts=starts, valid_mask=valid)
            sum(x.square().mean() for x in actual).backward()
            cache = model.create_cache(2)
            streamed = [[], []]
            for t in range(17):
                for output, token in zip(streamed, model.step(encoder[:, t:t + 1], decoder[:, t:t + 1], cache,
                                                              episode_starts=starts[:, t])):
                    output.append(token)
        for left, right, serial in zip(actual, expected, streamed):
            torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(torch.cat(serial, 1), right, rtol=1e-4, atol=1e-5)
        for parameter, reference in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, reference, rtol=2e-4, atol=2e-5)
        for dtype in (torch.bfloat16, torch.float16):
            cache = model.create_cache(2, dtype=dtype)
            with torch.autocast("cuda", dtype=dtype):
                self.assertTrue(model.step(encoder[:, :1], decoder[:, :1], cache)[1].isfinite().all())


if __name__ == "__main__":
    unittest.main()
