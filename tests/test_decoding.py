"""Unit tests for the decoding strategies (stdlib unittest, no pytest)."""

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.backend import NGramLM
from lab.decoding import (DecodingConfig, STRATEGIES, apply_repetition_penalty,
                          apply_temperature, filter_min_p, filter_top_k,
                          filter_top_p, generate, select_next_token)

LOGITS = [1.0, 5.0, 3.0, 4.0, 0.5, 2.0]
INF = math.inf


class TestFilters(unittest.TestCase):
    def test_top_k_keeps_only_k(self):
        out = filter_top_k(LOGITS, 3)
        kept = [i for i, x in enumerate(out) if x != -INF]
        self.assertEqual(sorted(kept), [1, 2, 3])
        self.assertEqual(out[1], 5.0)

    def test_top_k_zero_or_full_is_noop(self):
        self.assertEqual(filter_top_k(LOGITS, 0), LOGITS)
        self.assertEqual(filter_top_k(LOGITS, 100), LOGITS)

    def test_top_p_keeps_minimal_set(self):
        out = filter_top_p(LOGITS, 0.6)
        kept = [i for i, x in enumerate(out) if x != -INF]
        # probs: 5 -> .5, 4 -> .25, 3 -> .125 ... need >= .6 => top-2
        self.assertEqual(sorted(kept), [1, 3])

    def test_top_p_always_keeps_argmax(self):
        out = filter_top_p(LOGITS, 0.001)
        self.assertEqual(out[1], 5.0)

    def test_top_p_one_is_noop(self):
        self.assertEqual(filter_top_p(LOGITS, 1.0), LOGITS)

    def test_min_p_threshold(self):
        # max prob over softmax ~ 5.0 -> keep tokens with logit >= 5 + log(0.5)
        out = filter_min_p(LOGITS, 0.5)
        kept = [i for i, x in enumerate(out) if x != -INF]
        self.assertIn(1, kept)
        self.assertNotIn(0, kept)
        self.assertNotIn(4, kept)

    def test_temperature_zero_is_greedy(self):
        out = apply_temperature(LOGITS, 0.0)
        self.assertEqual(max(range(6), key=lambda i: out[i]), 1)
        self.assertEqual([x for x in out if x != -INF], [0.0])

    def test_temperature_sharpens(self):
        sharp = filter_top_p(apply_temperature(LOGITS, 0.5), 1.0)
        flat = filter_top_p(apply_temperature(LOGITS, 2.0), 1.0)
        self.assertGreater(sharp[1], flat[1])


class TestRepetitionPenalty(unittest.TestCase):
    def test_penalizes_seen_tokens(self):
        out = apply_repetition_penalty([2.0, -2.0, 1.0], [0, 1], 2.0)
        self.assertAlmostEqual(out[0], 1.0)   # positive logit divided
        self.assertAlmostEqual(out[1], -4.0)  # negative logit multiplied
        self.assertAlmostEqual(out[2], 1.0)   # unseen untouched

    def test_penalty_one_is_noop(self):
        self.assertEqual(apply_repetition_penalty(LOGITS, [0], 1.0), LOGITS)


class TestSelectNextToken(unittest.TestCase):
    def test_greedy_is_argmax_and_deterministic(self):
        cfg = DecodingConfig(strategy="greedy")
        a = select_next_token(LOGITS, cfg, rng=random.Random(1))
        b = select_next_token(LOGITS, cfg, rng=random.Random(2))
        self.assertEqual(a.token_id, 1)
        self.assertEqual(a.token_id, b.token_id)
        self.assertEqual(a.entropy, 0.0)

    def test_top_k_limits_candidates(self):
        cfg = DecodingConfig(strategy="topk", top_k=2)
        info = select_next_token(LOGITS, cfg, rng=random.Random(0))
        self.assertEqual(info.kept, 2)
        self.assertIn(info.token_id, (1, 3))

    def test_top_p_limits_candidates(self):
        cfg = DecodingConfig(strategy="topp", top_p=0.6)
        info = select_next_token(LOGITS, cfg, rng=random.Random(0))
        self.assertEqual(info.kept, 2)
        self.assertIn(info.token_id, (1, 3))

    def test_random_uniform_hits_every_token(self):
        cfg = DecodingConfig(strategy="random")
        seen = {select_next_token(LOGITS, cfg, rng=random.Random(s)).token_id
                for s in range(200)}
        self.assertEqual(seen, set(range(6)))

    def test_seed_reproducibility(self):
        cfg = DecodingConfig(strategy="temperature", temperature=1.0)
        a = select_next_token(LOGITS, cfg, rng=random.Random(7))
        b = select_next_token(LOGITS, cfg, rng=random.Random(7))
        self.assertEqual(a.token_id, b.token_id)

    def test_entropy_zero_for_greedy_high_for_uniform(self):
        greedy = select_next_token(LOGITS, DecodingConfig(strategy="greedy"))
        rnd = select_next_token(LOGITS, DecodingConfig(strategy="random"))
        self.assertLess(greedy.entropy, 1e-9)
        self.assertAlmostEqual(rnd.entropy, math.log(6), places=6)


class TestConfigValidation(unittest.TestCase):
    def test_bad_strategy_raises(self):
        with self.assertRaises(ValueError):
            DecodingConfig(strategy="nope")

    def test_bad_top_p_raises(self):
        for bad in (0.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                DecodingConfig(top_p=bad)

    def test_label_mentions_parameters(self):
        cfg = DecodingConfig(strategy="topp", top_p=0.9, temperature=0.8)
        self.assertIn("0.9", cfg.label())
        self.assertIn("T=0.8", cfg.label())


class TestGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lm = NGramLM(order=3)

    def test_backend_produces_probabilities(self):
        logits = self.lm.next_token_logits(self.lm.encode("the future of"))
        probs = [math.exp(x) for x in logits]
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

    def test_greedy_is_reproducible(self):
        cfg = DecodingConfig(strategy="greedy", max_new_tokens=10)
        a = generate(self.lm, "the future of", cfg)
        b = generate(self.lm, "the future of", cfg)
        self.assertEqual(a.continuation, b.continuation)

    def test_seeded_sampling_is_reproducible(self):
        cfg = DecodingConfig(strategy="topp", top_p=0.9, max_new_tokens=12,
                             seed=5)
        a = generate(self.lm, "the future of", cfg)
        b = generate(self.lm, "the future of", cfg)
        self.assertEqual(a.continuation, b.continuation)

    def test_different_seeds_usually_differ(self):
        cfg = DecodingConfig(strategy="topp", top_p=0.95, max_new_tokens=25)
        texts = {generate(self.lm, "the future of",
                          cfg.with_(seed=s)).continuation for s in range(8)}
        self.assertGreater(len(texts), 1)

    def test_stop_strings(self):
        cfg = DecodingConfig(strategy="greedy", max_new_tokens=40,
                             stop=(".",))
        result = generate(self.lm, "the future of", cfg)
        self.assertLess(result.num_tokens, 40)

    def test_max_tokens_respected(self):
        cfg = DecodingConfig(strategy="topp", top_p=0.99, max_new_tokens=7)
        result = generate(self.lm, "machine learning", cfg)
        self.assertLessEqual(result.num_tokens, 7)

    def test_all_strategies_smoke(self):
        for strategy in STRATEGIES:
            cfg = DecodingConfig(strategy=strategy, max_new_tokens=6, seed=1)
            result = generate(self.lm, "the future of", cfg)
            self.assertGreaterEqual(result.num_tokens, 1,
                                    msg=f"{strategy} produced nothing")


if __name__ == "__main__":
    unittest.main()
