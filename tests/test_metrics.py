"""Tests for the quality metrics in lab.metrics."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.backend import NGramLM
from lab.decoding import DecodingConfig, generate
from lab.metrics import (ReferenceIndex, aggregate_reports, analyze,
                         char_entropy, distinct_n, factuality, longest_repeated_ngram,
                         prompt_overlap, relevance_score, repetition_index,
                         split_sentences, topic_fit, word_entropy)
from lab.safety import load_knowledge

CORPUS = (Path(__file__).resolve().parent.parent / "data" / "corpus.txt").read_text(
    encoding="utf-8")

REPETITIVE = "the model the model the model the model the model the model"
VARIED = ("transformer models sample tokens from a probability distribution "
          "using greedy decoding temperature scaling or nucleus filtering "
          "strategies")


class TestDiversityAndRepetition(unittest.TestCase):
    def test_distinct_n_bounds(self):
        self.assertEqual(distinct_n(VARIED, 1), 1.0)
        self.assertGreater(distinct_n(REPETITIVE, 1), 0.0)
        self.assertLess(distinct_n(REPETITIVE, 2), 0.5)
        self.assertGreater(distinct_n(VARIED, 2), 0.8)

    def test_repetition_is_one_minus_distinct(self):
        self.assertAlmostEqual(repetition_index(REPETITIVE, 2),
                               1 - distinct_n(REPETITIVE, 2))

    def test_empty_text(self):
        self.assertEqual(distinct_n("", 2), 0.0)
        self.assertEqual(repetition_index("", 2), 1.0)

    def test_longest_repeat(self):
        phrase, count = longest_repeated_ngram(REPETITIVE)
        self.assertGreaterEqual(count, 3)
        self.assertIn("the model", phrase)


class TestEntropy(unittest.TestCase):
    def test_entropy_zero_for_constant(self):
        self.assertEqual(word_entropy("blah blah blah blah"), 0.0)

    def test_entropy_higher_for_varied(self):
        self.assertGreater(word_entropy(VARIED), word_entropy(REPETITIVE))

    def test_char_entropy(self):
        self.assertEqual(char_entropy("aaaa"), 0.0)
        self.assertGreater(char_entropy(VARIED), 2.0)


class TestCoherence(unittest.TestCase):
    def setUp(self):
        self.ref = ReferenceIndex(CORPUS)

    def test_reference_continuity_high_for_corpus_text(self):
        text = "transformer models learn patterns from large collections"
        self.assertGreater(self.ref.continuity(text), 0.5)

    def test_reference_continuity_low_for_nonsense(self):
        self.assertEqual(self.ref.continuity("zz qq xx ww vv"), 0.0)


class TestRelevance(unittest.TestCase):
    def test_prompt_overlap(self):
        prompt = "the future of artificial intelligence"
        text = "artificial intelligence will reshape the future of society"
        self.assertGreater(prompt_overlap(prompt, text), 0.5)

    def test_topic_fit(self):
        self.assertGreater(topic_fit("machine learning models and data"),
                           0.0)
        self.assertEqual(topic_fit("cat dog bird fish"), 0.0)

    def test_relevance_score_range(self):
        score = relevance_score("the future of artificial intelligence",
                                "transformer models learn from data")
        self.assertTrue(0.0 <= score <= 1.0)


class TestFactuality(unittest.TestCase):
    def setUp(self):
        self.kb = load_knowledge()

    def test_sentences_split(self):
        parts = split_sentences("One. Two! Three?")
        self.assertEqual(len(parts), 3)

    def test_verified_fact(self):
        report = factuality("Water boils at 100 degrees Celsius.", self.kb)
        self.assertEqual(len(report.verified), 1)
        self.assertEqual(report.contradictions, [])

    def test_contradiction_detected(self):
        report = factuality("Vaccines cause autism.", self.kb)
        self.assertEqual(len(report.contradictions), 1)
        self.assertLess(report.score, 0.5)

    def test_clean_text_scores_full(self):
        report = factuality("Sampling strategies shape generation quality.",
                            self.kb)
        self.assertEqual(report.score, 1.0)


class TestAnalyzeEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lm = NGramLM(order=3)
        cls.kb = load_knowledge()
        cls.ref = ReferenceIndex(CORPUS)

    def test_analyze_produces_all_metrics(self):
        result = generate(self.lm, "the future of artificial intelligence is",
                          DecodingConfig(strategy="topp", top_p=0.9,
                                         max_new_tokens=20, seed=3))
        report = analyze(result, self.lm, self.kb, self.ref)
        for key in ("distinct_1", "distinct_2", "repetition", "randomness",
                    "coherence", "relevance", "factuality", "safety",
                    "helpfulness"):
            self.assertIn(key, report.metrics)

    def test_greedy_has_zero_randomness(self):
        result = generate(self.lm, "the future of artificial intelligence is",
                          DecodingConfig(strategy="greedy", max_new_tokens=15))
        report = analyze(result, self.lm, self.kb, self.ref)
        self.assertEqual(report.metrics["randomness"], 0.0)

    def test_uniform_random_has_max_randomness(self):
        result = generate(self.lm, "the future of artificial intelligence is",
                          DecodingConfig(strategy="random", max_new_tokens=15,
                                         seed=9))
        report = analyze(result, self.lm, self.kb, self.ref)
        self.assertGreater(report.metrics["randomness"], 0.9)

    def test_aggregate_reports(self):
        reports = [analyze(generate(self.lm, "machine learning",
                                    DecodingConfig(strategy="greedy",
                                                   max_new_tokens=10)),
                           self.lm, self.kb, self.ref)]
        totals = aggregate_reports(reports)
        self.assertIn("distinct_2", totals)


if __name__ == "__main__":
    unittest.main()
