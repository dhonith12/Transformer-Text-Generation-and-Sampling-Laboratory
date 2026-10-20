"""Language-model backends used by the laboratory.

The decoding algorithms in :mod:`lab.decoding` only ever need a stream of
logits, so anything that answers ``next_token_logits(context_ids)`` can be
plugged in.  Three backends are provided:

``ngram`` (default)
    A pure-Python n-gram model (stupid backoff) trained on the bundled
    corpus in ``data/corpus.txt``.  No third-party dependency, instant start,
    fully offline — the reference backend for the classroom.

``hf``
    A pretrained HuggingFace decoder (default ``gpt2``).  Requires ``torch``
    and ``transformers`` plus downloaded weights.

``tiny``
    A miniature GPT trained from scratch by :mod:`lab.train_tiny`
    (requires ``torch``).  Useful to show that the same decoding code works
    on a model you trained yourself.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "data" / "corpus.txt"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "tiny_gpt.pt"

TOKEN_RE = re.compile(r"\w+|[^\w\s]")

# Decoding cosmetics -------------------------------------------------------
NO_SPACE_BEFORE = set(".,!?;:%)]}'\u2019")
NO_SPACE_AFTER = set("([{")
SENTENCE_END = re.compile(r"(^|[.!?]\s+)([a-z])")

_MISSING_DEP = (
    "'{pkg}' is not installed. Install the optional dependencies first, e.g.\n"
    "    pip install torch transformers\n"
    "or fall back to the dependency-free backend with --backend ngram."
)


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------
class WordTokenizer:
    """Whitespace/punctuation tokenizer with a vocabulary fitted on a corpus.

    Word-level keeps the vocabulary small, which lets the bundled n-gram model
    and the tiny transformer produce readable text from a tiny corpus.
    """

    UNK = "<unk>"

    def __init__(self, itos: Sequence[str], lowercase: bool = True) -> None:
        self.itos: List[str] = list(itos)
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self.lowercase = lowercase
        if self.UNK not in self.stoi:
            self.stoi[self.UNK] = len(self.itos)
            self.itos.append(self.UNK)
        self.unk_id = self.stoi[self.UNK]

    # -- construction -----------------------------------------------------
    @classmethod
    def fit(cls, texts: Sequence[str], min_freq: int = 1) -> "WordTokenizer":
        counts: Counter = Counter()
        for text in texts:
            counts.update(cls._split(text, lowercase=True))
        itos = sorted((t for t, c in counts.items() if c >= min_freq),
                      key=lambda t: (-counts[t], t))
        return cls(itos)

    @classmethod
    def from_saved(cls, payload: dict) -> "WordTokenizer":
        return cls(payload["itos"], lowercase=payload.get("lowercase", True))

    def saved(self) -> dict:
        return {"itos": self.itos, "lowercase": self.lowercase}

    # -- encoding ---------------------------------------------------------
    @staticmethod
    def _split(text: str, lowercase: bool) -> List[str]:
        if lowercase:
            text = text.lower()
        return TOKEN_RE.findall(text)

    def encode(self, text: str, add_bos: bool = False) -> List[int]:
        tokens = self._split(text, lowercase=self.lowercase)
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if add_bos and "<bos>" in self.stoi:
            ids.insert(0, self.stoi["<bos>"])
        return ids

    def token_str(self, token_id: int) -> str:
        if 0 <= token_id < len(self.itos):
            return self.itos[token_id]
        return self.UNK

    def decode(self, ids: Sequence[int]) -> str:
        tokens = [self.itos[i] for i in ids
                  if 0 <= i < len(self.itos) and self.itos[i] != self.UNK]
        out = ""
        for tok in tokens:
            if not out:
                out = tok
            elif tok in NO_SPACE_BEFORE:
                out += tok
            elif out[-1] in NO_SPACE_AFTER:
                out += tok
            else:
                out += " " + tok
        return SENTENCE_END.sub(lambda m: m.group(1) + m.group(2).upper(), out)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


# --------------------------------------------------------------------------
# Backend interface
# --------------------------------------------------------------------------
class Backend:
    """Minimal interface every language-model backend must implement."""

    name = "base"

    def encode(self, text: str) -> List[int]:            # pragma: no cover
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:         # pragma: no cover
        raise NotImplementedError

    def next_token_logits(self, context_ids: Sequence[int]) -> List[float]:
        """Raw (unnormalized) scores for the token that follows *context*."""
        raise NotImplementedError

    def token_str(self, token_id: int) -> str:
        return str(token_id)

    @property
    def vocab_size(self) -> int:                          # pragma: no cover
        raise NotImplementedError

    eos_token_id: Optional[int] = None
    masked_ids: set = set()          # tokens that must never be generated

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def log_softmax(logits: Sequence[float]) -> List[float]:
        finite = [x for x in logits if x != -math.inf]
        if not finite:
            return [-math.inf] * len(logits)
        mx = max(finite)
        exps = [math.exp(x - mx) if x != -math.inf else 0.0 for x in logits]
        total = sum(exps) or 1.0
        return [math.log(e / total) if e > 0 else -math.inf for e in exps]

    def sequence_log_probs(self, ids: Sequence[int]) -> List[float]:
        """Teacher-forced log-probability of each token given its prefix."""
        out: List[float] = []
        for i, tid in enumerate(ids):
            lps = self.log_softmax(self.next_token_logits(list(ids[:i])))
            lp = lps[tid] if 0 <= tid < len(lps) else -math.inf
            out.append(lp if lp > -1e9 else math.log(1e-12))
        return out

    def score_text(self, text: str) -> dict:
        """Fluency statistics of *text* under this backend."""
        ids = self.encode(text)
        if not ids:
            return {"tokens": 0, "mean_logprob": 0.0, "perplexity": 0.0,
                    "avg_prob": 0.0}
        lps = self.sequence_log_probs(ids)
        mean_lp = sum(lps) / len(lps)
        ppl = math.exp(-mean_lp)
        return {"tokens": len(ids), "mean_logprob": mean_lp,
                "perplexity": ppl, "avg_prob": math.exp(mean_lp)}

    def describe(self) -> str:
        return f"{self.name} (vocab={self.vocab_size})"


# --------------------------------------------------------------------------
# Backend 1: pure-Python n-gram model
# --------------------------------------------------------------------------
class NGramLM(Backend):
    """n-gram language model with stupid backoff and add-α smoothing.

    logits_w = log P(w | context), where P uses the highest available
    order and multiplies by ``backoff`` (default 0.4) each time it has to
    fall back to a shorter context, as in Katz backoff.
    """

    name = "ngram"

    def __init__(self, corpus_path: Optional[Path] = None, order: int = 3,
                 backoff: float = 0.4, alpha: float = 0.1) -> None:
        path = Path(corpus_path) if corpus_path else DEFAULT_CORPUS
        self.corpus_path = path
        self.order = max(2, order)
        self.backoff = backoff
        self.alpha = alpha
        text = path.read_text(encoding="utf-8")

        self.tok = WordTokenizer.fit([text])
        token_ids = self.tok.encode(text)
        tokens = [self.tok.token_str(i) for i in token_ids]

        self.unigram: Counter = Counter(tokens)
        self.total_unigrams = sum(self.unigram.values())
        # counts[k][context of length k-1] -> Counter of follow-up tokens
        self.counts: Dict[int, Dict[tuple, Counter]] = {
            k: defaultdict(Counter) for k in range(2, self.order + 1)}
        for i, tok in enumerate(tokens):
            for k in range(2, self.order + 1):
                if i >= k - 1:
                    self.counts[k][tuple(tokens[i - k + 1:i])][tok] += 1
        self.totals: Dict[int, Dict[tuple, int]] = {
            k: {ctx: sum(c.values()) for ctx, c in self.counts[k].items()}
            for k in self.counts}

        self.vocab: List[str] = [self.tok.token_str(i)
                                 for i in range(self.tok.vocab_size)]
        self._uni_prob: Dict[str, float] = {
            w: (self.unigram[w] + alpha) /
               (self.total_unigrams + alpha * len(self.vocab))
            for w in self.vocab}
        self.masked_ids = {self.tok.unk_id}

    # -- Backend interface ------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(ids)

    def token_str(self, token_id: int) -> str:
        return self.tok.token_str(token_id)

    def _log_probs(self, ctx: Sequence[str]) -> List[float]:
        """log P(w | ctx) for every id in vocabulary order.

        Stupid backoff produces *scores*, not a normalised distribution, so
        we renormalise over the vocabulary: the result is a proper log-prob
        vector (sums to 1) that can be fed straight into the decoders.
        """
        scores: Dict[str, float] = {}
        penalty = 1.0
        for k in range(min(self.order, len(ctx) + 1), 1, -1):
            table = self.counts[k].get(tuple(ctx[-(k - 1):]))
            if table:
                tot = self.totals[k][tuple(ctx[-(k - 1):])]
                for w, cnt in table.items():
                    scores[w] = penalty * cnt / tot
            penalty *= self.backoff
        probs = [scores.get(w, penalty * self._uni_prob[w]) for w in self.vocab]
        total = sum(probs) or 1.0
        return [math.log(max(p / total, 1e-12)) for p in probs]

    def next_token_logits(self, context_ids: Sequence[int]) -> List[float]:
        ctx = [self.tok.token_str(i) for i in context_ids]
        return self._log_probs(ctx)

    def sequence_log_probs(self, ids: Sequence[int]) -> List[float]:
        out: List[float] = []
        for i, tid in enumerate(ids):
            ctx = [self.tok.token_str(j) for j in ids[:i]]
            lp = self._log_probs(ctx)[tid] if tid < len(self.vocab) \
                else math.log(1e-12)
            out.append(lp)
        return out

    def describe(self) -> str:
        return (f"n-gram LM (order={self.order}, vocab={self.vocab_size}, "
                f"corpus={self.corpus_path.name})")


# --------------------------------------------------------------------------
# Backend 2: HuggingFace pretrained decoder
# --------------------------------------------------------------------------
class HFBackend(Backend):
    """Pretrained causal LM from HuggingFace (default: ``gpt2``)."""

    name = "hf"

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None,
                 local_files_only: bool = False) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - optional dependency
            pkg = "torch" if exc.name == "torch" else "transformers"
            raise RuntimeError(_MISSING_DEP.format(pkg=pkg)) from None

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, local_files_only=local_files_only)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, local_files_only=local_files_only)
        except Exception as exc:  # pragma: no cover - network/weights issue
            raise RuntimeError(
                f"could not load '{model_name}' ({exc.__class__.__name__}: "
                f"{exc}). Use --backend ngram to run fully offline.") from None
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device).eval()
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def token_str(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    def next_token_logits(self, context_ids: Sequence[int]) -> List[float]:
        torch = self._torch
        ctx = list(context_ids) or [self.tokenizer.eos_token_id]
        with torch.no_grad():
            ids = torch.tensor([ctx[-2048:]], device=self.device)
            return self.model(ids).logits[0, -1].tolist()

    def describe(self) -> str:
        return f"HuggingFace {self.model_name} on {self.device}"


# --------------------------------------------------------------------------
# Backend 3: miniature GPT trained from scratch
# --------------------------------------------------------------------------
class TinyBackend(Backend):
    """Miniature GPT checkpoint produced by :func:`lab.train_tiny.train`."""

    name = "tiny"

    def __init__(self, checkpoint: Optional[Path] = None,
                 device: Optional[str] = None) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover - optional dependency
            raise RuntimeError(_MISSING_DEP.format(pkg="torch")) from None
        from .train_tiny import TinyGPT

        path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
        if not path.exists():
            raise RuntimeError(
                f"no checkpoint at {path}. Create one with: "
                f"python run_lab.py train --steps 300")
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.load(path, map_location="cpu")
        self.tok = WordTokenizer.from_saved(data["tokenizer"])
        self.model = TinyGPT(**data["config"])
        self.model.load_state_dict(data["model"])
        self.model.to(self.device).eval()
        self.block_size = self.model.cfg.block_size
        self.checkpoint = path

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(ids)

    def token_str(self, token_id: int) -> str:
        return self.tok.token_str(token_id)

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size

    def next_token_logits(self, context_ids: Sequence[int]) -> List[float]:
        torch = self._torch
        ctx = list(context_ids) or [self.tok.unk_id]
        ctx = ctx[-self.block_size:]
        with torch.no_grad():
            ids = torch.tensor([ctx], device=self.device)
            return self.model(ids).logits[0, -1].tolist()

    def describe(self) -> str:
        return f"tiny GPT checkpoint ({self.checkpoint.name})"


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------
def resolve_backend(name: str = "auto", model: str = "gpt2",
                    corpus: Optional[Path] = None, order: int = 3,
                    checkpoint: Optional[Path] = None,
                    quiet: bool = False) -> Backend:
    """Return a backend.  ``auto`` tries hf -> tiny -> ngram and reports why it
    fell back, so the lab always runs somewhere."""

    def note(msg: str) -> None:
        if not quiet:
            print(f"[backend] {msg}")

    if name == "ngram":
        return NGramLM(corpus, order=order)
    if name == "hf":
        return HFBackend(model)
    if name == "tiny":
        return TinyBackend(checkpoint)

    if name != "auto":
        raise ValueError(f"unknown backend '{name}' "
                         "(choose auto|ngram|hf|tiny)")

    try:
        backend = HFBackend(model)
        note(f"using pretrained transformer '{model}'")
        return backend
    except Exception as exc:
        note(f"hf backend unavailable ({exc})")
    try:
        backend = TinyBackend(checkpoint)
        note("using the trained tiny transformer")
        return backend
    except Exception as exc:
        note(f"tiny backend unavailable ({exc})")
    note("using the dependency-free n-gram backend (data/corpus.txt)")
    return NGramLM(corpus, order=order)
