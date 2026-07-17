"""
Candidate pruning for ontology alignment using lightweight string similarity.

Reduces the O(|source| * |target|) cross-product of concept pairs down to a
much smaller candidate set before sending anything to an LLM.

Similarity combines:
  - normalized exact match (lowercase, underscores/dashes/camelCase -> spaces)
  - substring containment (one label contained in the other)
  - subword/token Jaccard overlap
  - difflib fuzzy ratio (character-level, catches near-misses/typos)

No external dependencies (uses stdlib `re` and `difflib`).
"""

import re
from difflib import SequenceMatcher
from typing import Iterable, List, Sequence, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_label(text: str) -> str:
    """
    Lowercase, split camelCase, and turn underscores/dashes/other separators
    into single spaces. E.g. "HasChemical_Property-ID" -> "has chemical property id"
    """
    if not text:
        return ""
    text = _CAMEL_BOUNDARY.sub(" ", text)          # split camelCase
    text = _NON_ALNUM.sub(" ", text.lower())        # kill _, -, punctuation
    return re.sub(r"\s+", " ", text).strip()


def get_subwords(text: str) -> set:
    """Return the set of normalized subword tokens for a label."""
    return set(normalize_label(text).split())


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------

def token_jaccard(a_tokens: set, b_tokens: set) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


def fuzzy_ratio(a_norm: str, b_norm: str) -> float:
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def label_similarity(a: str, b: str) -> float:
    """
    Combined similarity score in [0, 1] between two raw labels.
    Weighted blend: exact/substring gets strong credit, token overlap and
    fuzzy ratio fill in for partial/near matches.
    """
    a_norm, b_norm = normalize_label(a), normalize_label(b)
    if not a_norm or not b_norm:
        return 0.0

    if a_norm == b_norm:
        return 1.0

    a_tokens, b_tokens = set(a_norm.split()), set(b_norm.split())
    jaccard = token_jaccard(a_tokens, b_tokens)
    ratio = fuzzy_ratio(a_norm, b_norm)

    # substring containment (handles e.g. "cell" vs "cell type")
    containment = 0.0
    if a_norm in b_norm or b_norm in a_norm:
        shorter, longer = sorted([a_norm, b_norm], key=len)
        containment = len(shorter) / len(longer)

    return max(0.5 * jaccard + 0.5 * ratio, containment)


# ---------------------------------------------------------------------------
# Candidate pruning over full concept lists
# ---------------------------------------------------------------------------

def _extract_label(concept: Any, label_key: str) -> str:
    """Concept can be a dict (OntoAligner concept repr) or a plain string."""
    if isinstance(concept, dict):
        # OntoAligner concept dicts commonly use 'label' or 'name'
        return concept.get(label_key) or concept.get("label") or concept.get("name") or ""
    return str(concept)


def prune_candidates(
    source_concepts: Sequence[Any],
    target_concepts: Sequence[Any],
    threshold: float = 0.4,
    label_key: str = "label",
    top_k: int = None,
) -> List[Tuple[int, int, float]]:
    """
    Compare every source concept against every target concept using
    label_similarity, and keep only pairs scoring >= threshold.

    Args:
        source_concepts: list of concept dicts/strings from dataset['source']
        target_concepts: list of concept dicts/strings from dataset['target']
        threshold: minimum similarity score to keep a pair (0-1)
        label_key: dict key holding the concept's display label
        top_k: if set, keep only the top_k best target matches per source
                concept (applied after thresholding)

    Returns:
        List of (source_index, target_index, score) tuples, sorted by score desc.
    """
    # Pre-normalize once to avoid recomputation in the double loop
    src_norms = [normalize_label(_extract_label(c, label_key)) for c in source_concepts]
    tgt_norms = [normalize_label(_extract_label(c, label_key)) for c in target_concepts]
    src_tokens = [set(n.split()) for n in src_norms]
    tgt_tokens = [set(n.split()) for n in tgt_norms]

    # Bucket target indices by first token for a cheap inverted-index prefilter.
    # This avoids full O(n*m) fuzzy_ratio calls when ontologies are large.
    from collections import defaultdict
    tgt_by_token: Dict[str, List[int]] = defaultdict(list)
    for j, toks in enumerate(tgt_tokens):
        for t in toks:
            tgt_by_token[t].append(j)

    results: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)

    for i, (s_norm, s_toks) in enumerate(zip(src_norms, src_tokens)):
        if not s_norm:
            continue

        # candidate target indices: anything sharing at least one token,
        # plus we still fall back to nothing if there's zero token overlap
        # (rare — means labels share no subwords at all, likely not a match)
        candidate_js = set()
        for t in s_toks:
            candidate_js.update(tgt_by_token.get(t, []))

        for j in candidate_js:
            # reuse precomputed tokens/norms directly for speed:
            jacc = token_jaccard(s_toks, tgt_tokens[j])
            ratio = fuzzy_ratio(s_norm, tgt_norms[j])
            containment = 0.0
            if s_norm and tgt_norms[j] and (s_norm in tgt_norms[j] or tgt_norms[j] in s_norm):
                shorter, longer = sorted([s_norm, tgt_norms[j]], key=len)
                containment = len(shorter) / len(longer)
            score = max(0.5 * jacc + 0.5 * ratio, containment)

            if score >= threshold:
                results[i].append((i, j, score))

    pairs: List[Tuple[int, int, float]] = []
    for i, lst in results.items():
        lst.sort(key=lambda x: x[2], reverse=True)
        if top_k:
            lst = lst[:top_k]
        pairs.extend(lst)

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


if __name__ == "__main__":
    # quick smoke test
    source = ["HasChemicalProperty", "Cell_Type", "boiling-point", "Mass"]
    target = ["chemical_property", "cellType", "BoilingPoint", "AtomicMass", "Color"]

    cand = prune_candidates(source, target, threshold=0.35)
    for i, j, score in cand:
        print(f"{source[i]!r:30s} <-> {target[j]!r:20s} score={score:.3f}")
