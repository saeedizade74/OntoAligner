

import re
from difflib import SequenceMatcher
from collections import defaultdict
from collections import Counter
import math

# ---------------------------------------------------------------------------
# Optional WordNet support
# ---------------------------------------------------------------------------

try:
    import nltk
    from nltk.corpus import wordnet as wn
    _HAS_WORDNET = True
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except ImportError:
    _HAS_WORDNET = False


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_label(text):
    """
    'HasChemical_Property-ID' -> 'has chemical property id'
    'BankPaymentInKindInterest' -> 'bank payment in kind interest'
    """
    if not text:
        return ""
    text = _CAMEL_BOUNDARY.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def get_tokens(text):
    return set(normalize_label(text).split())


# ---------------------------------------------------------------------------
# WordNet helpers
# ---------------------------------------------------------------------------

# small cache so we don't re-query synsets thousands of times
_synset_cache = {}
_wn_sim_cache = {}

# short function words that shouldn't drive similarity
_STOP_WORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for",
    "is", "are", "was", "were", "be", "has", "had", "have",
    "and", "or", "but", "not", "with", "by", "from", "as",
})


def _get_synsets(word):
    if word not in _synset_cache:
        _synset_cache[word] = wn.synsets(word) if _HAS_WORDNET else []
    return _synset_cache[word]


def wordnet_word_sim(w1, w2):
    """
    Best WordNet path similarity between two individual words.
    Returns 0.0-1.0, or 0.0 if no relation found.
    """
    if not _HAS_WORDNET:
        return 0.0

    key = (w1, w2) if w1 <= w2 else (w2, w1)
    if key in _wn_sim_cache:
        return _wn_sim_cache[key]

    best = 0.0
    for s1 in _get_synsets(w1):
        for s2 in _get_synsets(w2):
            sim = s1.path_similarity(s2)
            if sim and sim > best:
                best = sim
    _wn_sim_cache[key] = best
    return best


def wordnet_token_similarity(tokens_a, tokens_b):
    """
    For each content word in the smaller set, find the best WordNet match
    in the larger set. Returns average of best matches.
    Gives partial credit for synonyms/hypernyms (money~cash) without
    inflating scores for stop words.
    """
    if not _HAS_WORDNET:
        return 0.0

    # filter out stop words — they shouldn't drive similarity
    a = [t for t in tokens_a if t not in _STOP_WORDS and len(t) > 2]
    b = [t for t in tokens_b if t not in _STOP_WORDS and len(t) > 2]

    if not a or not b:
        return 0.0

    # always iterate over the smaller set
    if len(a) > len(b):
        a, b = b, a

    total = 0.0
    for wa in a:
        best = 0.0
        for wb in b:
            if wa == wb:
                best = 1.0
                break
            sim = wordnet_word_sim(wa, wb)
            if sim > best:
                best = sim
        total += best

    return total / len(a)


# ---------------------------------------------------------------------------
# Core similarity scoring
# ---------------------------------------------------------------------------

def token_jaccard(a_list, b_list):
    if not a_list or not b_list:
        return 0.0
    inter = len(list((Counter(a_list) & Counter(b_list)).elements()))
    union = len(list((Counter(a_list) | Counter(b_list)).elements()))
    return inter / union if union else 0.0


def containment_score(a_norm, b_norm):
    """How much of the shorter string is contained in the longer one."""
    if not a_norm or not b_norm:
        return 0.0
    if a_norm in b_norm or b_norm in a_norm:
        shorter, longer = sorted([a_norm, b_norm], key=len)
        return len(shorter) / len(longer)
    return 0.0


def fuzzy_ratio(a_norm, b_norm):
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def token_overlap_ratio(a_tokens, b_tokens):
    """
    What fraction of the smaller token set appears in the larger one.
    Catches cases like {payment, in, kind, interest} vs {interest, payment, in, kind}
    where Jaccard is high, but also partial overlaps where the smaller set
    is mostly covered.
    """
    if not a_tokens or not b_tokens:
        return 0.0
    smaller, larger = sorted([a_tokens, b_tokens], key=len)
    if not smaller:
        return 0.0
    return len(smaller & larger) / len(smaller)


def label_similarity(a, b):
    """
    Combined similarity in [0, 1] between two raw labels.

    Handles:
      - exact match after normalization
      - word reordering (token Jaccard + overlap ratio)
      - substring containment
      - character-level fuzzy matching
      - WordNet synonym similarity (if nltk available)

    Weights:
      - token_jaccard:     0.30  (order-independent word overlap)
      - overlap_ratio:     0.20  (how much of smaller set is covered)
      - fuzzy_ratio:       0.15  (character-level, catches typos)
      - containment:       0.10  (one label inside another)
      - wordnet_sim:       0.25  (semantic similarity for non-identical words)
    """
    a_norm = normalize_label(a)
    b_norm = normalize_label(b)

    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0

    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())

    jacc = token_jaccard(a_tokens, b_tokens)
    overlap = token_overlap_ratio(a_tokens, b_tokens)
    ratio = fuzzy_ratio(a_norm, b_norm)
    contain = containment_score(a_norm, b_norm)

    if _HAS_WORDNET:
        wn_sim = wordnet_token_similarity(a_tokens, b_tokens)
        score = (
            0.30 * jacc
            + 0.20 * overlap
            + 0.15 * ratio
            + 0.10 * contain
            + 0.25 * wn_sim
        )
    else:
        # without wordnet, redistribute weight to token-level signals
        score = (
            0.35 * jacc
            + 0.25 * overlap
            + 0.25 * ratio
            + 0.15 * contain
        )

    # containment alone can indicate a strong match (e.g. "cell" in "cell type")
    # so use it as a floor
    return max(score, contain * 0.8)


# ---------------------------------------------------------------------------
# Concept label extraction
# ---------------------------------------------------------------------------

def _extract_label(concept, label_key="label"):
    if isinstance(concept, dict):
        return (
            concept.get(label_key)
            or concept.get("label")
            or concept.get("iri", "").split("/")[-1]
            or ""
        )
    return str(concept)


# ---------------------------------------------------------------------------
# Candidate pruning
# ---------------------------------------------------------------------------
def FinalScore(scores):
    return math.sqrt(sum(x**2 for x in scores) / len(scores))       

def prune_candidates(source_concepts, target_concepts,
                     threshold=0.4, label_key="label", top_k=None):
    """
    Compare every source against every target using label_similarity.
    Keep pairs scoring >= threshold.

    Uses an inverted index on tokens to avoid full O(n*m) for large ontologies,
    but falls back to brute force for sources with no token overlap (rare).
    """
    src_norms = [normalize_label(_extract_label(c, label_key)) for c in source_concepts]
    tgt_norms = [normalize_label(_extract_label(c, label_key)) for c in target_concepts]
    src_tokens = [set(n.split()) for n in src_norms]
    tgt_tokens = [set(n.split()) for n in tgt_norms]

    # inverted index: token -> list of target indices
    tgt_by_token = defaultdict(list)
    for j, toks in enumerate(tgt_tokens):
        for t in toks:
            tgt_by_token[t].append(j)

    results = defaultdict(list)

    for i, (s_norm, s_toks) in enumerate(zip(src_norms, src_tokens)):
        if not s_norm:
            continue

        # find candidate targets sharing at least one token
        candidate_js = set()
        for t in s_toks:
            candidate_js.update(tgt_by_token.get(t, []))

        for j in candidate_js:
            t_norm = tgt_norms[j]
            t_toks = tgt_tokens[j]

            jacc = token_jaccard(s_toks, t_toks)
            overlap = token_overlap_ratio(s_toks, t_toks)
            ratio = fuzzy_ratio(s_norm, t_norm)
            contain = containment_score(s_norm, t_norm)

            if _HAS_WORDNET:
                wn_sim = wordnet_token_similarity(s_toks, t_toks)
                score = FinalScore(
                    [jacc,
                     overlap,
                     ratio,
                     contain,
                     wn_sim])
            else:
                score = FinalScore(
                   [jacc,
                     overlap,
                     ratio,
                     contain])


            if score >= threshold:
                results[i].append((i, j, score))

    pairs = []
    for i, lst in results.items():
        lst.sort(key=lambda x: x[2], reverse=True)
        if top_k:
            lst = lst[:top_k]
        pairs.extend(lst)

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs

