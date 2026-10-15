import glob
import json
import os
import pickle
import re
import threading
from pathlib import Path
from typing import Optional
from collections import Counter
import requests
import faiss
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BM25_INDEX_PATH = str(SCRIPT_DIR / "WebSearchEnv" / "indexes" / "bm25")
DEFAULT_FAISS_INDEX_GLOB = str(SCRIPT_DIR / "WebSearchEnv" / "indexes" / "qwen3-embedding-8b" / "corpus.*.pkl")
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_EMBED_API_URL = os.getenv("VLLM_EMBEDDING_URL", "http://localhost:8001/v1/embeddings")
DEFAULT_SEARCH_TOP_K = 10
DEFAULT_SEARCH_MODE = "hybrid"
DEFAULT_QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
MAX_DOCUMENT_TEXT_CHARS = 10000
FAISS_NUM_THREADS = 2


_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her", "it",
    "its", "they", "them", "their", "this", "that", "these", "those",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "into", "about", "between", "through", "during", "before", "after",
    "and", "but", "or", "not", "no", "if", "then", "so", "than",
    "who", "which", "what", "when", "where", "how", "all", "each",
    "any", "both", "few", "more", "most", "some", "such", "very",
}



# ---------------------------------------------------------------------------
# Answer normalization & extraction
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Normalize answers for matching. Preserves unicode letters."""
    import unicodedata
    normalized = str(text).strip().lower()
    # Decompose unicode and remove combining marks (accents) for fuzzy matching
    normalized = unicodedata.normalize("NFD", normalized)
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    # Remove non-alphanumeric (but keep spaces)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _split_gold_alternatives(gold: str) -> list[str]:
    """Split gold answer into alternative forms.
    
    Handles patterns like:
    - "Marie of Romania; Queen consort of Romania" → ["Marie of Romania", "Queen consort of Romania"]
    - "Jenny Karezi (Tzeni Karezi)" → ["Jenny Karezi", "Tzeni Karezi"]
    - "Claude Beri Langmann (or Claude Berel Langmann)" → ["Claude Beri Langmann", "Claude Berel Langmann"]
    - "Gautsch (known professionally as Françoise Arnoul)" → ["Gautsch", "Françoise Arnoul"]
    - "Professor (or Dr.) James Russell" → ["Professor James Russell", "Dr. James Russell"]
    - "Name (ABBR) / Other Name" → ["Name", "ABBR", "Other Name"]
    """
    alternatives = []
    
    # Split by semicolons and " / " (slash with surrounding spaces)
    parts = re.split(r"\s*;\s*|\s+/\s+", gold)
    parts = [p.strip() for p in parts if p.strip()]
    
    for part in parts:
        # Extract text before parentheses, inside parentheses, and after parentheses
        if "(" in part and ")" in part:
            before_paren = part[:part.index("(")].strip()
            after_paren = part[part.rindex(")") + 1:].strip()
            
            # Main alternative: text before + text after parentheses
            main = f"{before_paren} {after_paren}".strip() if before_paren else after_paren
            if main:
                alternatives.append(main)
            
            # Extract text inside parentheses (remove common prefixes)
            paren_content = re.findall(r"\(([^)]+)\)", part)
            for content in paren_content:
                cleaned = re.sub(
                    r"^(?:or\s+(?:possibly\s+)?"
                    r"|also\s+(?:known|spelled|written|referred\s+to)(?:\s+as)?\s+"
                    r"|(?:commonly\s+)?known\s+(?:professionally\s+)?as\s+)",
                    "", content.strip(), flags=re.IGNORECASE,
                )
                if cleaned:
                    alt = f"{cleaned.strip()} {after_paren}".strip() if after_paren else cleaned.strip()
                    alternatives.append(alt)
        else:
            alternatives.append(part)
    
    return alternatives if alternatives else [gold]


def answer_matches(prediction: str, gold: str) -> bool:
    """Check if prediction matches any alternative form of the gold answer.
    
    For each gold alternative, checks if the normalized alternative appears as 
    a whole-word substring in the normalized prediction, or vice versa.
    Uses word-boundary matching to avoid false positives (e.g. gold "9" matching "1992").
    """
    norm_pred = normalize_answer(prediction)
    if not norm_pred:
        return not normalize_answer(gold)
    
    for alt in _split_gold_alternatives(gold):
        norm_alt = normalize_answer(alt)
        if not norm_alt:
            continue
        # Word-boundary substring matching in both directions
        if re.search(r'\b' + re.escape(norm_alt) + r'\b', norm_pred):
            return True
        if re.search(r'\b' + re.escape(norm_pred) + r'\b', norm_alt):
            return True
    return False


def extract_exact_answer(text: str) -> str:
    """Extract the short answer from verbose gold/prediction text.

    Gold answers in the bundle data contain an ``Exact Answer:`` line.
    """
    if not text:
        return ""
    raw = str(text).strip()
    # Strip markdown formatting: bold/italic/code markers and heading prefixes
    label_text = re.sub(r"[*_`]+", "", raw)
    label_text = re.sub(r"^#+\s*", "", label_text, flags=re.MULTILINE)
    # Try labelled lines first
    for label in ["Exact Answer", "Final Answer", "Answer"]:
        m = re.search(
            rf"^\s*{label}\s*:\s*(.+?)\s*$",
            label_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if m:
            return m.group(1).strip()
    # Fallback: return last non-empty line (skip "Confidence:" lines)
    for line in reversed(label_text.splitlines()):
        line = line.strip()
        if line and not re.match(r"^confidence\s*:", line, flags=re.IGNORECASE):
            return line
    return raw



class BrowseCompPlusEnv:
    """
    Environment for the BrowseComp-Plus benchmark.

    Supports both BM25 (Pyserini Lucene) and dense (FAISS + Qwen3-Embedding)
    search over the ~100K document corpus.
    """

    def __init__(
        self,
        bm25_index_path: str = DEFAULT_BM25_INDEX_PATH,
        search_top_k: int = DEFAULT_SEARCH_TOP_K,
        snippet_max_chars: int = 512,
        search_mode: str = DEFAULT_SEARCH_MODE,
        faiss_index_glob: str = DEFAULT_FAISS_INDEX_GLOB,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        embed_api_url: str = DEFAULT_EMBED_API_URL,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
    ):
        self.search_top_k = search_top_k
        self.snippet_max_chars = snippet_max_chars
        self.search_mode = search_mode
        self._lucene_lock = threading.RLock()

        # Always init BM25 for get_document (retrieve full text by docid)
        self._init_lucene_searcher(bm25_index_path)
        # Init FAISS and embedding model if needed
        if search_mode in ("embedding", "hybrid"):
            self._init_faiss_searcher(
                faiss_index_glob,
                embed_model_name,
                embed_api_url,
                query_prefix,
            )


    def _init_lucene_searcher(self, index_path: str) -> None:
        """Initialize the Pyserini LuceneSearcher for full-corpus BM25."""
        try:
            os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
            from pyserini.search.lucene import LuceneSearcher
        except ImportError:
            raise ImportError(
                "pyserini is required for BM25 search. "
                "Install it with: pip install pyserini"
            )
        print(f"Initializing BM25 searcher with index: {index_path}")
        self._lucene_searcher = LuceneSearcher(index_path)
        print("BM25 searcher initialized successfully")

    def _init_faiss_searcher(
        self,
        index_glob: str,
        model_name: str,
        api_url: str,
        query_prefix: str,
    ) -> None:
        """Initialize FAISS index and vLLM embedding endpoint for dense retrieval."""
        faiss.omp_set_num_threads(FAISS_NUM_THREADS)
        self._query_prefix = query_prefix
        self._embed_model_name = model_name
        self._embed_api_url = api_url

        # --- Load FAISS index from pickle shards ---
        shard_files = sorted(glob.glob(index_glob))
        if not shard_files:
            raise FileNotFoundError(
                f"No FAISS index shards found matching: {index_glob}"
            )
        print(f"Loading {len(shard_files)} FAISS shard(s)...")
        self._faiss_lookup: list[str] = []
        faiss_index = None
        for path in shard_files:
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            reps = np.array(reps, dtype=np.float32)
            if faiss_index is None:
                faiss_index = faiss.IndexFlatIP(reps.shape[1])
            faiss_index.add(reps)
            self._faiss_lookup.extend(lookup)

        print("Using FAISS index on CPU")

        self._faiss_index = faiss_index
        print(f"FAISS index loaded: {len(self._faiss_lookup)} vectors")
        print(f"Using vLLM embedding endpoint: {api_url}")

    def _extract_snippet(self, text: str, query: str) -> str:
        """Extract a snippet with a document header and the most query-relevant window."""
        max_chars = self.snippet_max_chars
        if len(text) <= max_chars:
            return text

        # --- 1. Header: document opening for entity/topic identification ---
        header_budget = min(128, max_chars // 4)
        header_end = text.rfind(" ", 0, header_budget)
        if header_end <= 0:
            header_end = header_budget
        header = text[:header_end].rstrip()

        # --- 2. Find best query-matching window ---
        query_terms = set(t.strip(".,;:!?\"'()[]{}") for t in query.lower().split())
        query_terms -= _STOP_WORDS
        query_terms.discard("")

        win_start = -1  # sentinel: no useful window found
        window_budget = max_chars - len(header) - 7  # 7 = len(" ...") + len("...")

        if query_terms and window_budget >= 80:
            words = text.split()
            n = len(words)
            cleaned = [w.lower().strip(".,;:!?\"'()[]{}") for w in words]
            match = [c if c in query_terms else None for c in cleaned]

            # Build character offsets
            offsets = []
            pos = 0
            for w in words:
                idx = text.find(w, pos)
                offsets.append(idx)
                pos = idx + len(w)

            # Sliding window maximising (distinct_terms, total_hits)
            best = (0, 0)
            counts: Counter = Counter()
            j = 0
            for i in range(n):
                while j < n and offsets[j] + len(words[j]) - offsets[i] <= window_budget:
                    if match[j]:
                        counts[match[j]] += 1
                    j += 1
                score = (len(counts), sum(counts.values()))
                if score > (0, 0) and score >= best:
                    best = score
                    win_start = offsets[i]
                if match[i]:
                    counts[match[i]] -= 1
                    if not counts[match[i]]:
                        del counts[match[i]]

        # --- 3. Assemble ---
        # No useful window, or window overlaps header → return document prefix
        if win_start <= header_end + 20:
            return text[:max_chars - 3] + "..."

        # If window hits the end of the document, shift start backward to
        # fill the budget and provide more surrounding context.
        win_end = min(win_start + window_budget, len(text))
        if win_end - win_start < window_budget:
            win_start = max(header_end + 21, win_end - window_budget)
        tail = "..." if win_end < len(text) else ""
        return f"{header} ...{text[win_start:win_end]}{tail}"


    def bm25_search(self, query: str, k: Optional[int] = None) -> str:
        """
        Search the full BrowseComp-Plus corpus via BM25.

        Returns a JSON list:
        [
            {"docid": str, "snippet": str},
            ...
        ]
        """
        k = k or self.search_top_k
        results = []
        with self._lucene_lock:
            hits = self._lucene_searcher.search(query, k)
            for hit in hits:
                raw = json.loads(hit.lucene_document.get("raw"))
                text = raw.get("contents", "")
                results.append({
                    "docid": hit.docid,
                    # "score": round(hit.score, 6),
                    "snippet": self._extract_snippet(text, query),
                })
        return json.dumps(results, ensure_ascii=False, indent=2)

    def embedding_search(self, query: str, k: Optional[int] = None) -> str:
        """
        Search the full BrowseComp-Plus corpus via dense embedding (FAISS).

        Returns a JSON list:
        [
            {"docid": str, "snippet": str},
            ...
        ]
        """
        k = k or self.search_top_k

        # Encode query via vLLM embedding endpoint
        resp = requests.post(
            self._embed_api_url,
            json={
                "model": self._embed_model_name,
                "input": self._query_prefix + query,
            },
        )
        resp.raise_for_status()
        q_reps = np.array(
            [resp.json()["data"][0]["embedding"]], dtype=np.float32
        )

        faiss.omp_set_num_threads(FAISS_NUM_THREADS)
        scores, indices = self._faiss_index.search(q_reps, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            docid = self._faiss_lookup[idx]
            with self._lucene_lock:
                doc = self._lucene_searcher.doc(docid)
                if doc is not None:
                    raw = json.loads(doc.raw())
                    text = raw.get("contents", "")
                else:
                    text = ""
            results.append({
                "docid": docid,
                # "score": round(float(score), 6),
                "snippet": self._extract_snippet(text, query),
            })
        return json.dumps(results, ensure_ascii=False, indent=2)

    def hybrid_search(self, query: str, k: Optional[int] = None) -> str:
        """
        Search via both BM25 and embedding, merge results with dedup.
        
        Returns a JSON list:
        [
            {"docid": str, "snippet": str},
            ...
        ]
        """
        k = k or self.search_top_k
        # half from BM25, half from embedding, then dedup
        bm25_results = json.loads(self.bm25_search(query, k//2+1))
        emb_results = json.loads(self.embedding_search(query, k//2+1))
        seen: set[str] = set()
        merged = []
        for r in bm25_results + emb_results:
            if r["docid"] not in seen:
                seen.add(r["docid"])
                merged.append(r)
        return json.dumps(merged, ensure_ascii=False, indent=2)

    def search(self, query: str, k: Optional[int] = None) -> str:
        """Dispatch to the configured search backend."""
        if self.search_mode == "hybrid":
            return self.hybrid_search(query, k)
        elif self.search_mode == "embedding":
            return self.embedding_search(query, k)
        return self.bm25_search(query, k)


    def get_document(self, docid: str) -> str:
        """Retrieve one full document by docid from the corpus index."""
        with self._lucene_lock:
            doc = self._lucene_searcher.doc(docid)
            if doc is None:
                return json.dumps(
                    {"error": f"docid {docid} not found in the corpus index"},
                    ensure_ascii=False,
                    indent=2,
                )
            raw = json.loads(doc.raw())
        text = raw.get("contents", "")
        if len(text) > MAX_DOCUMENT_TEXT_CHARS:
            text = text[:MAX_DOCUMENT_TEXT_CHARS] + "..."
        return json.dumps(
            {"docid": docid, "text": text},
            ensure_ascii=False,
            indent=2,
        )