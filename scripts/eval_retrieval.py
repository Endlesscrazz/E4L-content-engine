"""
Retrieval quality evaluation — two-step evidence for top_k selection.

STEP 1: Offline relevance curve (no LLM calls)
  For 4 representative queries × k=1–15, log cosine similarity at each rank.
  Shows the "elbow" — where adding more chunks stops gaining semantic relevance.

STEP 2: Downstream citation quality sweep (Writer API calls)
  For candidate k values from Step 1 (e.g., 4, 6, 8, 10), run one brief through
  run_writer() and measure citation resolution rate:
    - % of cited_chunk_ids that exist in the provided chunk set (in-context citations)
    - % of cited chunks where cited_substring actually appears in chunk content

Both steps run automatically. Step 2 costs ~$0.05–0.10 (4 Writer calls, Sonnet 4.6).

Usage:
  python scripts/eval_retrieval.py                # both steps
  python scripts/eval_retrieval.py --step1-only   # offline curve only (free)

Output: prints tables + writes results to eval_retrieval_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(".env.local", override=True)
load_dotenv()

from app.clients import AnthropicClient, GeminiEmbedder
from app.corpus_store import get_conn, query_similar
from app.models import Audience, ContentBrief, FunnelStage, Platform, SourceChunk, ToneRegister
from agents.writer import run_writer

# ── Test queries — representative of real briefs we'll demo ──────────────────

TEST_QUERIES = [
    {
        "label": "consumer_fatigue",
        "text": "miHealth device for chronic fatigue cellular energy recovery",
        "audience": "consumer",
    },
    {
        "label": "practitioner_clinical",
        "text": "bioenergetics clinical applications miHealth practitioner protocol",
        "audience": "practitioner",
    },
    {
        "label": "broad_e4l_brand",
        "text": "Energy4Life bioenergetic wellness system body field",
        "audience": "consumer",
    },
    {
        "label": "product_mechanism",
        "text": "miHealth PEMF photobiomodulation magnetic therapy infoceuticals",
        "audience": "practitioner",
    },
]

# k values to sweep in the offline curve
K_RANGE = list(range(1, 16))

# k candidates to test downstream (subset informed by elbow)
K_CANDIDATES = [4, 6, 8, 10, 12]

# Brief used for downstream sweep (consumer/cold — our primary demo scenario)
SWEEP_BRIEF = ContentBrief(
    goal="Introduce the miHealth device to health-conscious consumers experiencing chronic fatigue",
    audience=Audience.CONSUMER,
    funnel_stage=FunnelStage.COLD,
    tone=ToneRegister.CONVERSATIONAL,
    topic_focus="chronic fatigue and cellular energy",
    product_focus=["miHealth"],
)


def l2_to_cosine(l2_dist: float) -> float:
    """Convert L2 distance to cosine similarity for unit-norm vectors.
    gemini-embedding-001 produces unit-norm vectors, so:
      cosine_sim = 1 - L2² / 2
    """
    return max(0.0, 1.0 - (l2_dist ** 2) / 2.0)


def query_with_distances(
    conn,
    query_embedding: list[float],
    top_k: int = 15,
    audience_filter: str | None = None,
) -> list[tuple[SourceChunk, float]]:
    """
    Like query_similar but returns (SourceChunk, l2_distance) pairs.
    Fetches top_k * 4 candidates to have headroom after audience filtering.
    """
    import json as _json
    candidate_limit = top_k * 4
    rows = conn.execute(
        """
        SELECT c.*, ve.distance
        FROM chunks c
        JOIN (
            SELECT chunk_id, distance
            FROM vec_embeddings
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        ) ve ON c.chunk_id = ve.chunk_id
        WHERE (
            c.do_not_discuss_mode IS NULL
            OR c.do_not_discuss_mode != 'never_in_generated_content'
        )
        ORDER BY ve.distance
        """,
        (_json.dumps(query_embedding), candidate_limit),
    ).fetchall()

    results: list[tuple[SourceChunk, float]] = []
    for row in rows:
        if audience_filter:
            tags = _json.loads(row["audience_tags"] or "[]")
            if audience_filter not in tags:
                continue
        from app.corpus_store import _row_to_chunk
        results.append((_row_to_chunk(row), float(row["distance"])))
        if len(results) >= top_k:
            break
    return results


# ── Step 1: offline relevance curve ──────────────────────────────────────────

def run_step1(embedder: GeminiEmbedder, conn) -> dict:
    print("\n" + "=" * 70)
    print("STEP 1: Offline Relevance Curve")
    print("Embedding test queries and measuring cosine similarity by rank.")
    print("No LLM calls. Cost: Gemini embed only (~$0).")
    print("=" * 70)

    results: dict[str, list[dict]] = {}

    for q in TEST_QUERIES:
        label = q["label"]
        audience = q["audience"]
        print(f"\n── Query: {label}  (audience={audience}) ──────────────────────")
        print(f"   \"{q['text']}\"")

        vec = embedder.embed_query(q["text"])

        # Single query capturing distance alongside chunk metadata
        ranked_raw = query_with_distances(conn, vec, top_k=15, audience_filter=audience)

        ranked: list[dict] = []
        for rank, (chunk, dist) in enumerate(ranked_raw, start=1):
            cosine = l2_to_cosine(dist)
            ranked.append({
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "doc_name": chunk.doc_name[:35],
                "audience_tags": chunk.audience_tags,
                "l2_dist": round(dist, 4),
                "cosine_sim": round(cosine, 4),
            })

        results[label] = ranked

        # Print rank table
        print(f"   {'Rank':>4}  {'CosSim':>7}  {'Chunk ID':35}  {'Doc':35}  Tags")
        for r in ranked:
            sim = f"{r['cosine_sim']:.4f}" if r["cosine_sim"] is not None else "  N/A"
            print(f"   {r['rank']:>4}  {sim:>7}  {r['chunk_id']:35}  {r['doc_name']:35}  {r['audience_tags']}")

        # Compute marginal gain: cosine_sim[k] - cosine_sim[k-1]
        # Elbow = rank where marginal gain first drops below 0.01
        print()
        print("   Marginal cosine gain by rank:")
        prev = ranked[0]["cosine_sim"] if ranked else 0
        for r in ranked[1:]:
            if r["cosine_sim"] is not None and prev is not None:
                gain = r["cosine_sim"] - prev
                bar = "▓" * max(0, int(abs(gain) * 100))
                print(f"     k={r['rank']:>2}: Δ{gain:+.4f}  {bar}")
                prev = r["cosine_sim"]

    # Summarise elbow recommendations across queries
    print("\n── Elbow Summary (where marginal gain first drops below 0.010) ──────")
    elbows: list[int] = []
    for label, ranked in results.items():
        prev = ranked[0]["cosine_sim"] if ranked else 0
        elbow_k = len(ranked)  # default: no elbow found
        for r in ranked[1:]:
            if r["cosine_sim"] is not None and prev is not None:
                gain = abs(r["cosine_sim"] - prev)
                if gain < 0.010:
                    elbow_k = r["rank"] - 1
                    break
                prev = r["cosine_sim"]
        elbows.append(elbow_k)
        print(f"  {label:30s}  elbow @ k={elbow_k}  "
              f"(sim@k={elbow_k}: {ranked[elbow_k-1]['cosine_sim'] if elbow_k <= len(ranked) else 'N/A'})")

    median_elbow = sorted(elbows)[len(elbows) // 2]
    print(f"\n  Median elbow across queries: k={median_elbow}")
    print(f"  Suggested k range to sweep in Step 2: {max(2, median_elbow-2)}–{median_elbow+2}")

    return {"rankings": results, "elbows": dict(zip([q["label"] for q in TEST_QUERIES], elbows)), "median_elbow": median_elbow}


# ── Step 2: downstream citation quality sweep ─────────────────────────────────

async def run_step2(embedder: GeminiEmbedder, conn, anthropic_client: AnthropicClient) -> dict:
    print("\n" + "=" * 70)
    print("STEP 2: Downstream Citation Quality Sweep")
    print(f"Running run_writer(linkedin) for k in {K_CANDIDATES}.")
    print("Measuring: in-context citation rate + substring resolution rate.")
    print("Cost: ~$0.01–0.02 per k value (Sonnet 4.6 Writer call).")
    print("=" * 70)

    brief = SWEEP_BRIEF
    query_text = " ".join(filter(None, [brief.goal, brief.topic_focus, *brief.product_focus]))
    vec = embedder.embed_query(query_text)

    sweep_results: list[dict] = []

    for k in K_CANDIDATES:
        print(f"\n── k={k} ───────────────────────────────────────────────────────")
        ranked_raw = query_with_distances(conn, vec, top_k=k, audience_filter=brief.audience.value)
        chunks = [c for c, _ in ranked_raw]
        print(f"  Retrieved: {len(chunks)} chunks")
        for c, dist in ranked_raw:
            cos = l2_to_cosine(dist)
            print(f"    [{c.chunk_id}]  cosine={cos:.4f}  {c.doc_name[:40]}")

        t0 = time.time()
        draft = await run_writer(
            platform=Platform.LINKEDIN,
            brief=brief,
            corpus_chunks=chunks,
            voice_anchors=[],
            research=None,
            revision_notes=None,
            anthropic_client=anthropic_client,
            trace_id=f"eval_k{k}",
        )
        elapsed = time.time() - t0

        if draft is None:
            print(f"  [FAIL] Writer returned None (parse failure)")
            sweep_results.append({"k": k, "writer_success": False})
            continue

        provided_ids = {c.chunk_id for c in chunks}
        cited_ids = set(draft.cited_chunk_ids)

        in_context = cited_ids & provided_ids
        hallucinated = cited_ids - provided_ids

        # Substring resolution: for each claim with a chunk_id + cited_substring,
        # check if the substring appears in the provided chunk's content
        substring_pass = 0
        substring_fail = 0
        substring_total = 0
        chunk_content_map = {c.chunk_id: c.content for c in chunks}
        for claim in draft.claims:
            if claim.chunk_id and claim.cited_substring:
                substring_total += 1
                content = chunk_content_map.get(claim.chunk_id, "")
                if claim.cited_substring in content:
                    substring_pass += 1
                else:
                    substring_fail += 1

        in_context_rate = len(in_context) / len(cited_ids) if cited_ids else 1.0
        substring_rate = substring_pass / substring_total if substring_total > 0 else None

        print(f"  Claims: {len(draft.claims)}  |  Cited IDs: {len(cited_ids)}")
        print(f"  In-context citations: {len(in_context)}/{len(cited_ids)}  "
              f"({in_context_rate:.0%})  ← chunk IDs that exist in the provided set")
        if hallucinated:
            print(f"  Hallucinated IDs: {hallucinated}  ← cited but NOT in provided chunks")
        if substring_total > 0:
            print(f"  Substring resolution: {substring_pass}/{substring_total}  "
                  f"({substring_rate:.0%})  ← exact text match in source chunk")
        print(f"  Cost: ${draft.cost_usd:.4f}  |  Latency: {elapsed:.1f}s")

        sweep_results.append({
            "k": k,
            "writer_success": True,
            "num_chunks": len(chunks),
            "num_claims": len(draft.claims),
            "cited_ids": len(cited_ids),
            "in_context_cited": len(in_context),
            "hallucinated_ids": len(hallucinated),
            "in_context_rate": round(in_context_rate, 4),
            "substring_checks": substring_total,
            "substring_pass": substring_pass,
            "substring_rate": round(substring_rate, 4) if substring_rate is not None else None,
            "cost_usd": draft.cost_usd,
            "latency_s": round(elapsed, 1),
        })

    # Summary table
    print("\n" + "=" * 70)
    print("STEP 2 SUMMARY: Citation Quality by k")
    print("=" * 70)
    print(f"  {'k':>3}  {'Chunks':>6}  {'In-Ctx%':>8}  {'Substr%':>8}  {'Halluc':>7}  {'Cost':>7}")
    for r in sweep_results:
        if not r["writer_success"]:
            print(f"  {r['k']:>3}  FAIL")
            continue
        substr = f"{r['substring_rate']:.0%}" if r["substring_rate"] is not None else "  N/A"
        print(f"  {r['k']:>3}  {r['num_chunks']:>6}  "
              f"{r['in_context_rate']:>7.0%}  {substr:>8}  "
              f"{r['hallucinated_ids']:>7}  ${r['cost_usd']:.4f}")

    # Recommendation
    best = [r for r in sweep_results if r.get("writer_success") and r.get("in_context_rate", 0) == 1.0]
    if best:
        best_k = min(b["k"] for b in best)
        print(f"\n  Recommended k: {best_k}  (smallest k with 100% in-context citation rate)")
    else:
        best_by_rate = max(
            [r for r in sweep_results if r.get("writer_success")],
            key=lambda r: (r.get("in_context_rate", 0), -r["k"]),
            default=None,
        )
        if best_by_rate:
            print(f"\n  Recommended k: {best_by_rate['k']}  "
                  f"(best in-context rate: {best_by_rate['in_context_rate']:.0%})")

    return {"sweep": sweep_results}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(step1_only: bool) -> None:
    print("E4L Retrieval Quality Evaluation")
    print(f"Corpus: corpus/corpus.db  |  Mode: {'step1 only' if step1_only else 'step1 + step2'}")

    embedder = GeminiEmbedder()
    conn = get_conn()

    step1_results = run_step1(embedder, conn)

    all_results = {"step1": step1_results}

    if not step1_only:
        anthropic_client = AnthropicClient()
        step2_results = await run_step2(embedder, conn, anthropic_client)
        all_results["step2"] = step2_results

    out_path = Path("eval_retrieval_results.json")
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults written to {out_path}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1-only", action="store_true",
                        help="Run only the offline relevance curve (no API cost)")
    args = parser.parse_args()
    asyncio.run(main(args.step1_only))
