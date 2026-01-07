#!/usr/bin/env python3
"""
Cluster drills into small groups (size ~2-3), synthesize cluster guidelines,
score them (with parallelism), and build a guideline tree.

Outputs:
- guideline_tree.json: clusters with candidate guidelines and scores
- best_guidelines_leaf.json: per-drill best guideline from leaf clusters
- guideline_tree.log: detailed per-cluster logs (scores, variants)
"""
import sys
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
import litellm
from dspy_ml.signature import CatanDrillSignature
from api.routes import _canonical_action_dict


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    embeddings = []
    for i, t in enumerate(texts, 1):
        resp = litellm.embedding(model=model, input=t)
        vec = resp["data"][0]["embedding"]
        embeddings.append(vec)
    return np.array(embeddings)


def build_features(drills: List[Dict[str, Any]]) -> np.ndarray:
    # Use only observation text (available at inference) for clustering.
    texts = []
    for d in drills:
        obs = d.get("observation") or ""
        texts.append(obs)
    return embed_texts(texts)


def cluster_drills(X: np.ndarray, target_cluster_size: int = 3) -> np.ndarray:
    k = max(1, math.ceil(len(X) / target_cluster_size))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels


def split_large_clusters(clusters: Dict[int, List[Dict[str, Any]]], max_size: int = 4) -> Dict[int, List[Dict[str, Any]]]:
    """Force clusters to size <= max_size by KMeans splitting within oversized clusters."""
    new_clusters = {}
    next_id = 0
    for cid, clist in clusters.items():
        if len(clist) <= max_size:
            new_clusters[next_id] = clist
            next_id += 1
        else:
            # Re-cluster this cluster into chunks of size <= max_size
            sub_k = max(1, math.ceil(len(clist) / max_size))
            X_sub = build_features(clist)
            km = KMeans(n_clusters=sub_k, random_state=42, n_init=10)
            labels = km.fit_predict(X_sub)
            for sub in range(sub_k):
                sub_drills = [d for d, lbl in zip(clist, labels) if lbl == sub]
                new_clusters[next_id] = sub_drills
                next_id += 1
    return new_clusters


def synthesize_guidelines(cluster_drills: List[Dict[str, Any]], num_variants: int, model: str) -> List[str]:
    return synthesize_guidelines_with_feedback(cluster_drills, num_variants, model, feedback=None)


def synthesize_guidelines_with_feedback(cluster_drills: List[Dict[str, Any]], num_variants: int, model: str, feedback: str = None) -> List[str]:
    prompt_base = []
    prompt_base.append("You are creating a strategic guideline for a small cluster of Catan drills.")
    prompt_base.append("Each drill has a desired action. Some drills may have human-written guidelines; others do not.")
    prompt_base.append("OVERFIT to these drills: your goal is to push the LLM to pick the correct action from viable_actions for THESE drills, even if the tone is aggressive or the advice is very specific.")
    prompt_base.append("You may use CAPS, imperative voice, and explicit DO/DO NOT instructions. You may explicitly forbid wrong patterns you infer.")
    prompt_base.append("Write 2-4 sentences, start with WHEN/IF [situation], then [action/principle]. Be specific and actionable.")
    prompt_base.append("Do NOT hardcode edge/intersection/tile IDs or fixed numeric payloads; instead, reference viable_actions and choose by rule (e.g., shortest path to settlement, highest-value build, avoid speculative trades/devs).")
    prompt_base.append("You may include a short ALL-CAPS motto line if it helps force the choice (e.g., NEVER TRADE HERE—BUILD THE ROAD NOW).")
    if feedback:
        prompt_base.append("\nFeedback from previous attempt (failures to fix):")
        prompt_base.append(feedback)
    prompt_base.append("\nCluster drills:")
    for d in cluster_drills:
        gid = d["drill_id"]
        act = d.get("expected_action", {}).get("type", "unknown")
        gtxt = d.get("guideline", "")
        prompt_base.append(f"- Drill {gid}: expected_action={act}; guideline={'NONE' if not gtxt else gtxt[:200]}")
    base = "\n".join(prompt_base)

    variants = []
    for _ in range(num_variants):
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": base}],
            reasoning_effort="high"
        )
        variants.append(resp.choices[0].message.content.strip())
    return variants


def robust_parse(chosen_action_str: str) -> Any:
    if not chosen_action_str or str(chosen_action_str).lower() == "null":
        return None
    try:
        return json.loads(chosen_action_str)
    except Exception:
        try:
            cleaned = chosen_action_str.rstrip("}").rstrip() + "}"
            return json.loads(cleaned)
        except Exception:
            import re
            m = re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", chosen_action_str)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None


def score_guideline_on_cluster(guideline: str, cluster_drills: List[Dict[str, Any]], model: str) -> Tuple[float, Dict[int, bool], List[Dict[str, Any]]]:
    lm = dspy.LM(model=model)
    dspy.configure(lm=lm)
    module = dspy.ChainOfThought(CatanDrillSignature)

    correct = 0
    total = len(cluster_drills)
    successes: Dict[int, bool] = {}
    details: List[Dict[str, Any]] = {}
    details = []
    for ex in cluster_drills:
        result = module(
            game_rules=ex["game_rules"],
            observation=ex["observation"],
            viable_actions=ex["viable_actions"],
            guideline=guideline
        )
        predicted = robust_parse(getattr(result, "chosen_action", "null") or "null")
        if not predicted:
            successes[ex["drill_id"]] = False
            details.append({
                "drill_id": ex["drill_id"],
                "expected": ex.get("expected_action"),
                "predicted": None,
                "matched": False,
                "note": "no prediction",
                "viable_excerpt": (ex.get("viable_actions") or "")[:200],
            })
            continue
        pred_canon = _canonical_action_dict(predicted, state=None)
        matched = False
        note = ""
        for ca in ex["correct_actions"]:
            ca_canon = _canonical_action_dict(ca, state=None)
            # Phase-agnostic mapping for setup/build equivalence when state is None
            def normalize_setup_build(canon):
                if not isinstance(canon, dict):
                    return canon
                t = canon.get("type")
                if t == "build_road":
                    canon = canon.copy(); canon["type"] = "setup_place_road"
                elif t == "build_settlement":
                    canon = canon.copy(); canon["type"] = "setup_place_settlement"
                return canon
            if normalize_setup_build(ca_canon) == normalize_setup_build(pred_canon):
                correct += 1
                matched = True
                if ca_canon.get("type") != pred_canon.get("type"):
                    note = "normalized build/setup"
                break
        successes[ex["drill_id"]] = matched
        details.append({
            "drill_id": ex["drill_id"],
            "expected": ex.get("expected_action"),
            "predicted": predicted,
            "matched": matched,
            "note": note or ("type_mismatch" if not matched else ""),
            "viable_excerpt": (ex.get("viable_actions") or "")[:200],
        })
    return (correct / total if total else 0.0), successes, details


def build_feedback(cluster_drills: List[Dict[str, Any]], details: List[Dict[str, Any]], guideline: str) -> str:
    failed = [det for det in details if not det.get("matched")]
    if not failed:
        return "All drills succeeded; keep concise and consistent. You may keep strong wording."
    lines = ["Some drills FAILED with the previous guideline. Rewrite to fix these; use strong, explicit, imperative language; include ALL-CAPS warnings if helpful."]
    for det in failed:
        lines.append(f"- Drill {det['drill_id']} expected={det.get('expected')} predicted={det.get('predicted')} note={det.get('note','')}")
    lines.append("\nRewrite ONE guideline (2-4 sentences, WHEN/IF ... then ...) that forces the correct action. Be aggressive if needed. If payload differs, be explicit about the edge/intersection.")
    return "\n".join(lines)


def process_cluster_iterative(args_tuple):
    cid, clist, model, iterations, attempts = args_tuple
    best_candidates = []
    feedback = None
    iter_log = []
    success_prompts = 0
    for _ in range(iterations):
        variants = synthesize_guidelines_with_feedback(clist, num_variants=attempts, model=model, feedback=feedback)
        scored = []
        for v in variants:
            score, successes, details = score_guideline_on_cluster(v, clist, model=model)
            scored.append((score, v, successes, details))
        scored.sort(reverse=True, key=lambda x: x[0])
        top_score, top_guideline, top_success, top_details = scored[0]
        best_candidates.append((top_score, top_guideline))
        fail_count = sum(1 for ok in top_success.values() if not ok)
        iter_log.append({"iter_best_score": top_score, "iter_best_guideline": top_guideline[:200], "fail_count": fail_count})
        feedback = build_feedback(clist, top_details, top_guideline)
        if top_score >= 1.0:
            success_prompts += 1
        if success_prompts >= 3:
            break
    best_candidates.sort(reverse=True, key=lambda x: x[0])
    top = best_candidates[:3]
    return cid, clist, top, iter_log


def main():
    import argparse
    p = argparse.ArgumentParser(description="Cluster drills and synthesize hierarchical guidelines (leaf level)")
    p.add_argument("--dataset", required=True, help="Path to drills_dataset.json")
    p.add_argument("--model", default="gpt-5.2", help="LLM for synthesis and scoring")
    p.add_argument("--variants", type=int, default=10, help="(unused now)")
    p.add_argument("--target-cluster-size", type=int, default=3, help="Target size for leaf clusters")
    p.add_argument("--top-k", type=int, default=3, help="Keep top-k variants per cluster")
    p.add_argument("--output-tree", default="backend/dspy_ml/data/guideline_tree.json", help="Path to save tree JSON")
    p.add_argument("--output-best", default="backend/dspy_ml/data/best_guidelines_leaf.json", help="Path to save best per drill")
    p.add_argument("--log", default="backend/dspy_ml/data/guideline_tree.log", help="Path to detailed log")
    p.add_argument("--iterations", type=int, default=3, help="Iterations per cluster")
    p.add_argument("--attempts-per-iter", type=int, default=3, help="Guideline candidates per iteration")
    args = p.parse_args()

    drills = load_dataset(Path(args.dataset))
    drills = [d for d in drills if d.get("game_rules") and d.get("observation") and d.get("viable_actions")]

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        lf.write("== Guideline Tree Run ==\n")
        lf.write(f"Dataset: {args.dataset}\nModel: {args.model}\n")
        lf.write(f"Iterations: {args.iterations} attempts_per_iter: {args.attempts_per_iter}\n")
        lf.write(f"Target cluster size: {args.target_cluster_size}\nTop-k: {args.top_k}\n")
        lf.write(f"Total drills: {len(drills)}\n")
        lf.flush()

    with open(log_path, "a") as lf:
        lf.write("Embedding texts...\n")
        lf.flush()
    X = build_features(drills)
    with open(log_path, "a") as lf:
        lf.write(f"Embedding complete. Shape={X.shape}\n")
        lf.flush()

    labels = cluster_drills(X, target_cluster_size=args.target_cluster_size)

    clusters = defaultdict(list)
    for d, lbl in zip(drills, labels):
        clusters[int(lbl)].append(d)

    # Enforce max cluster size 4
    clusters = split_large_clusters(clusters, max_size=4)

    with open(log_path, "a") as lf:
        lf.write(f"Clusters: {len(clusters)}\n")
        for cid, clist in clusters.items():
            lf.write(f"  Cluster {cid}: size={len(clist)} drills={[d['drill_id'] for d in clist]}\n")
        lf.flush()

    tasks = []
    for cid, clist in clusters.items():
        tasks.append((cid, clist, args.model, args.iterations, args.attempts_per_iter))

    cluster_results = {}
    best_per_drill = {}

    with ProcessPoolExecutor(max_workers=min(8, len(tasks))) as ex:
        futs = {ex.submit(process_cluster_iterative, t): t[0] for t in tasks}
        for f in as_completed(futs):
            cid = futs[f]
            cid_ret, clist, top, iter_log = f.result()
            cluster_results[cid_ret] = {
                "cluster_id": cid_ret,
                "drill_ids": [d["drill_id"] for d in clist],
                "candidates": [{"score": s, "guideline": g} for s, g in top],
                "size": len(clist),
            }
            # Assign best
            for d in clist:
                best_per_drill[d["drill_id"]] = top[0][1]
            with open(log_path, "a") as lf:
                lf.write(f"\n[Cluster {cid_ret}] size={len(clist)} drills={cluster_results[cid_ret]['drill_ids']}\n")
                for ilog in iter_log:
                    lf.write(f"  iter best={ilog['iter_best_score']:.3f} fail_count={ilog['fail_count']} | {ilog['iter_best_guideline'].replace(chr(10),' ')}\n")
                for s, g in top:
                    lf.write(f"  score={s:.3f} | {g[:200].replace(chr(10),' ')}...\n")
                lf.flush()

    Path(args.output_tree).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_tree, "w") as f:
        json.dump({"clusters": list(cluster_results.values())}, f, indent=2)

    Path(args.output_best).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_best, "w") as f:
        json.dump(best_per_drill, f, indent=2)

    with open(log_path, "a") as lf:
        lf.write("\n== Summary ==\n")
        lf.write(f"Saved tree: {args.output_tree}\n")
        lf.write(f"Saved best: {args.output_best}\n")
        lf.write(f"Clusters: {len(cluster_results)}\n")
        for cid, res in cluster_results.items():
            lf.write(f"  Cluster {cid}: size={res['size']} best_score={res['candidates'][0]['score']:.3f}\n")

    print(f"Saved tree to {args.output_tree}")
    print(f"Saved best-per-drill guidelines to {args.output_best}")
    print(f"Logs at {log_path}")


if __name__ == "__main__":
    sys.exit(main())

