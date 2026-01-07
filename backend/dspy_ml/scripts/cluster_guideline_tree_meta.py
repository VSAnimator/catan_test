#!/usr/bin/env python3
"""
Meta-level clustering: cluster leaf clusters (from guideline_tree.json) into super-clusters,
then synthesize overfitted meta-guidelines with iterative feedback, scoring on all drills
in each super-cluster.
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
    for t in texts:
        resp = litellm.embedding(model=model, input=t)
        vec = resp["data"][0]["embedding"]
        embeddings.append(vec)
    return np.array(embeddings)


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
    details: List[Dict[str, Any]] = []

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


def synthesize_guidelines_with_feedback(cluster_drills: List[Dict[str, Any]], num_variants: int, model: str, feedback: str = None) -> List[str]:
    prompt_base = []
    prompt_base.append("You are creating an overfitted strategic guideline for a meta-cluster of Catan drills.")
    prompt_base.append("OVERFIT: force the LLM to pick the correct action from viable_actions for THESE drills, even with aggressive wording.")
    prompt_base.append("You may use CAPS, imperative voice, explicit DO/DO NOT, and forbid wrong patterns. 2-4 sentences, WHEN/IF ... then ...")
    prompt_base.append("Do NOT hardcode edge/intersection/tile IDs or fixed numeric payloads; instead, reference viable_actions and choose by rule (shortest path to settle, highest-value build, avoid speculative trades/devs).")
    if feedback:
        prompt_base.append("\nFeedback from previous attempt (failures to fix):")
        prompt_base.append(feedback)
    prompt_base.append("\nMeta-cluster drills:")
    for d in cluster_drills:
        gid = d["drill_id"]
        act = d.get("expected_action", {}).get("type", "unknown")
        gtxt = d.get("guideline", "")
        prompt_base.append(f"- Drill {gid}: expected_action={act}; leaf_guideline={'NONE' if not gtxt else gtxt[:200]}")
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


def cluster_meta(nodes: List[Dict[str, Any]], drills_by_id: Dict[int, Dict[str, Any]], target_size: int = 3) -> Dict[int, List[Dict[str, Any]]]:
    """
    Cluster leaf clusters using ONLY observation text (no guideline or drill name),
    by averaging embeddings of member drill observations.
    """
    cluster_embeddings = []
    for n in nodes:
        obs_texts = []
        for did in n["drill_ids"]:
            obs = drills_by_id[did]["observation"]
            obs_texts.append(obs)
        if not obs_texts:
            emb = np.zeros((1536,))
        else:
            emb_matrix = embed_texts(obs_texts)
            emb = emb_matrix.mean(axis=0)
        cluster_embeddings.append(emb)

    X = np.vstack(cluster_embeddings)
    k = max(1, math.ceil(len(nodes) / target_size))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    clusters = defaultdict(list)
    for node, lbl in zip(nodes, labels):
        clusters[int(lbl)].append(node)
    return clusters


def main():
    import argparse
    p = argparse.ArgumentParser(description="Meta-level clustering of leaf clusters with overfitted guideline synthesis")
    p.add_argument("--dataset", required=True, help="Path to drills_dataset.json")
    p.add_argument("--leaf-tree", required=True, help="Path to leaf guideline_tree.json")
    p.add_argument("--model", default="gpt-5.2", help="LLM for synthesis and scoring")
    p.add_argument("--iterations", type=int, default=10, help="Iterations per meta-cluster")
    p.add_argument("--attempts-per-iter", type=int, default=3, help="Guideline candidates per iteration")
    p.add_argument("--target-cluster-size", type=int, default=3, help="Target size for meta clusters")
    p.add_argument("--top-k", type=int, default=3, help="Keep top-k variants per meta-cluster")
    p.add_argument("--output-tree", default="backend/dspy_ml/data/guideline_tree_meta.json", help="Path to save meta tree JSON")
    p.add_argument("--output-best", default="backend/dspy_ml/data/best_guidelines_meta.json", help="Path to save best per drill (meta)")
    p.add_argument("--log", default="backend/dspy_ml/data/guideline_tree_meta.log", help="Path to detailed log")
    args = p.parse_args()

    drills = load_dataset(Path(args.dataset))
    drills_by_id = {d["drill_id"]: d for d in drills}

    leaf_tree = json.load(open(args.leaf_tree))
    leaf_nodes = leaf_tree.get("clusters", [])

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        lf.write("== Meta Guideline Tree Run ==\n")
        lf.write(f"Dataset: {args.dataset}\nLeaf tree: {args.leaf_tree}\nModel: {args.model}\n")
        lf.write(f"Iterations: {args.iterations} attempts_per_iter: {args.attempts_per_iter}\n")
        lf.write(f"Target meta cluster size: {args.target_cluster_size}\nTop-k: {args.top_k}\n")
        lf.write(f"Leaf clusters: {len(leaf_nodes)}\n")
        lf.flush()

    meta_clusters = cluster_meta(leaf_nodes, drills_by_id, target_size=args.target_cluster_size)

    with open(log_path, "a") as lf:
        lf.write(f"Meta clusters: {len(meta_clusters)}\n")
        for cid, cnodes in meta_clusters.items():
            lf.write(f"  Meta {cid}: leaf_ids={[n['cluster_id'] for n in cnodes]} drills={sum(len(n['drill_ids']) for n in cnodes)}\n")
        lf.flush()

    # Build meta tasks with drills expanded
    tasks = []
    for cid, cnodes in meta_clusters.items():
        drill_ids = []
        for n in cnodes:
            drill_ids.extend(n["drill_ids"])
        clist = []
        for did in drill_ids:
            d = drills_by_id[did]
            # attach leaf best guideline if available
            # pick first candidate guideline from leaf node containing this drill
            leaf_guideline = ""
            for n in cnodes:
                if did in n["drill_ids"] and n.get("candidates"):
                    leaf_guideline = n["candidates"][0].get("guideline", "")
                    break
            d = d.copy()
            d["guideline"] = leaf_guideline
            clist.append(d)
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
            for d in clist:
                best_per_drill[d["drill_id"]] = top[0][1]
            with open(log_path, "a") as lf:
                lf.write(f"\n[Meta {cid_ret}] size={len(clist)} drills={cluster_results[cid_ret]['drill_ids']}\n")
                for ilog in iter_log:
                    lf.write(f"  iter best={ilog['iter_best_score']:.3f} fail_count={ilog['fail_count']} | {ilog['iter_best_guideline'].replace(chr(10),' ')}\n")
                for s, g in top:
                    lf.write(f"  score={s:.3f} | {g[:200].replace(chr(10),' ')}...\n")
                lf.flush()

    Path(args.output_tree).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_tree, "w") as f:
        json.dump({"meta_clusters": list(cluster_results.values())}, f, indent=2)

    Path(args.output_best).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_best, "w") as f:
        json.dump(best_per_drill, f, indent=2)

    with open(log_path, "a") as lf:
        lf.write("\n== Summary ==\n")
        lf.write(f"Saved meta tree: {args.output_tree}\n")
        lf.write(f"Saved best: {args.output_best}\n")
        lf.write(f"Meta clusters: {len(cluster_results)}\n")
        for cid, res in cluster_results.items():
            lf.write(f"  Meta {cid}: size={res['size']} best_score={res['candidates'][0]['score']:.3f}\n")

    print(f"Saved meta tree to {args.output_tree}")
    print(f"Saved best-per-drill guidelines to {args.output_best}")
    print(f"Logs at {log_path}")


if __name__ == "__main__":
    sys.exit(main())

