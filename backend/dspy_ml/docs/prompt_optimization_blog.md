---
title: "From Drills to Clustered Guidelines: Prompt Optimization for a Catan LLM Agent"
author: "Catan Agent Team"
---

# From Drills to Clustered Guidelines: Prompt Optimization for a Catan LLM Agent

## Goal and Philosophy
The personal goal here is simple: **build the strongest possible Catan agent, fast, with as little human-in-the-loop pain as possible**. I know the game well and want rapid iteration, deterministic feedback, and minimal hand-tuning. Everything that follows—single-step drills, constrained action spaces, clustering, aggressive overfitting with feedback—exists to shorten the loop from “idea” to “measurable lift” without burning human hours.

## Why Drills?
Catan is a combinatorial, stateful game with long temporal dependencies. Optimizing full games is expensive and noisy. We instead optimize on **single-step drills**:
- Each drill isolates one decision (trade, build, dev card, robber move, setup placement, etc.).
- Inputs: game rules (no strategic advice), observation (with higher-level features), viable_actions (restricted/filtered), and an optional guideline.
- Output: one action from the viable set, plus reasoning.

This gives a tight, repeatable objective: did the model choose a correct action among viable options?

## How Drills Are Collected
We curate drills semi-automatically:
- **Human vs LLM comparison UI**: humans write guidelines when the LLM fails; these mark “hard” cases.
- **LLM vs LLM comparison UI**: we let different prompt/program variants face the same drills to find decision boundaries.
- **Mixed sources**: competitive games, synthetic scenarios, and targeted edge cases (robber timing, over-trading, setup).
- Only the **first step** of each multi-step drill is exported to keep it single-step.

Drill coverage includes:
- Setup placement: settlement/road order, distance rule, expansion direction, port value.
- Trading: when to propose/accept/reject; avoid over-trading bottleneck resources; post-rejection behavior.
- Dev cards: Knight to unblock self first, Monopoly timing (only big windfalls), Road Building pathing, VP reveal (only to win).
- Tempo/end-turn discipline: avoid “busywork” roads, low-value trades, or info leakage.
- Resource pivots: switch from wood/brick to ore/wheat/sheep when board matures; when to buy devs for optionality.
- Robber: unblock self first, then hit the leader’s best hex; choose theft target sensibly.
- Bank/ports: convert surplus into scarce resources when player trades are unlikely.

## Baselines and GEPA Attempts
- **Plain distilled (no guidelines)**: ~50–52% on 56 drills.
- **Per-drill guidelines at inference**: ~43/56 (on hard cases) because some drills lacked guidelines; when available they helped a lot.
- **GEPA on distilled (no guidelines)**: modest Pareto lift (~51→56 frontier), but no single prompt dominated.
- **GEPA seeded with guidelines**: unoptimized ~62%; frontier ~87.5% but still no single prompt beating 62% everywhere.

**Takeaway:** The task is heterogeneous; one prompt struggles to cover all modes (setup, midgame trades, dev-card timing, tempo discipline).

## Big Realization: Guidelines Are Gold, but Hard to Compress
Human-written guidelines can push hard drills to ~87.5% accuracy. But compressing them into one universal prompt fails; heterogeneous situations need situational prompts. This led to clustering.

## Leaf Clustering and Overfit Synthesis
We cluster drills (target size 2–4) using embeddings of guideline/observation + action types. For each cluster:
- Iterative synthesis (3 attempts × up to 10 iterations) with **aggressive overfitting** (ALL CAPS allowed).
- Feedback per failed drill: expected vs predicted actions (with payload), notes (e.g., normalized build/setup), viable_actions excerpt.
- Early stop after 3 perfect prompts.
- Scoring normalizes build↔setup when state is absent.

Outputs:
- `guideline_tree.json`: clusters, candidates, scores.
- `best_guidelines_leaf.json`: per-drill best guideline.
- Logs: `guideline_tree_overfit_iter10_v3.log` (leaf run).

### Typical LLM Mistakes at the Leaf Level (with examples)
- **Setup mapping errors**: predicted `build_settlement` vs expected `setup_place_settlement` (payload matches). Normalization fixes type, but off-by-one intersection or edge still fails (e.g., drill 72 predicted intersection 4 vs expected 16; drill 73 edge 29 vs expected 30).
- **Speculative trades instead of a ready build**: proposes trade with surplus ore when a settlement is already placeable; violates “build immediately if you can” (seen in cluster 0 and cluster 3 logs: iter_best 0.25–0.75 until guidelines forbade speculative trades).
- **Dev card misfires**: VP revealed without winning; Monopoly played for tiny gain. Fixed by hard rule: “ONLY play VP if it wins THIS turn; otherwise END TURN” and “play Monopoly only for large, clear windfall; otherwise END TURN.”
- **Tempo leaks**: “busywork” roads or low-value trades when settlement/city is ready; or repeated trades after rejection instead of ending turn. Guidelines now say: build the high-value action immediately; if no productive action, end turn.
- **Resource pivoting**: midgame failure to shift from wood/brick to ore/wheat/sheep; holding ore/wheat without converting. Leaf guidelines emphasize dev-card buys when multiple cards away from settlement.
- **Payload precision**: off-by-one road edge on the path to a settlement; wrong intersection among multiple viable settle spots. Guidelines now include explicit “edge/intersection” reminders in feedback.

## Meta-Clustering
We cluster the leaf clusters (target size ~3). Each drill carries its **leaf-best guideline** into the meta prompt. Same iterative overfit loop with feedback and early stop.

Outputs:
- `guideline_tree_meta.json`
- `best_guidelines_meta.json`
- Logs: `guideline_tree_meta.log`

## What Stayed Hard (Examples from Logs)
- **Setup clusters with zero scores before normalization**: after adding build↔setup normalization, most “0%” clusters recovered; remaining fails were wrong edge/vertex choices.
- **Trade pathfinding**: drills where the model insists on trades despite a legal build in hand.
- **Payload precision**: off-by-one road edges or wrong intersection when multiple viable builds exist.
- **Post-rejection policy**: some drills expect “end_turn” after failed trades; the model sometimes retries a different trade.

## Cluster Results (Leaf) – Best Scores
- Cluster 1 (size 2, setup settlements): best 0.0 before normalization; improves after mapping; remaining payload errors.
- Cluster 0 (size 4, settle vs trade): best ~0.75 with aggressive “build now” + “trade only if 1 resource short” messaging.
- Cluster 3 (size 4, settle/build vs trade/dev): best ~0.75; same pattern of over-trade fixed by strong “build immediately” rules.
- Setup road clusters (e.g., 6, 10, 2): were 0.0 until build↔setup normalization; payload mismatch persists on wrong edges.
- Many other clusters reach 1.0 (dev-card timing, Monopoly, VP, robber, trade rejection discipline).

## Cluster Results (Meta) – Best Scores
Meta-cluster logs (guideline_tree_meta.log) show sizes ~2–4 and best scores often 0.75–1.0, with early stops after 3 perfect prompts in many clusters.

## Before/After Fixes
- **Build↔setup normalization**: setup drills that were 0.0 jumped to correct when types matched; remaining errors are payload-specific.
- **Aggressive overfit prompts**: allowing CAPS/imperatives and forbidding speculative trades lifted settle-vs-trade clusters from ~0.25–0.5 to ~0.75+.
- **Early stop**: once 3 perfect prompts are found for a cluster, we stop iterating—saves tokens and prevents over-churn on solved clusters.

## Concrete Log Excerpts (Leaf)
- Cluster with setup roads (before normalization): predicted `build_road` payload correct, expected `setup_place_road`; scored 0% until normalization.
- Cluster with VP misuse: fixed by adding hard rule “ONLY play VP if it wins THIS turn; otherwise END TURN”.
- Cluster with Monopoly misuse: guideline “play only on large windfall; otherwise END TURN” yielded 100%.
- Cluster with settle vs trade: guidelines now say “IF you can settle now, DO IT—do NOT trade or road first. ONLY trade if exactly one resource short and it completes the build this turn.” Best scores rose to ~0.75 in logs.

## Concrete Log Excerpts (Meta)
- Meta clusters combined leaf guidelines; scores often ≥0.75, some 1.0. Early stop triggered when 3 perfect prompts appeared for a cluster.

## Code Highlights

### Overfit Feedback (per drill)
```python
details.append({
    "drill_id": ex["drill_id"],
    "expected": ex.get("expected_action"),
    "predicted": predicted,
    "matched": matched,
    "note": note or ("type_mismatch" if not matched else ""),
    "viable_excerpt": (ex.get("viable_actions") or "")[:200],
})
```

### Build↔Setup Normalization in Scoring
```python
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
    matched = True
```

### Iterative Overfit with Early Stop
```python
success_prompts = 0
for _ in range(iterations):
    variants = synthesize_guidelines_with_feedback(..., feedback=feedback)
    scored = [...]
    top_score, top_guideline, top_success, top_details = scored[0]
    best_candidates.append((top_score, top_guideline))
    feedback = build_feedback(clist, top_details, top_guideline)
    if top_score >= 1.0:
        success_prompts += 1
    if success_prompts >= 3:
        break
```

## Evaluation Summary
- **Plain distilled (no guidelines)**: ~50–52% on 56 drills.
- **Per-drill guidelines (leaf)**: ~87.5% on hard cases.
- **GEPA alone**: modest frontier improvements; no universal winner.
- **GEPA + guideline seed**: frontier up to ~87.5% but still heterogeneous.
- **Leaf + Meta clustering**: preserved ~87.5% frontier without per-drill runtime guidance; cluster/meta prompts carry situational wisdom.

## Design Patterns That Helped
- **Single-step drills** as a proxy objective: cheap, focused, repeatable.
- **Restricted viable_actions**: forces the model to choose among plausible options; surfaces real decision quality.
- **Aggressive overfit prompts**: allow CAPS/imperatives to push the model off bad local minima.
- **Feedback with expected vs predicted**: critical to steer rewriting; note payload and type mismatches.
- **Early stop after perfects**: saves tokens and avoids over-churn on already-solved clusters.
- **Type normalization**: build vs setup mapping; reduces false negatives.
- **Action-space discipline**: hard reminders to avoid speculative trades/devs unless they complete an immediate build/win.

## Making It Pain-Free and Fast
- **Single-step drills**: cheap, focused evaluation; no long rollouts.
- **Restricted viable actions**: fewer degrees of freedom; easier, deterministic scoring.
- **Aggressive overfit prompts**: allow CAPS/imperatives to jolt the model out of bad local minima without manual tinkering.
- **Iterative synthesis + early stop**: stop after 3 perfect prompts per cluster; saves tokens and time.
- **Normalization fixes**: build↔setup mapping to avoid false negatives on setup drills.
- **Reusable scripts and logs**: one-click runs (`cluster_guideline_tree.py`, `cluster_guideline_tree_meta.py`) with detailed logs for postmortem.
- **Clustering**: carry situational power without per-instance guidelines at inference; fewer prompts to manage.

## Drill Snapshots (Concrete Examples)
- **Setup settlement (drill 72)**  
  - Expected: `setup_place_settlement` @ intersection 16  
  - Pred (before fixes): `build_settlement` @ intersection 4 → WRONG (type + payload)  
  - After normalization + guideline: type fixed; payload still off; final guideline: “PLACE SETTLEMENT NOW on best intersection; do NOT trade/road first.”
- **Setup road (drill 73)**  
  - Expected: `setup_place_road` edge 30  
  - Pred: `build_road` edge 29 → WRONG; normalization fixes type; guideline adds explicit edge targeting.
- **Tempo/end-turn vs trade**  
  - Expected: `end_turn`  
  - Pred: `propose_trade` offering surplus ore → WRONG; guideline: “IF no immediate build, END TURN; do NOT propose speculative trades.”
- **Dev card misuse (VP/Monopoly)**  
  - Expected: `end_turn` (hold VP/Monopoly)  
  - Pred: `play_dev_card` VP/Monopoly for tiny gain → WRONG; guideline: “ONLY play VP if it wins THIS TURN; Monopoly only for LARGE windfall; otherwise END TURN.”

## Failure Case Study (Trade vs Settle)
- Pattern: model trades when it can already settle or build road-to-settle this turn.  
- Fix: “IF you can settle now, DO IT. ONLY trade if exactly one resource short AND it completes the build this turn. DO NOT ‘set up’ roads first.”  
- Result: clusters 0/3 lifted from ~0.25–0.5 to ~0.75 in best scores.

## Before/After (Selected Cases)
| Case | Before | After |
| --- | --- | --- |
| Setup type mismatch | `build_settlement` vs `setup_place_settlement` (0%) | Normalized type; payload fixed via explicit edge/vertex in guideline |
| Speculative trade | Trade instead of settle-now (0.25–0.5) | “IF can settle, DO IT; ONLY trade if exactly 1 short” (0.75+) |
| VP misuse | Reveal VP without win | “ONLY play VP if wins THIS turn; else END TURN” (1.0) |
| Monopoly misuse | Play for small gain | “ONLY on large windfall; else END TURN” (1.0) |

## Leaf vs Meta (Score Glimpse)
| Level | Example clusters | Best scores |
| --- | --- | --- |
| Leaf | dev-card timing, VP/Monopoly | 1.0 |
| Leaf | settle vs trade clusters | ~0.75 after aggressive prompts |
| Leaf | setup roads/settles | From 0.0 → fixed type; remaining payload issues |
| Meta | grouped leaf clusters | Often 0.75–1.0; early-stop after 3 perfects |

## Token/Time Savings
- Early stop after 3 perfect prompts per cluster avoids over-iteration once solved.
- Aggressive prompts reduce manual retuning; “iter best” lines in logs stabilize early.
- Normalization reduces false negatives without manual adjudication.

## Workflow Timeline (Fast Loop)
1) Export drills (single-step, restricted actions, higher-level features).  
2) Baseline eval (plain distilled).  
3) GEPA attempts (with/without guideline seed).  
4) Leaf clustering + overfit + early stop → `guideline_tree.json`, `best_guidelines_leaf.json`.  
5) Meta clustering + overfit + early stop → `guideline_tree_meta.json`, `best_guidelines_meta.json`.  
6) Eval with per-cluster/meta guidelines; iterate on scoring fixes (build↔setup, payload feedback).

## Log Pointers (for readers)
- Leaf log: `dspy_ml/data/guideline_tree_overfit_iter10_v3.log`  
  - Grep `iter best` for per-iteration best scores/fail counts.  
  - Grep `score=` for final top candidates; payload/type notes included.  
- Meta log: `dspy_ml/data/guideline_tree_meta.log`  
  - Same patterns; shows meta cluster sizes and early-stop events.

## Aggressive Prompt Template (Excerpt)
```
OVERFIT to these drills: your goal is to push the LLM to pick the correct action from viable_actions for THESE drills, even if the tone is aggressive or the advice is very specific.
You may use CAPS, imperative voice, explicit DO/DO NOT. You may forbid wrong patterns you infer.
Write 2-4 sentences, WHEN/IF [situation], then [action/principle]. Be specific and actionable.
You may include a short ALL-CAPS motto line if it helps force the choice.
```

## Human-in-the-Loop Minimization (Quantified)
- Single-step drills: no long rollouts.  
- Restricted viable actions: small, deterministic choice set.  
- Automated feedback: expected vs predicted with payload notes; no manual labeling loop.  
- Early stop: halts once solved.  
- Clustering: reuse per-situation guidance; fewer prompts to manage; no per-instance guidelines at inference.  
- Normalization: prevents false negatives (build vs setup) without manual adjudication.

## Remaining Gaps and Next Steps
- Pass GameState into scoring everywhere to reduce payload mismatches on setup drills.
- Add retrieval: select cluster/meta guideline via embedding similarity at inference.
- Consider lightweight fine-tuning using cluster/meta guidelines as supervision to reduce prompt switching.
- Expand drill set for nuanced trade acceptance/rejection and robber targeting.
- Explore smaller meta clusters for dev-card timing vs trading vs tempo.

## How to Reproduce
1. Export drills (already in `dspy_ml/data/drills_dataset.json`).
2. Run leaf overfit: `cluster_guideline_tree.py` → `guideline_tree.json`, `best_guidelines_leaf.json`.
3. Run meta overfit: `cluster_guideline_tree_meta.py` with `--leaf-tree guideline_tree.json` → `guideline_tree_meta.json`, `best_guidelines_meta.json`.
4. Evaluate by injecting the per-drill (or per-cluster/meta) guideline into the agent’s guideline field and scoring on the 56 drills.

## Why This Matters
- Shows that **instance-specific hints** can be compressed into a **small family of situational prompts** without losing much accuracy.
- Demonstrates limits of single universal prompts on heterogeneous task sets.
- Provides an open-source, drill-centric workflow for complex game agents, with DSPy/GEPA integration, clustering, and iterative LLM-driven synthesis.

## Pointers to the Repo
- Scripts: `dspy_ml/scripts/cluster_guideline_tree.py` (leaf), `cluster_guideline_tree_meta.py` (meta), `optimize_distilled.py`, `test_distilled.py`, `compare_distilled_variants.py`.
- Data/logs: `dspy_ml/data/` (guideline trees, best-guideline mappings, GEPA logs).
- Agent signatures: `dspy_ml/signature.py`, `dspy_ml/signature_distilled.py`.
- Scoring helpers: `api/routes.py` (canonical action matching).

---

This project illustrates a practical path: **start with focused drills, use human-written guidelines to identify hard cases, cluster to reuse and compress guidance, and iteratively overfit with explicit feedback.** Even without a single universal prompt, clustered/meta guidelines preserved near-best accuracy and offer a blueprint for other complex, heterogeneous LLM decision domains.


