# 🧠 Skill Collapse in Hierarchical Reinforcement Learning

> *Does learning more skills make agents worse at old ones? This project formally measures and mitigates catastrophic forgetting in skill-based RL agents.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🔥 The Problem Nobody Has Studied

Modern Hierarchical RL agents (like SkillRL, 2026) build a **skill library** over time — reusable behaviors chained together to solve complex tasks. But a critical question has been ignored:

**What happens to Skill #1 when the agent learns Skill #50?**

In standard neural networks, *catastrophic forgetting* is well-documented (Kirkpatrick et al., 2017). Skill-based RL makes this worse:

- Skills share underlying policy parameters — they are **not isolated**
- A new skill occupying similar state-action space **overwrites** representations used by old skills
- The skill library grows, but **early skill quality silently degrades**
- There is **no training signal** warning the agent that old skills are failing

This project calls that phenomenon **Skill Collapse** — the first formal measurement and mitigation study of this effect.

---

## 💡 Research Hypothesis

> *"In HRL agents with a growing skill library, the performance of previously learned skills degrades as a function of (a) total skill library size, (b) pairwise skill similarity, and (c) available network capacity."*

**Three predicted regimes:**

| Regime | Skill Count | Behavior |
|--------|-------------|----------|
| **Stable** | 1–10 | No degradation, skills may reinforce |
| **Compression** | 10–30 | Marginal degradation, recoverable |
| **Collapse** | 30+ | Significant degradation, non-recoverable without replay |

---

## 📁 Project Structure

```
skill-collapse-rl/
├── main.py                          # Run all experiments from one entry point
├── requirements.txt
│
├── envs/
│   └── grid_world.py                # Custom 8×8 goal-conditioned GridWorld (Gymnasium)
│
├── agents/
│   └── skill_agent.py               # PPO agent — SharedPolicyNet + IsolatedHeadPolicyNet
│
├── skills/
│   └── skill_library.py             # Skill generation, embedding, similarity matrix
│
├── experiments/
│   ├── exp1_retention.py            # Retention matrix measurement
│   ├── exp2_similarity.py           # Skill similarity vs. collapse rate
│   └── exp3_mitigation.py           # Compare: Baseline / Replay / EWC / Isolated Heads
│
├── visualizations/
│   └── retention_heatmap.py         # Heatmap + age curve + mitigation bar chart
│
└── results/                         # Auto-created, JSON outputs saved here
    ├── exp1/
    ├── exp2/
    └── exp3/
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/yourusername/skill-collapse-rl
cd skill-collapse-rl
pip install -r requirements.txt

# Quick run — all 3 experiments (~5–10 mins on CPU)
python main.py

# Full run — more skills, more training (~30–60 mins)
python main.py --full

# Run a specific experiment
python main.py --exp 1
python main.py --exp 1 --n_skills 20

# Visualize results (requires matplotlib)
python visualizations/retention_heatmap.py \
    --results results/exp1/results.json \
    --mitigation_results results/exp3/results.json
```

---

## 📊 Input & Output — Simple Version

**Input:**
An RL agent that learns skills one by one, like a student studying subjects in sequence.

**Output:**
A measurement of how much the agent *forgets* old skills as it learns new ones — plus which fix works best.

---

**Concrete example:**

```
Train on Skill 1 (go to top-left corner)     → agent succeeds 90%
Train on Skill 2 (collect item at center)    → agent succeeds 85%
Train on Skill 3 (avoid hazard + reach goal) → agent succeeds 80%

Now re-test Skill 1...
→ agent only succeeds 40%  ← THIS is Skill Collapse
```

The project measures *how fast* that 90% → 40% drop happens, *why* it happens (similar skills = faster drop), and *how to stop it*.

---

## 🔬 Does This Already Exist?

**Catastrophic forgetting** in neural networks — ✅ heavily studied since 2017.

**Continual learning in RL** (single task sequences) — ✅ studied.

**Skill Collapse specifically in growing skill libraries** — ❌ not formally studied.

The papers that come closest:

| Paper | What it does | What it misses |
|-------|-------------|----------------|
| EWC (2017) | Measures forgetting in supervised learning | Not skill-based RL |
| DIAYN (2018) | Discovers skills | Never checks if old ones degrade |
| SkillRL (2026) | Builds skill libraries | No retention measurement at all |

---

**So the gap is real** — nobody has put a number on "how much does skill quality degrade as library size grows" in HRL. That's exactly what this project measures. It's a small but genuine original contribution, which is all a portfolio project needs to be.

---

## 🧪 Experiments

### Experiment 1 — Baseline Skill Retention

Train an agent **sequentially** on N skills. After each new skill is learned, evaluate **all previous skills**. Build a retention matrix R[i][j]:
- `i` = how many skills have been learned so far
- `j` = which skill is being evaluated

```bash
python experiments/exp1_retention.py --n_skills 15 --updates_per_skill 30
```

**Output:** `results/exp1/results.json` containing the full retention matrix, per-age retention statistics, and collapse detection.

---

### Experiment 2 — Skill Similarity vs. Collapse Rate

**Hypothesis:** Skills that are more *similar* to each other cause greater mutual degradation.

- Computes cosine similarity between skills using their state-distribution embeddings
- Splits pairs into high-similarity / low-similarity groups
- Measures collapse rate per group and computes Pearson correlation

```bash
python experiments/exp2_similarity.py --n_skills 20
```

**Output:** `results/exp2/results.json` with per-pair similarity/retention data and correlation coefficient.

---

### Experiment 3 — Mitigation Strategies

Compares 4 approaches on identical skill sets:

| Strategy | How It Works | Expected Tradeoff |
|----------|-------------|-------------------|
| **Baseline** | Shared network, no mitigation | — |
| **Skill Replay** | Re-train on old skill episodes after each new skill | Slower training |
| **EWC** | Penalize changes to weights critical for past skills | Partial reduction, no memory cost |
| **Isolated Heads** | Separate policy/value head per skill, shared encoder | Eliminates collapse, linear memory growth |

```bash
python experiments/exp3_mitigation.py --n_skills 15
```

**Output:** Summary table + `results/exp3/results.json` with per-strategy per-skill success rates.

---

## 🤖 Architecture — Base Model

The core agent (`agents/skill_agent.py`) uses **PPO** with two network modes:

```
SharedPolicyNet (baseline — causes collapse):
  obs (8,) → Linear(128) → ReLU → Linear(128) → ReLU
                                                  ├── policy_head → action logits (4,)
                                                  └── value_head  → V(s) scalar

IsolatedHeadPolicyNet (mitigation):
  obs (8,) → shared encoder (same as above)
                                 ├── policy_heads[skill_id] → action logits
                                 └── value_heads[skill_id]  → V(s)
```

The **observation space** (8-dim, all normalized to [0,1]):

```
[agent_x, agent_y, goal_x, goal_y, goal_type, hazard_x, hazard_y, steps_remaining]
```

**Skills** are goal-conditioned tasks injected via `env.set_skill(skill_dict)`:
- `goal_type=0` — Navigate to goal
- `goal_type=1` — Collect item at goal
- `goal_type=2` — Reach goal while avoiding hazard

---

## 📊 Example Results

*(Fill these in after running your experiments)*

**Experiment 1 — Retention by Skill Age:**

| Skill Age | Avg. Retention |
|-----------|---------------|
| 0 (just learned) | ~0.85 |
| 5 skills ago | TBD |
| 10 skills ago | TBD |
| 15 skills ago | TBD |

**Experiment 3 — Mitigation Comparison:**

| Strategy | Avg Success Rate | vs Baseline |
|----------|-----------------|-------------|
| Baseline | TBD | — |
| Skill Replay | TBD | TBD |
| EWC | TBD | TBD |
| Isolated Heads | TBD | TBD |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **PyTorch 2.0+** | PPO policy networks |
| **Gymnasium** | RL environment interface |
| **NumPy** | Skill embeddings, retention matrix |
| **Matplotlib** | Heatmaps, retention curves |

---

## 🔗 Related Work

| Paper | Relevance |
|-------|-----------|
| **SkillRL** (2026) | Direct inspiration — gap identified here |
| **EWC** — Kirkpatrick et al. (2017) | Continual learning baseline used in Exp 3 |
| **Progressive Networks** — Rusu et al. (2016) | Inspiration for Isolated Heads strategy |
| **PackNet** — Mallya & Lazebnik (2018) | Binary masking for skill protection |
| **DIAYN** — Eysenbach et al. (2018) | Skill discovery (acquisition, not retention) |
| **HIRO** — Nachum et al. (2018) | Hierarchical RL baseline |

---

## 🤝 Contributing

Open research project. Contributions welcome for:
- Testing on new environments (MuJoCo, Atari)
- New mitigation strategies (distillation, LoRA-style skill adapters)
- Theoretical analysis of the collapse threshold
- Scaling to LLM-based tool-use agents

Please open an issue first to discuss.

---

## 📝 Citation

```bibtex
@misc{skillcollapse2026,
  title   = {Skill Collapse in Hierarchical Reinforcement Learning},
  author  = {Ruparani Thupakula},
  year    = {2026},
  url     = {https://github.com/Ruparani777/-Skill-Collapse-in-Hierarchical-RL}
}
```

---

*Inspired by the research gap identified in SkillRL (2026). Built as an independent open-source portfolio project.*
