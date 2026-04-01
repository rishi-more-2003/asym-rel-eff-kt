# Can Adding Asymmetric Relation Dynamics Make Knowledge Transfer More Efficient in Language Agents?

**EN.601.773 Machine Social Intelligence, Spring 2026 — Johns Hopkins University**

[Rishi More](mailto:rmore2@jhu.edu)

[[Paper]](documentation/final_report.tex) [[Slides]](documentation/presentation.tex) [[Proposal]](documentation/proposal.tex)

---

<p align="center">
  <img src="documentation/figures/learning_curves.png" width="48%"/>
  <img src="documentation/figures/training_success.png" width="48%"/>
</p>

## TL;DR

We investigate whether a persistent **caregiver–child relationship** improves knowledge transfer in language agents. A Qwen3-235B caregiver teaches a Qwen3-8B child across 160 household tasks using a cognitively-inspired memory architecture (instinct buffer, working memory, long-term episodic store, LoRA-based habits) and a **salience-gated consolidation** mechanism. Caregiver-assisted agents achieve **100% training success** with **25% fewer turns**, but this advantage **does not transfer** to independent evaluation — mirroring the *scaffolding dependency* phenomenon from developmental psychology.

## Key Results

| Metric | Solo | Sym. Peer | Role-Labeled | **Relational** |
|:---|:---:|:---:|:---:|:---:|
| H1 Transfer Accuracy | 0.686 ± 0.046 | 0.686 ± 0.078 | **0.708 ± 0.033** | 0.682 ± 0.087 |
| Training Success Rate | 81.5% | 84.4% | **100%** | 99.8% |
| Avg. Turns to Complete | 8.36 | 8.52 | **6.10** | 6.42 |
| Total LoRA Updates | 130 | 135 | **160** | 160 |

<p align="center">
  <img src="documentation/figures/teaching_efficiency.png" width="48%"/>
  <img src="documentation/figures/category_heatmap.png" width="48%"/>
</p>

## Setup

### Requirements

- Python 3.10+
- [Tinker API](https://thinkingmachines.ai/tinker/) key

### Installation

```bash
git clone https://github.com/rmore2/msi-project.git
cd msi-project

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
TINKER_API_KEY="your-tinker-api-key-here"
```

All hyperparameters are centralized in [`config.py`](config.py).

## Usage

### 1. Generate Task Database

```bash
python run_generate_tasks.py
```

Runs the multi-stage generation pipeline (ontology → skeletons → expansion → verification → filtering) to produce 197 household tasks. Output: `data/task_database.json`.

### 2. Run Experiment

```bash
# Run full pipeline (training + evaluation)
python run_experiment.py all

# Or run phases separately
python run_experiment.py train
python run_experiment.py eval
```

This launches 12 concurrent training runs (4 conditions × 3 seeds) via `asyncio`, then evaluates on 40 held-out tasks.

### 3. Generate Figures

```bash
python analyze_results.py
```

Produces publication-quality figures in `documentation/figures/`.

## Project Structure

```
├── config.py                    # Centralized hyperparameters
├── run_generate_tasks.py        # Task generation entry point
├── run_experiment.py            # Training + evaluation entry point
├── analyze_results.py           # Figure generation
├── requirements.txt
│
├── src/
│   ├── agents/
│   │   ├── caregiver.py         # 235B caregiver agent
│   │   └── child.py             # 8B child agent with LoRA
│   ├── memory/
│   │   ├── instinct.py          # M1: Fixed behavioral priors
│   │   ├── working_memory.py    # M2: Sliding window (k=8)
│   │   └── long_term.py         # M3: Episodic store + BM25 retrieval
│   ├── salience.py              # Salience signal (novelty + pred. error + teaching)
│   ├── trainer.py               # M4: Salience-weighted LoRA SFT
│   ├── episode.py               # Async episode runner
│   ├── training.py              # Concurrent training loop
│   ├── evaluation.py            # H1/H2/H3 evaluation suite
│   ├── judge.py                 # Semantic action judge (235B)
│   ├── scaffolding.py           # Adaptive scaffolding controller
│   ├── child_model.py           # Caregiver's model of the child
│   ├── curriculum.py            # Difficulty-ordered curriculum
│   ├── conditions.py            # Experimental condition configs
│   ├── bm25.py                  # Dependency-free BM25
│   ├── reward.py                # Reward computation
│   ├── metrics_logger.py        # JSONL metrics logging
│   ├── tinker_utils.py          # Tinker API utilities
│   ├── ontology.py              # Object ontology generation
│   ├── generate_skeletons.py    # Task skeleton generation
│   ├── expand_tasks.py          # Full task expansion
│   ├── verify_tasks.py          # Self-verification loop
│   └── filter_tasks.py          # Dedup + balancing + train/eval split
│
├── data/
│   ├── task_database.json       # Generated task database (197 tasks)
│   ├── object_ontology.json     # Household object ontology
│   ├── task_skeletons.json      # Intermediate skeletons
│   └── runs/                    # Experiment outputs
│       ├── evaluation_results.json
│       ├── {condition}_seed{n}/
│       │   ├── metrics.jsonl    # Per-episode metrics
│       │   ├── ltm.json         # Long-term memory state
│       │   └── transcripts/     # Full episode dialogues
│       └── ...
│
└── documentation/
    ├── final_report.tex         # Final report
    ├── presentation.tex         # Beamer slides
    ├── bibliography.bib         # References
    └── figures/                 # Generated figures (PDF + PNG)
```

## Method

### Memory Architecture

| Module | Implementation | Updated? |
|:---|:---|:---:|
| **M1.** Instinct Buffer | Fixed system prompt (role priors) | Never |
| **M2.** Working Memory | Last *k*=8 dialogue turns | Every turn |
| **M3.** Long-Term Memory | Episodic store, BM25 retrieval, 235B compression | If salience > τ |
| **M4.** Habit Store | LoRA adapter (rank 16, lr=2e-5, batch ≥ 4) | Salience-weighted SFT |

### Salience Signal

$$s = \alpha \cdot \text{novelty}(e) + \beta \cdot \text{prediction\\_error}(e) + \gamma \cdot \text{teaching}(e)$$

- **Novelty** (α=0.3): Category frequency decay + BM25 distance to LTM
- **Prediction error** (β=0.4): Rescorla-Wagner surprise + ZPD match (Gaussian at *r*=0.5)
- **Teaching signal** (γ=0.3): Productive struggle patterns + effort ratio; γ=0 without caregiver

### Experimental Conditions

| Condition | Agents | Teaching Signal |
|:---|:---|:---:|
| Solo | 8B child alone | γ = 0 |
| Symmetric Peer | Two 8B agents | γ = 0 |
| Role-Labeled | 235B caregiver + 8B child | γ = 0 |
| **Relational** | 235B caregiver + 8B child | **γ = 0.3** |

## Figures

<p align="center">
  <img src="documentation/figures/h1_transfer.png" width="38%"/>
  <img src="documentation/figures/transfer_by_difficulty.png" width="58%"/>
</p>
<p align="center">
  <img src="documentation/figures/salience_ltm.png" width="48%"/>
  <img src="documentation/figures/difficulty_progression.png" width="48%"/>
</p>

## Citation

```bibtex
@misc{more2026asymmetric,
  author = {More, Rishi},
  title  = {Can Adding Asymmetric Relation Dynamics Make Knowledge
            Transfer More Efficient in Language Agents?},
  year   = {2026},
  school = {Johns Hopkins University},
  note   = {EN.601.773 Machine Social Intelligence, Spring 2026}
}
```

## Acknowledgements

This project uses the [Tinker API](https://thinkingmachines.ai/tinker/) for LLM inference and LoRA fine-tuning. Experiments were run on the JHU CS research compute cluster. Total API cost: ~$45.

## License

This project is for academic purposes. Please contact the author for usage inquiries.
