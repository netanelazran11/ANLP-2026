# ANLP Exercise 2 - Part 2

**Netanel Azran** | Advanced NLP, MSc — Exercise 2 (June 2026)

Sentiment analysis failure analysis on two 8B instruct models:
`Qwen/Qwen3-8B` and `meta-llama/Llama-3.1-8B-Instruct`.

## Failure Type Investigated

**Contrast / expectation-reversal sentences** — inputs that start with
strong sentiment language (positive or negative) but conclude with the
*opposite* overall sentiment (e.g., glowing praise followed by a damning
verdict, or a bleak opening redeemed by a positive ending).

**Hypothesis:** The models over-weight salient, emotionally charged words
that appear early in the sentence and under-weight the final evaluative
clause that carries the true overall sentiment.

The hypothesis tests vary word order, conjunction type, adjective intensity,
and the presence of negation to map the precise extent of the problem.

## Setup on Moriah

```bash
git clone https://github.com/netanelazran11/anlp-ex2.git
cd anlp-ex2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`meta-llama/Llama-3.1-8B-Instruct` is gated. Accept the license on HuggingFace
and log in once before running:

```bash
huggingface-cli login   # or: export HF_TOKEN=hf_xxx
```

## Run

With Slurm:

```bash
sbatch run.sbatch
```

Or directly on a GPU node:

```bash
python ex2_part2.py
```

The script prints `gold -> model output` for every input and saves four
result tables as CSV files:

| File | Contents |
|------|----------|
| `table1_break_model1.csv` | Break examples — Qwen3-8B |
| `table2_hyp_model1.csv` | Hypothesis tests — Qwen3-8B |
| `table3_break_model2.csv` | Break examples — Llama-3.1-8B-Instruct |
| `table4_hyp_model2.csv` | Hypothesis tests — Llama-3.1-8B-Instruct |
