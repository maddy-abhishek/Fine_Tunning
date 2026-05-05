# ATS Resume Screener — Fine-Tuned LLM with LoRA

## What I Built
An AI-powered resume screening system that automatically scores resumes against job descriptions and returns a relevance score, missing keywords, and improvement recommendations — all as structured JSON output.

## Problem I Solved
Traditional ATS systems use rigid keyword matching and miss context entirely. A resume saying "built scalable APIs" gets rejected if the JD says "REST API development" — same skill, different words. I replaced that with a fine-tuned LLM that understands semantic relevance and gives recruiters an explainable score instead of a binary pass/fail.

## What I Did
- Took a 1,200-row structured resume dataset with 14 columns and serialized it into natural language text
- Cleaned and normalized the text, derived relevance labels using skill overlap and experience heuristics
- Filled missing job descriptions using role-specific fallback templates
- Split data 80/10/10, formatted into Llama chat-style JSONL for fine-tuning
- Fine-tuned TinyLlama-1.1B using LoRA (only ~0.24% of parameters trained) with 4-bit NF4 quantization
- Built an inference pipeline that takes a PDF resume + job description and returns a full ATS analysis

## Model & Why
I used **TinyLlama-1.1B** with **LoRA via the PEFT library** trained using **SFTTrainer from TRL**.

Chose TinyLlama because it fits on a free Colab T4 GPU with 4-bit quantization — LLaMA-3-8B requires a paid A100. LoRA was chosen over full fine-tuning because it trains only a small set of adapter weights injected into attention layers, making it fast, memory-efficient, and the adapter file is only a few hundred MB instead of the full model.

## Output
```json
{
  "relevance_score": 75.4,
  "missing_keywords": ["AWS", "Kubernetes"],
  "recommendations": [
    "Add cloud platform experience",
    "Include containerization tools"
  ]
}
```
Candidates scoring above 60 are shortlisted automatically.

## Problems I Faced

**1. No labels in the dataset** — The CSV had no human-annotated scores so I had to derive weak labels using a rule-based heuristic (skill overlap + experience + title matching). This means the model learns to replicate a formula, not actual recruiter judgment.

**2. fp16 + 4-bit quantization conflict** — Training crashed with `NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16`. Root cause was `fp16=True` in TrainingArguments conflicting with bitsandbytes 4-bit compute dtype. Fixed by disabling fp16, switching compute dtype to bfloat16 on A100 or disabling both on T4, and using `paged_adamw_8bit` optimizer.

**3. Column name assumptions** — Initial code assumed Kaggle-style column names (`Resume_str`, `Resume_html`) but the actual dataset had 14 structured columns with completely different names. Always `print(df.columns.tolist())` before writing any column-specific code.

**4. Small dataset** — 1,200 rows gives ~960 training examples after splitting. The model learns the scoring pattern but generalizing to unseen industries and roles requires significantly more data (10K+ ideally).

## Stack
- **Model:** TinyLlama-1.1B-Instruct
- **Fine-tuning:** LoRA (PEFT) + SFTTrainer (TRL)
- **Quantization:** BitsAndBytes 4-bit NF4
- **Framework:** Hugging Face Transformers
- **Environment:** Google Colab (T4 / A100)
- **Language:** Python
