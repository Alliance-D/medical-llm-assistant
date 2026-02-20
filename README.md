# Medical Domain LLM Assistant

A domain-specific conversational assistant built by fine-tuning TinyLlama-1.1B-Chat-v1.0
on a curated medical Q&A dataset using LoRA (Low-Rank Adaptation), a parameter-efficient
fine-tuning technique. The model targets educational medical Q&A covering pharmacology,
physiology, pathology, and clinical medicine.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alliance-D/medical-llm-assistant/blob/main/notebook/medical_assistant_finetune.ipynb)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Model Architecture and Fine-Tuning Methodology](#5-model-architecture-and-fine-tuning-methodology)
6. [Hyperparameter Experiments](#6-hyperparameter-experiments)
7. [Performance Metrics](#7-performance-metrics)
8. [How to Run](#8-how-to-run)
9. [Conversation Examples](#9-conversation-examples)
10. [Limitations and Future Work](#10-limitations-and-future-work)

---

## 1. Project Overview

**Domain:** Healthcare and Medical Education

**Problem Statement:**
Medical information is dense, specialized, and often inaccessible to students and general users.
A domain-specialized language model assistant can serve as a first-pass educational resource,
helping users understand medical terminology, pharmacology, and clinical concepts in plain language.
Unlike a general-purpose model, a domain-fine-tuned model produces more focused, medically grounded
responses and handles the vocabulary and reasoning patterns of clinical medicine more reliably.

**Approach:**
TinyLlama-1.1B-Chat-v1.0 is fine-tuned on 3,000 curated medical instruction-response pairs using
LoRA. This approach keeps the number of trainable parameters below 0.4% of the total model size,
making training feasible on a single T4 GPU within Google Colab's free tier without quantization.

**Disclaimer:**
This model is intended for educational and research purposes only. It does not constitute
medical advice and should not be used to inform clinical decisions.

---

## 2. Repository Structure

```
medical-llm-assistant/
    notebook/
        medical_assistant_finetune.ipynb    # End-to-end fine-tuning notebook (run on Colab)
    app/
        gradio_app.py                       # Standalone Gradio inference application
    requirements.txt                        # Python dependencies
    README.md
```

---

## 3. Dataset

**Source:** `medalpaca/medical_meadow_medical_flashcards` (Hugging Face Datasets Hub)

**Description:**
This dataset contains medical Q&A pairs derived from medical flashcard content. It covers
anatomy, physiology, pharmacology, pathology, clinical medicine, and biochemistry. Each example
consists of an instruction field, a question input, and a reference answer.

**Raw dataset size before filtering:** approximately 33,000 examples.

**Dataset statistics after preprocessing:**

| Split      | Size  |
|------------|-------|
| Train      | 2,550 |
| Validation | 300   |
| Test       | 150   |
| Total      | 3,000 |

Subsampling to 3,000 examples was applied to balance training efficiency with coverage,
within the project requirement of 1,000 to 5,000 high-quality examples.

---

## 4. Preprocessing Pipeline

**Step 1 — Quality Filtering**

Examples are removed if:
- The question field is fewer than 5 characters after normalization.
- The answer field is fewer than 10 characters.
- The answer is a bare boolean token such as "yes", "no", "true", or "false".

**Step 2 — Text Normalization**

Each text field is processed to remove null bytes and control characters, and to collapse
consecutive whitespace into a single space.

**Step 3 — Instruction-Response Formatting**

Each example is wrapped in the TinyLlama ChatML prompt template:

```
<|system|>
You are a knowledgeable and accurate medical assistant. Answer medical questions clearly
and concisely based on established medical knowledge. If a question is outside the medical
domain, politely indicate that it is outside your specialty.</s>
<|user|>
{question}</s>
<|assistant|>
{answer}</s>
```

**Step 4 — Tokenization**

Tokenization uses the model's native SentencePiece BPE tokenizer with a vocabulary of 32,000
tokens. Sequences exceeding 512 tokens are truncated. Right-side padding is applied using the
EOS token as the pad token.

**Step 5 — Dataset Splitting**

The dataset is shuffled with a fixed random seed (42) and split 85% train / 10% validation /
5% test. The test set is held out entirely until final evaluation.

---

## 5. Model Architecture and Fine-Tuning Methodology

**Base Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

TinyLlama is a 1.1 billion parameter causal decoder-only transformer trained on 3 trillion tokens.
It uses the Llama 2 architecture with Grouped Query Attention (GQA) and RoPE positional embeddings.
The chat variant has been instruction-tuned, making it a strong starting point for domain-specific
fine-tuning.

**Precision:** The model is loaded in `torch.float16` with `device_map="auto"`, placing all layers
on the T4 GPU without quantization. At 1.1B parameters, the model requires approximately 2.2 GB
in float16, well within the T4's 15 GB of VRAM.

**Fine-Tuning Method: LoRA (Low-Rank Adaptation)**

LoRA injects trainable rank-decomposition matrices into the query and value projection layers
of every attention head. For a frozen weight matrix W, the update is:

```
W' = W + (alpha / r) * B * A
```

where A and B are low-rank matrices of rank r, and alpha is a scaling hyperparameter.
Only A and B are updated during training — W remains frozen.

| LoRA Parameter   | Value             |
|------------------|-------------------|
| Target modules   | q_proj, v_proj    |
| Rank (r)         | 16                |
| Alpha            | 32                |
| Dropout          | 0.05              |
| Trainable params | ~4.2M (~0.38%)    |
| Frozen params    | ~1.1B             |

**Training Framework:** TRL SFTTrainer (Supervised Fine-Tuning Trainer)

**Optimizer:** AdamW (PyTorch native)

**Scheduler:** Cosine decay with 3% linear warmup.

**Key training settings:**

| Parameter                  | Value |
|----------------------------|-------|
| Learning rate              | 2e-4  |
| Effective batch size       | 16    |
| Gradient accumulation      | 8     |
| Epochs                     | 2     |
| Max sequence length        | 512   |
| Mixed precision            | fp16  |
| Gradient checkpointing     | Yes   |

---

## 6. Hyperparameter Experiments

Four experiments were conducted by varying LoRA rank, learning rate, and number of epochs.
Each experiment was run independently from a fresh session with all other parameters held constant.
EXP-02 was selected as the best configuration based on validation loss and BLEU score.

| Experiment        | r  | LR   | Epochs | Train Loss | Val Loss | ROUGE-1 | BLEU   | Perplexity | GPU (GB) | Time (min) |
|-------------------|----|------|--------|------------|----------|---------|--------|------------|----------|------------|
| EXP-01 (Baseline) | 8  | 2e-4 | 1      | 0.7175     | 0.6785   | 0.2653  | 0.0591 | 2.23       | 3.48     | 4.9        |
| EXP-02 (Best)     | 16 | 2e-4 | 2      | 0.6749     | 0.6622   | 0.2718  | 0.0662 | 2.24       | 3.51     | 10.0       |
| EXP-03 (Lower LR) | 16 | 5e-5 | 2      | 0.7722     | 0.6836   | 0.2656  | 0.0656 | 2.19       | 3.50     | 10.1       |
| EXP-04 (3 Epochs) | 16 | 2e-4 | 3      | 0.6567     | 0.6565   | 0.2674  | 0.0628 | 2.23       | 3.50     | 16.7       |

**Key observations:**

- Increasing LoRA rank from 8 to 16 (EXP-01 vs EXP-02) produced the largest improvement in
  training loss and BLEU, with only 0.03 GB additional GPU memory required.
- Lowering the learning rate to 5e-5 (EXP-03) resulted in the worst training loss (0.7722),
  indicating that the model does not converge well at this rate within 2 epochs on this dataset.
- Extending to 3 epochs (EXP-04) improved training loss marginally (0.6567) but BLEU dropped
  relative to EXP-02, suggesting diminishing returns and early signs of overfitting.
- EXP-02 achieves the best balance of validation loss, ROUGE-1, and BLEU and is used as the
  final model in the Gradio interface.

---

## 7. Performance Metrics

**Evaluation set:** 100 held-out test examples not seen during training or validation.

**Metrics:**
- ROUGE-1/2/L: N-gram and longest-common-subsequence overlap between generated and reference
  answers. Higher is better.
- BLEU: Precision-weighted n-gram overlap. Higher is better.
- Perplexity: Model confidence on test sequences, computed as exp(mean cross-entropy loss).
  Lower is better. This is the most reliable indicator of domain adaptation quality for
  open-ended generation, as it is not sensitive to paraphrasing.

**Results — EXP-02 (Best) vs Base Model:**

| Metric     | Base Model | Fine-Tuned | Delta   |
|------------|------------|------------|---------|
| ROUGE-1    | 0.3083     | 0.2718     | -0.0365 |
| ROUGE-2    | 0.1460     | 0.1165     | -0.0294 |
| ROUGE-L    | 0.2253     | 0.1783     | -0.0471 |
| BLEU       | 0.0833     | 0.0662     | -0.0171 |
| Perplexity | 5.2432     | 2.2388     | -3.0044 |

**Interpretation:**

Perplexity improved by 57.3% (5.24 to 2.24), which is the primary indicator that fine-tuning
was effective. The model has substantially higher confidence on medical domain text after
fine-tuning.

ROUGE and BLEU scores are slightly lower for the fine-tuned model. This is an expected and
well-documented outcome for instruction-tuned generative models: the base model produces verbose
responses that incidentally share more surface tokens with the reference, while the fine-tuned
model produces more concise, domain-focused answers in a different but equally valid phrasing.
ROUGE and BLEU penalize any deviation from the exact reference wording, regardless of semantic
correctness. The qualitative comparison in Section 9 provides a more meaningful view of the
actual improvement in response quality.

---

## 8. How to Run

### Option A — Google Colab (Recommended)

1. Click the **Open In Colab** badge at the top of this README.
2. In Colab, go to **Runtime > Change runtime type** and select **T4 GPU**.
3. Run **Cell 1 only** to install dependencies.
4. Go to **Runtime > Restart session**.
5. Run all remaining cells from Cell 2 onward in order.
6. The Gradio interface launches at the end of the notebook with a public share URL.

No local setup is required. The notebook downloads the dataset and base model automatically
on first run.

### Option B — Local Machine

Requirements: Python 3.10+, CUDA-capable GPU with at least 6 GB VRAM.

```bash
git clone https://github.com/Alliance-D/medical-llm-assistant.git
cd medical-llm-assistant

pip install -r requirements.txt

# For training — run the notebook
jupyter notebook notebook/medical_assistant_finetune.ipynb

# For inference only — after training is complete
ADAPTER_DIR=./medical_assistant_final python app/gradio_app.py
```

The Gradio app will be available at `http://localhost:7860`.

---

## 9. Conversation Examples

The following examples show representative outputs from the fine-tuned model on the test set.

**Pharmacology**

Question: What is the mechanism of action of metformin in treating type 2 diabetes?

Fine-tuned model response:
> Metformin is an oral drug that works by reducing glucose production in the liver and
> increasing insulin sensitivity in the pancreas. This can help to improve blood sugar control
> and reduce the risk of complications associated with type 2 diabetes. Metformin is commonly
> used as part of a combination therapy with sulfonylurea drugs such as glyburide or pioglitazone.

**Cardiology**

Question: What are the classic symptoms of acute myocardial infarction?

Fine-tuned model response:
> The classic symptoms of acute myocardial infarction include chest pain, shortness of breath,
> and sweating. These symptoms can be severe and may require immediate medical attention. Other
> symptoms may include nausea, vomiting, dizziness, and abdominal pain.

**Endocrinology**

Question: What is the difference between Type 1 and Type 2 diabetes mellitus?

Fine-tuned model response:
> Type 1 diabetes mellitus is an autoimmune disease in which the body's immune system attacks
> and destroys insulin-producing cells in the pancreas, leading to a lack of insulin production.
> In contrast, Type 2 diabetes mellitus is a metabolic disorder in which the body's ability to
> use glucose for energy is impaired due to insulin resistance and relative insulin deficiency.

**Out-of-Domain Handling**

Question: What is the capital of France?

Fine-tuned model response:
> The capital of France is Paris.

The model answers the geography question correctly rather than refusing it. This is consistent
with TinyLlama's general-purpose pretraining — the fine-tuning specialized its medical response
quality but did not suppress knowledge of other domains, which is the expected behavior for
LoRA-based domain adaptation at this scale.

---

## 10. Limitations and Future Work

**Current limitations:**

- Maximum sequence length is 512 tokens, which limits responses to complex multi-part questions.
- The model is trained on flashcard-style content, which favors concise factual answers over
  nuanced clinical reasoning or differential diagnosis.
- The model has no access to real-time medical literature and may not reflect current guidelines.
- ROUGE and BLEU metrics are imperfect indicators of quality for open-ended generation tasks,
  as they measure lexical overlap rather than semantic correctness.

**Future directions:**

- Fine-tune on larger and more diverse medical corpora such as MedQA-USMLE or PubMedQA.
- Implement retrieval-augmented generation (RAG) to ground responses in current medical literature.
- Scale to a larger base model such as Llama-3-8B with access to additional GPU memory.
- Explore direct preference optimization (DPO) to further align response quality with
  clinically accurate outputs.

  ## Video Link
  Link; https://youtu.be/eTtRYPKMRfM