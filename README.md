# LiquiMedThink 1.2B

[![Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/Rumiii/LiquiMedThink1.2B)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://huggingface.co/Rumiii/LiquiMedThink1.2B)


<img width="1200" height="900" alt="liquimedthink_summary" src="https://github.com/user-attachments/assets/a6f9fb7a-0694-43fb-8441-9a09745da268" />




A fine-tuned version of [LiquidAI/LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) adapted for medical chain-of-thought reasoning. The model retains the base model's explicit reasoning traces while developing clinical domain knowledge through supervised fine-tuning on a purpose-built medical reasoning dataset.

---

## Model Details

| Property | Value |
|---|---|
| Base model | LiquidAI/LFM2.5-1.2B-Thinking |
| Parameters | 1.17B |
| Architecture | Hybrid LIV convolution + GQA (16 layers) |
| Fine-tuning method | QLoRA (4-bit) via Unsloth |
| Trainable parameters | 9,142,272 (0.78% of total) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Max sequence length | 4096 tokens |
| Training dataset | FreedomIntelligence/medical-o1-reasoning-SFT (19,704 samples) |
| Training epochs | 1 |
| Effective batch size | 16 |
| Learning rate | 2e-4 (cosine schedule) |
| Optimizer | AdamW 8-bit |
| Hardware | Kaggle Tesla T4 (15.6 GB VRAM) |
| Training time | ~3 hours |
| Final training loss | 1.718 |

---

## Training Data

The model was trained on the English split of [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT), a dataset containing 19,704 clinical questions paired with two distinct components per sample:

- `Complex_CoT` — a genuine internal reasoning trace written as exploratory, step-by-step clinical thinking, including uncertainty, self-correction, and differential reasoning
- `Response` — a clean, concise final answer, separate in both content and style from the reasoning trace

Each training sample was structured as follows:

```
User:      [clinical question]
Assistant: <think>
           [Complex_CoT — the full reasoning trace]
           </think>

           [Response — the final answer]
```

The separation between reasoning trace and final answer is what preserves the base model's thinking behaviour through fine-tuning. Using a dataset where `Complex_CoT` and `Response` are genuinely different content — rather than reformatted versions of the same text — is the critical design decision behind this model.

---

## Installation

```bash
pip install torch transformers accelerate bitsandbytes
```

---

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, re

model_id = "Rumiii/LiquiMedThink1.2B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    device_map="auto",
)
model.eval()

question = "A 45-year-old male presents with crushing chest pain radiating to the left arm, diaphoresis, and shortness of breath for 30 minutes. What is the most likely diagnosis and immediate management?"

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": question}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.05,
        top_k=50,
        repetition_penalty=1.05,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

raw = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=False,
)

match = re.search(r"<think>(.*?)</think>(.*)", raw, re.DOTALL)
if match:
    thinking = match.group(1).strip()
    answer   = re.sub(r"<[^>]+>", "", match.group(2)).strip()
    print("THINKING:\n", thinking)
    print("\nANSWER:\n", answer)
else:
    print(re.sub(r"<[^>]+>", "", raw).strip())
```

---

## Sample Output

**Question:** A 45-year-old male presents with crushing chest pain radiating to the left arm, diaphoresis, and shortness of breath for 30 minutes. What is the most likely diagnosis and immediate management?

**Thinking trace (abridged):**
The presentation of crushing chest pain radiating to the left arm, combined with diaphoresis and shortness of breath, strongly points toward an acute coronary event. The duration of 30 minutes is significant — this exceeds the threshold for transient ischemia and raises concern for myocardial infarction. I need to consider STEMI versus NSTEMI and think through the immediate priorities...

**Final answer:**
The most likely diagnosis is acute myocardial infarction, specifically STEMI given the classic presentation. Immediate management includes activating emergency services, obtaining a 12-lead ECG, administering aspirin 300mg, establishing IV access, and arranging urgent percutaneous coronary intervention within 90 minutes of first medical contact.

---

## Research Background

This model investigates the relationship between training data format and reasoning trace retention in small language models.

The key finding: **thinking ability retention during SFT is determined by dataset format, not model capacity.** A dataset where the reasoning trace and final answer are genuinely different in content and purpose is required to preserve explicit chain-of-thought behaviour through fine-tuning. Artificially wrapping answers in thinking tags produces structural imitation without semantic reasoning.

---

## Intended Use

This model is intended for:

- Research into small language model capabilities in healthcare and clinical reasoning
- Medical education and clinical reasoning demonstration
- Prototyping and evaluation of medical AI pipelines
- Study of how fine-tuning data format affects reasoning behaviour in instruction-tuned models

This model is not intended for:

- Clinical decision support in real patient care
- Diagnostic or treatment decisions in any clinical setting
- Replacement of licensed medical professionals
- Any high-stakes medical application without expert oversight

---

## Limitations

- Clinical accuracy is not guaranteed. The model can produce plausible-sounding but clinically incorrect responses, particularly for rare or complex presentations. All outputs must be verified by a qualified medical professional.
- Thinking traces are exploratory, not authoritative. The reasoning trace reflects the style of the training data — the model thinks out loud and sometimes reaches incorrect conclusions.
- Single-epoch training on approximately 20,000 samples limits performance on rare or highly specialised clinical topics.
- Knowledge cutoff follows the base model (mid-2024). Clinical guidelines or drug approvals published after this date are not reflected.
- English only.

---

## License

Apache 2.0, consistent with the base model license.

---

## Author

Rumi Iqbal Sufi
Graduate Trainee, Excelra Knowledge Solutions, Hyderabad
HuggingFace: [Rumiii](https://huggingface.co/Rumiii)
GitHub: [sufirumii](https://github.com/sufirumii)
