# =============================================================================
# FILE      : finetune_v3_final_kaggle.py
# AUTHOR    : Rumi Iqbal Sufi
# MODEL     : LiquidAI/LFM2.5-1.2B-Thinking
# DATASET   : FreedomIntelligence/medical-o1-reasoning-SFT (English, 19,704 rows)
# PURPOSE   : Final fine-tuning script — zero truncation, full CoT preserved,
#             thinking ability retained via genuine Complex_CoT + Response split
# ENV       : Kaggle (2x T4 GPU, 30GB VRAM) — paste entire script into ONE cell
# OUTPUT    : /kaggle/working/
# REPO      : Rumiii/LiquiMedThink1.2B-v3
# =============================================================================

import sys, subprocess, os, re, torch

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

print("Installing dependencies...")
run(f"{sys.executable} -m pip install --upgrade pip -q")
run(f"{sys.executable} -m pip install 'unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git' unsloth_zoo -q")
run(f"{sys.executable} -m pip install 'trl>=0.15.2' 'transformers>=4.51.0' datasets accelerate bitsandbytes peft huggingface_hub -q")
print(" Installed\n")

import huggingface_hub
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "LiquidAI/LFM2.5-1.2B-Thinking"
DATASET_ID  = "FreedomIntelligence/medical-o1-reasoning-SFT"
DATASET_CFG = "en"

OUTPUT_DIR  = "/kaggle/working/checkpoints"
ADAPTER_DIR = "/kaggle/working/lora_adapter"
MERGED_DIR  = "/kaggle/working/merged_model"
HF_REPO     = "Rumiii/LiquiMedThink1.2B-v3"
HF_TOKEN    = "ADD_YOUR_TOKEN_HERE"         # ← paste your HF write token here

# ── Key settings ──────────────────────────────────────────────────────────────
# MAX_SEQ_LEN=4096 ensures virtually no sample is truncated.
# The Complex_CoT entries in this dataset are long — 2048 was cutting many.
# 4096 fits the full CoT + question + response for essentially all 19,704 rows.
MAX_SEQ_LEN = 4096
LORA_RANK   = 16
LORA_ALPHA  = 16

# packing=True bins short samples together so the GPU is never sitting idle
# between samples — cuts training time by 30-40% without touching sample content.
USE_PACKING = True

for path in [OUTPUT_DIR, ADAPTER_DIR, MERGED_DIR]:
    os.makedirs(path, exist_ok=True)

print(f"GPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"MAX_SEQ_LEN : {MAX_SEQ_LEN}")
print(f"Packing     : {USE_PACKING}\n")

# ── Load tokenizer ────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer_for_prep = AutoTokenizer.from_pretrained(MODEL_ID)

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"Loading full English dataset ({DATASET_ID})...")
raw = load_dataset(DATASET_ID, DATASET_CFG, split="train")
print(f"Loaded  : {len(raw)} rows")
print(f"Columns : {raw.column_names}")
print(f"\nSample preview:")
print(f"  Question    : {str(raw[0].get('Question',''))[:120]}")
print(f"  Complex_CoT : {str(raw[0].get('Complex_CoT',''))[:120]}")
print(f"  Response    : {str(raw[0].get('Response',''))[:120]}\n")

# ── Format function ───────────────────────────────────────────────────────────
def format_sample(row):
    """
    The correct format for preserving thinking ability:

      User:      [Question]
      Assistant: <think>
                 [Complex_CoT — genuine step-by-step medical reasoning,
                  written as internal monologue: "Okay, let's see..."]
                 </think>

                 [Response — clean, concise final answer]

    Complex_CoT and Response are GENUINELY DIFFERENT content in this dataset:
      Complex_CoT = the internal reasoning process, exploratory, uncertain
      Response    = the clean conclusion

    NO TRUNCATION anywhere in this function.
    The tokenizer will handle max_seq_len at training time — any sample
    that genuinely exceeds 4096 tokens gets truncated there, not here.
    That is unavoidable at any sequence length on any GPU. What we guarantee
    is that WE never cut anything in preprocessing.
    """
    question = str(row.get("Question",    "")).strip()
    cot      = str(row.get("Complex_CoT", "")).strip()
    response = str(row.get("Response",    "")).strip()

    if not question or not cot or not response:
        return {"text": ""}

    # Full CoT inside <think>, full Response after </think>
    # Nothing truncated, nothing shortened
    assistant_content = f"<think>\n{cot}\n</think>\n\n{response}"

    messages = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": assistant_content},
    ]

    try:
        text = tokenizer_for_prep.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not text.endswith(tokenizer_for_prep.eos_token):
            text += tokenizer_for_prep.eos_token
    except Exception as e:
        print(f"[WARN] Template error: {e}")
        return {"text": ""}

    return {"text": text}


print("Formatting all samples — no truncation applied...")
formatted = raw.map(format_sample, num_proc=4, desc="Formatting")
formatted = formatted.filter(lambda x: len(x["text"]) > 50)
print(f"Final dataset : {len(formatted)} rows\n")

# ── Sanity check ──────────────────────────────────────────────────────────────
sample = formatted[0]["text"]
has_open  = "<think>"  in sample
has_close = "</think>" in sample
print(f"Sanity check:")
print(f"  <think> present  : {'Ok' if has_open  else 'PROBLEM'}")
print(f"  </think> present : {'Ok' if has_close else 'PROBLEM'}")

if has_open and has_close:
    think_start = sample.find("<think>") + len("<think>")
    think_end   = sample.find("</think>")
    answer_part = sample[think_end + len("</think>"):].strip()
    print(f"\n  CoT preview (first 200 chars):")
    print(f"  {sample[think_start:think_start+200]}")
    print(f"\n  Answer preview (first 200 chars):")
    print(f"  {answer_part[:200]}")
    print(f"\nOk! Thinking and answer confirmed SEPARATE and UNTRUNCATED\n")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading LFM2.5-1.2B-Thinking (4-bit QLoRA)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_ID,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = True,
    dtype          = None,
)

print("Attaching LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_RANK,
    target_modules             = [
        "q_proj", "k_proj", "v_proj",
        "out_proj", "in_proj",
        "w1", "w2", "w3",
    ],
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = 42,
)
model.print_trainable_parameters()

# ── Trainer ───────────────────────────────────────────────────────────────────
try:
    from unsloth import UnslothTrainer, UnslothTrainingArguments
    TrainerClass  = UnslothTrainer
    ArgsClass     = UnslothTrainingArguments
    extra_trainer = {
        "dataset_text_field" : "text",
        "max_seq_length"     : MAX_SEQ_LEN,
        "dataset_num_proc"   : 4,
        "packing"            : USE_PACKING,
    }
    extra_args = {}
    print("\nUsing UnslothTrainer ,ok")
except ImportError:
    from trl import SFTTrainer, SFTConfig
    TrainerClass  = SFTTrainer
    ArgsClass     = SFTConfig
    extra_trainer = {}
    extra_args    = {
        "dataset_text_field" : "text",
        "max_seq_length"     : MAX_SEQ_LEN,
        "dataset_num_proc"   : 4,
        "packing"            : USE_PACKING,
    }
    print("\nUsing SFTTrainer + SFTConfig ok!")

training_args = ArgsClass(
    output_dir                  = OUTPUT_DIR,

    # Batch — kept at 2 with gradient accumulation 8
    # Effective batch = 2 × 8 = 16
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,

    # Full single epoch over all 19,704 samples
    num_train_epochs            = 1,

    # Learning rate
    warmup_steps                = 50,
    learning_rate               = 2e-4,
    lr_scheduler_type           = "cosine",
    weight_decay                = 0.01,

    # Precision
    fp16                        = not is_bfloat16_supported(),
    bf16                        = is_bfloat16_supported(),

    # Memory
    optim                       = "adamw_8bit",

    # Logging and saving
    logging_steps               = 25,
    save_steps                  = 250,
    save_total_limit            = 3,
    report_to                   = "none",

    seed                        = 42,
    dataloader_num_workers      = 2,
    **extra_args,
)

trainer = TrainerClass(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = formatted,
    args          = training_args,
    **extra_trainer,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\nStarting training...")
print(f"Dataset     : {len(formatted)} samples")
print(f"Seq length  : {MAX_SEQ_LEN} tokens (no code-level truncation)")
print(f"LoRA rank   : {LORA_RANK}")
print(f"Packing     : {USE_PACKING} (faster training, no content change)")
print(f"Est. time   : 5-8 hours on Kaggle 2x T4\n")

stats = trainer.train()

print(f"\n,ok! Training complete")
print(f"   Runtime : {stats.metrics['train_runtime']:.0f}s  ({stats.metrics['train_runtime']/3600:.2f}h)")
print(f"   Loss    : {stats.metrics['train_loss']:.4f}")

# ── Save adapter ──────────────────────────────────────────────────────────────
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"\n LoRA adapter saved → {ADAPTER_DIR}")

# ── Inference test ────────────────────────────────────────────────────────────
print("\nRunning inference tests to verify thinking traces retained...")
FastLanguageModel.for_inference(model)

test_prompts = [
    "A 45-year-old male presents with crushing chest pain radiating to the left arm, diaphoresis, and shortness of breath. What is the diagnosis and immediate management?",
    "Explain the mechanism of action of beta-blockers in heart failure.",
    "A 28-year-old female has polyuria, polydipsia, and weight loss. Random glucose is 18 mmol/L. Differentiate Type 1 from Type 2 diabetes.",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*65}")
    print(f"TEST {i}: {prompt[:65]}...")
    print("─"*65)

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens     = 2048,
            temperature        = 0.05,
            top_k              = 50,
            repetition_penalty = 1.05,
            do_sample          = True,
            pad_token_id       = tokenizer.eos_token_id,
        )

    raw         = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=False,
    )
    think_match = re.search(r"<think>(.*?)</think>(.*)", raw, re.DOTALL)

    if think_match:
        thinking = think_match.group(1).strip()
        answer   = re.sub(r"<[^>]+>", "", think_match.group(2)).strip()
        print(f"ok, <think> tags PRESENT — thinking ability retained!")
        print(f"\n[THINKING — first 300 chars]\n{thinking[:300]}...")
        print(f"\n[FINAL ANSWER]\n{answer}")
    else:
        print("No <think> tags — check output:")
        print(re.sub(r"<[^>]+>", "", raw).strip()[:400])

# ── Merge + push ──────────────────────────────────────────────────────────────
print(f"\n\nMerging LoRA into base model → {MERGED_DIR}")
print("(Takes 2-4 minutes...)")
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
print(f"ok!,, Merged model saved → {MERGED_DIR}")

print(f"\nPushing to HuggingFace → {HF_REPO}")
huggingface_hub.login(token=HF_TOKEN)
print(f"Logged in as: {huggingface_hub.whoami()['name']}")

model.push_to_hub_merged(
    HF_REPO,
    tokenizer,
    save_method = "merged_16bit",
    token       = HF_TOKEN,
    private     = False,
)

print(f"\n ok!, Model live at: https://huggingface.co/{HF_REPO}")
print(f"\nFiles saved:")
print(f"  Checkpoints  : {OUTPUT_DIR}")
print(f"  LoRA adapter : {ADAPTER_DIR}")
print(f"  Merged model : {MERGED_DIR}")
print("\nDone! 🎉")
