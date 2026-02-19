"""
Medical Domain LLM Assistant — Gradio Inference Application

This script loads the fine-tuned TinyLlama LoRA adapter and launches an
interactive Gradio web interface. Run this after completing the fine-tuning
notebook, or point MODEL_DIR at any compatible checkpoint directory.

Usage:
    python gradio_app.py

Requirements:
    pip install transformers peft bitsandbytes accelerate gradio torch
"""

import math
import os
import sys
import torch
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "./medical_assistant_final")
MAX_SEQ_LENGTH = 512

SYSTEM_PROMPT = (
    "You are a knowledgeable and accurate medical assistant. "
    "Answer medical questions clearly and concisely based on established medical knowledge. "
    "If a question is outside the medical domain, politely indicate that it is outside your specialty."
)


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------

def load_model_and_tokenizer(base_model_name: str, adapter_dir: str):
    """
    Load the base model with 4-bit quantization, then apply the LoRA adapter.
    Falls back to the base model only if the adapter directory is not found.
    """
    print(f"Loading tokenizer from: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = True

    if os.path.isdir(adapter_dir):
        print(f"Loading LoRA adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        print("Fine-tuned model loaded successfully.")
    else:
        print(
            f"WARNING: Adapter directory '{adapter_dir}' not found. "
            "Running with base model only. "
            "Set the ADAPTER_DIR environment variable to point to your fine-tuned adapter."
        )
        model = base_model

    model.eval()
    return model, tokenizer


# -----------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------

def build_prompt(user_message: str) -> str:
    """Construct the TinyLlama chat prompt for a given user message."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{user_message}</s>\n"
        f"<|assistant|>\n"
    )


def generate_response(
    model,
    tokenizer,
    user_message: str,
    max_new_tokens: int = 200,
    temperature: float = 0.1,
) -> str:
    """
    Generate a response for a given user message using the loaded model.
    Only the newly generated tokens (after the prompt) are returned.
    """
    prompt = build_prompt(user_message)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH - max_new_tokens,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# -----------------------------------------------------------------------
# Gradio interface
# -----------------------------------------------------------------------

def build_interface(model, tokenizer):
    """Construct and return the Gradio Blocks interface."""

    def chat(user_message: str, history: list, max_new_tokens: int, temperature: float):
        if not user_message.strip():
            return history, ""
        response = generate_response(model, tokenizer, user_message, max_new_tokens, temperature)
        history.append((user_message, response))
        return history, ""

    with gr.Blocks(
        title="Medical Assistant — Fine-Tuned TinyLlama",
        theme=gr.themes.Soft(),
        css=".disclaimer { font-size: 0.8em; color: #888; border-top: 1px solid #ddd; padding-top: 8px; }",
    ) as demo:

        gr.Markdown(
            """
            # Medical Domain Assistant
            **Model:** TinyLlama-1.1B-Chat fine-tuned on medical Q&A (LoRA / PEFT)

            Ask any medical question. The model is specialized on medical flashcard content
            covering pharmacology, physiology, pathology, and clinical medicine.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                    bubble_full_width=False,
                )
                user_input = gr.Textbox(
                    label="Your Medical Question",
                    placeholder="e.g., What is the mechanism of action of beta-blockers?",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=3)
                    clear_btn = gr.Button("Clear", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### Generation Settings")
                max_tokens_slider = gr.Slider(
                    minimum=50,
                    maximum=400,
                    value=200,
                    step=25,
                    label="Max New Tokens",
                    info="Maximum number of tokens the model will generate.",
                )
                temperature_slider = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.1,
                    step=0.05,
                    label="Temperature",
                    info="Lower values produce more deterministic answers.",
                )
                gr.Markdown(
                    """
                    ### Example Questions
                    - What is the mechanism of action of aspirin?
                    - What are the classic signs of congestive heart failure?
                    - Explain the pathophysiology of asthma.
                    - What differentiates Type 1 from Type 2 diabetes?
                    - What causes essential hypertension?
                    - What are the first-line antibiotics for community-acquired pneumonia?
                    - Describe the Starling law of the heart.
                    """
                )

                gr.Markdown(
                    """
                    <div class="disclaimer">
                    <b>Disclaimer:</b> This model is for educational purposes only.
                    It does not constitute medical advice, diagnosis, or treatment.
                    Always consult a qualified healthcare professional.
                    </div>
                    """,
                    elem_classes="disclaimer",
                )

        # Event bindings
        submit_btn.click(
            fn=chat,
            inputs=[user_input, chatbot, max_tokens_slider, temperature_slider],
            outputs=[chatbot, user_input],
        )
        user_input.submit(
            fn=chat,
            inputs=[user_input, chatbot, max_tokens_slider, temperature_slider],
            outputs=[chatbot, user_input],
        )
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, user_input],
        )

    return demo


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_mem:.1f} GB)")

    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_NAME, ADAPTER_DIR)

    demo = build_interface(model, tokenizer)

    # share=True generates a public Gradio URL valid for 72 hours
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
