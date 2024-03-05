import gradio as gr
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import fire
import torch
import os
import sys
import time
from typing import List

from transformers import LlamaTokenizer
from safety_utils import get_safety_checker
from model_utils import load_model, load_peft_model

model = load_model(model_name='/mnt/workspace/demos/llama2_lora/llama2-7b',quantization=True)
tokenizer = LlamaTokenizer.from_pretrained('/mnt/workspace/demos/llama2_lora/llama2-7b')
tokenizer.add_special_tokens(
    {
        "pad_token": "<PAD>",
    }
)
model = load_peft_model(model, '/mnt/workspace/demos/llama2_lora/sft-llama2-7b')
model.eval()

def inference(text):
    text='Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n ### Instruction:\n'+text+'\n\n### Response:'
    batch = tokenizer([text], return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=100,
            do_sample=True,
            top_p=1,
            temperature=1,
            min_length=None,
            use_cache=True,
            top_k=50,
            repetition_penalty=1,
            length_penalty=1,
        )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text[len(text):]

demo = gr.Blocks()
with demo:
    input_prompt = gr.Textbox(label="Llama对话助手", 
                                value="What are the three primary colors?",
                                lines=6)
    generated_txt = gr.Textbox(lines=6)

    b1 = gr.Button("发送")
    b1.click(inference, inputs=[input_prompt], outputs=generated_txt) 

demo.launch(enable_queue=True, share=True)