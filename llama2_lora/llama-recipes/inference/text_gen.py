import fire
import torch
import os
import sys
import time
from typing import List

from transformers import LlamaTokenizer
from safety_utils import get_safety_checker
from model_utils import load_model, load_peft_model

model = load_model(model_name='/mnt/workspace/llama2-7b',quantization=True)
tokenizer = LlamaTokenizer.from_pretrained('/mnt/workspace/llama2-7b')
tokenizer.add_special_tokens(
    {
        "pad_token": "<PAD>",
    }
)
model = load_peft_model(model, '/mnt/workspace/sft-llama2-7b')
model.eval()

batch = tokenizer(['hello who are you?'], return_tensors="pt")
batch = {k: v.to("cuda") for k, v in batch.items()}
start = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(
        **batch,
        max_new_tokens=20,
        do_sample=True,
        top_p=1,
        temperature=1,
        min_length=None,
        use_cache=True,
        top_k=50,
        repetition_penalty=1,
        length_penalty=1,
    )
# e2e_inference_time = (time.perf_counter()-start)*1000
# print(f"the inference time is {e2e_inference_time} ms")
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)