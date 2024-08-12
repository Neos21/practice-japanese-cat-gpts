import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# - https://huggingface.co/rinna/japanese-gpt2-xsmall

model_name = 'rinna/japanese-gpt2-xsmall'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Start : Model Name [', model_name, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Tokenizer')
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False, clean_up_tokenization_spaces=True)
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Model')
model = AutoModelForCausalLM.from_pretrained(model_name)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Is Cuda Available?')
if torch.cuda.is_available():
    print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Use Cuda')
    model = model.to('cuda')

input_text = '私が飼っている猫は今、'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Prompt : [', input_text, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Prompt Tokenizer')
inputs = tokenizer(input_text, return_tensors='pt').to(model.device)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Model Generate')
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=100,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Decode')
output_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '----------')
print(output_text)
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '==========')

# 2024-08-12 14:17:52.468453+09:00 Start : Model Name [ rinna/japanese-gpt2-xsmall ]
# 2024-08-12 14:17:52.468516+09:00 Tokenizer
# 2024-08-12 14:17:53.121280+09:00 Model
# 2024-08-12 14:17:54.234062+09:00 Is Cuda Available?
# 2024-08-12 14:17:55.169652+09:00 Use Cuda
# 2024-08-12 14:17:56.626346+09:00 Prompt : [ 私が飼っている猫は今、 ]
# 2024-08-12 14:17:56.626406+09:00 Prompt Tokenizer
# 2024-08-12 14:17:56.626933+09:00 Model Generate
# 2024-08-12 14:17:57.411109+09:00 Decode
# 2024-08-12 14:17:57.411994+09:00 ----------
# 私が飼っている猫は今、さんのブログで 何十万もかけて作りましたので
# 2024-08-12 14:17:57.412053+09:00 ==========

# 150MB
