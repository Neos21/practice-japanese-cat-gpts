import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# - https://huggingface.co/cyberagent/open-calm-7b
# - https://zenn.dev/fusic/articles/try-various-llms

model_name = 'cyberagent/open-calm-7b'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Start : Model Name [', model_name, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Tokenizer')
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)  # ワーニングが出るから clean_up_tokenization_spaces を入れた
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Model')
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)

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
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id
    )

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Decode')
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '----------')
print(output_text)
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '==========')

# 2024-08-12 12:22:10.957396+09:00 Start : Model Name [ cyberagent/open-calm-7b ]
# 2024-08-12 12:22:10.957468+09:00 Tokenizer
# 2024-08-12 12:22:11.333186+09:00 Model
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.13it/s]
# Some parameters are on the meta device device because they were offloaded to the cpu.
# 2024-08-12 12:22:15.217656+09:00 Is Cuda Available?
# 2024-08-12 12:22:15.217712+09:00 Use Cuda
# You shouldn't move a model that is dispatched using accelerate hooks.
# Traceback (most recent call last):
#   File "./practice/cyberagent-open-calm-7b.py", line 20, in <module>
#     model = model.to('cuda')
#   File "/home/neo/.cache/pypoetry/virtualenvs/practice-4D3MFvjB-py3.8/lib/python3.8/site-packages/accelerate/big_modeling.py", line 456, in wrapper
#     raise RuntimeError("You can't move a model that has some modules offloaded to cpu or disk.")
# RuntimeError: You can't move a model that has some modules offloaded to cpu or disk.

# 13GB
