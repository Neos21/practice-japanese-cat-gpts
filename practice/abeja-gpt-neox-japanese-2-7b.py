import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# - https://huggingface.co/abeja/gpt-neox-japanese-2.7b

model_name = 'abeja/gpt-neox-japanese-2.7b'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Start : Model Name [', model_name, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Tokenizer')
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)  # ワーニングが出るから clean_up_tokenization_spaces を入れた
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Model')
model = AutoModelForCausalLM.from_pretrained(model_name)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Is Cuda Available?')
if torch.cuda.is_available():
    print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Use Cuda')
    model = model.to('cuda')

input_text = '私が飼っている猫は今、'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Prompt : [', input_text, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Prompt Tokenizer')
inputs = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Model Generate')
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=100,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Decode')
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '----------')
print(output_text)
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '==========')

# 2024-08-12 13:55:34.412659+09:00 Start : Model Name [ abeja/gpt-neox-japanese-2.7b ]
# 2024-08-12 13:55:34.412749+09:00 Tokenizer
# 2024-08-12 13:55:34.801928+09:00 Model
# 2024-08-12 13:55:42.625168+09:00 Is Cuda Available?
# 2024-08-12 13:55:47.537523+09:00 Use Cuda
# 2024-08-12 13:56:05.431556+09:00 Prompt : [ 私が飼っている猫は今、 ]
# 2024-08-12 13:56:05.431854+09:00 Prompt Tokenizer
# 2024-08-12 13:56:05.522031+09:00 Model Generate
# The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# 2024-08-12 13:56:27.125210+09:00 Decode
# 2024-08-12 13:56:27.156094+09:00 ----------
# 私が飼っている猫は今、6歳になったんですが、やっぱり若い猫は成長も早いですし、元気もいいですよ！
# 2024-08-12 13:56:27.156176+09:00 ==========

# 5.0GB
