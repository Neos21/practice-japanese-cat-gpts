import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# - https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo
# - https://zenn.dev/fusic/articles/try-various-llms

model_name = 'rinna/japanese-gpt-neox-3.6b-instruction-ppo'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Start : Model Name [', model_name, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Tokenizer')  # ワーニングが出るから legacy と clean_up_tokenization_spaces を入れた・float16 は高速化のため
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False, clean_up_tokenization_spaces=True, torch_dtype=torch.float16)
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Model')
model = AutoModelForCausalLM.from_pretrained(model_name)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Is Cuda Available?')
if torch.cuda.is_available():
    print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Use Cuda')
    model = model.to('cuda')

input_text = '私が飼っている猫は今、'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Prompt : [', input_text, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Prompt Tokenizer')
inputs = tokenizer(input_text, add_special_tokens=False, return_tensors='pt').to(model.device)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Model Generate')
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=256,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Decode')
output_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '----------')
print(output_text)
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '==========')

# 2024-08-12 12:49:29.319469+09:00 Start : Model Name [ rinna/japanese-gpt-neox-3.6b-instruction-ppo ]
# 2024-08-12 12:49:29.319530+09:00 Tokenizer
# 2024-08-12 12:49:29.865723+09:00 Model
# 2024-08-12 12:49:50.775352+09:00 Is Cuda Available?
# 2024-08-12 12:49:55.881196+09:00 Use Cuda
# 2024-08-12 12:50:26.745551+09:00 Prompt : [ 私が飼っている猫は今、 ]
# 2024-08-12 12:50:26.745645+09:00 Prompt Tokenizer
# 2024-08-12 12:50:26.975398+09:00 Model Generate
# 2024-08-12 12:52:23.952208+09:00 Decode
# 2024-08-12 12:52:23.980473+09:00 ----------
# 私が飼っている猫は今、避妊手術を受けています。手術後、猫の尿臭が増しました。これは再発性の膀胱炎の症状です。原因としては、手術後のストレスや抗生物質の使用が考えられます。また、猫を清潔に保つように心がけ、トイレを頻繁に洗うことで再発を防止できます。
# 2024-08-12 12:52:23.980540+09:00 ==========

# 6.9GB
