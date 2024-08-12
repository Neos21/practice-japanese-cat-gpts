import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# - Python v3.8 以降というのは Poetry にお任せ
# - import してるので必須で入れた (poetry add でバージョン番号はお任せ
#   - torch = "^2.4.0"
#   - transformers = "^4.44.0"
# - Cuda 利用のためにいるらしいので入れた
#   - accelerate = "^0.33.0"
# - ないと実行時に怒られたので入れた
#   - sentencepiece = "^0.2.0"
#   - protobuf = "^5.27.3"
# - https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2
# - https://zenn.dev/fusic/articles/try-various-llms

model_name = 'rinna/japanese-gpt-neox-3.6b-instruction-sft-v2'
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

# 2024-08-12 11:53:53.661991+09:00 Start : Model Name [ rinna/japanese-gpt-neox-3.6b-instruction-sft-v2 ]
# 2024-08-12 11:53:53.662073+09:00 Tokenizer
# 2024-08-12 11:53:54.189648+09:00 Model
# 2024-08-12 11:54:19.038057+09:00 Is Cuda Available?
# 2024-08-12 11:54:25.664252+09:00 Use Cuda
# 2024-08-12 11:55:01.049738+09:00 Prompt : [ 私が飼っている猫は今、 ]
# 2024-08-12 11:55:01.049828+09:00 Prompt Tokenizer
# 2024-08-12 11:55:01.199220+09:00 Model Generate
# 2024-08-12 11:57:12.624825+09:00 Decode
# 2024-08-12 11:57:12.653721+09:00 ----------
# 私が飼っている猫は今、4歳です。最近、彼女がよく咳をするようになったので、獣医さんに連れて行きました。猫を診察してもらったところ、彼女は鼻水がたくさん出ているとのことでした。医師によると、彼女の鼻の中にポリープがあり、それが彼女の咳を引き起こしているとのことです。そこで、ポリープを取り除く手術を行うことにしました。私ができる限りお手伝いしようと思います!
# 2024-08-12 11:57:12.653808+09:00 ==========

# 6.9GB
