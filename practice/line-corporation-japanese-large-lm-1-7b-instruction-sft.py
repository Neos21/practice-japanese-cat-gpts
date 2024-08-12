import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# - https://huggingface.co/line-corporation/japanese-large-lm-1.7b-instruction-sft
# - https://github.com/line/japanese-large-lm-instruction-sft
# - https://engineering.linecorp.com/ja/blog/3.6b-japanese-language-model-with-improved-dialog-performance-by-instruction-tuning

model_name = 'line-corporation/japanese-large-lm-1.7b-instruction-sft'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Start : Model Name [', model_name, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Tokenizer')
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)  # ワーニングが出るので legacy を入れた
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Model')
model = AutoModelForCausalLM.from_pretrained(model_name)
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Pipeline Generator')
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

input_text = '私が飼っている猫は今、'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Prompt : [', input_text, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Generator')
output_text = generator(
    input_text,
    max_length = 256,
    do_sample = True,
    temperature = 0.7,
    top_p = 0.9,
    top_k = 0,
    repetition_penalty = 1.1,
    num_beams = 1,
    pad_token_id = tokenizer.pad_token_id,
    num_return_sequences = 1,
    truncation = True  # ワーニングが出るから追加した
)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '----------')
print(output_text[0].get('generated_text'))
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '==========')

# 2024-08-12 13:14:39.647355+09:00 Start : Model Name [ line-corporation/japanese-large-lm-1.7b-instruction-sft ]
# 2024-08-12 13:14:39.647412+09:00 Tokenizer
# 2024-08-12 13:14:40.282839+09:00 Model
# 2024-08-12 13:14:42.713497+09:00 Pipeline Generator
# 2024-08-12 13:14:46.304447+09:00 Prompt : [ 私が飼っている猫は今、 ]
# 2024-08-12 13:14:46.304503+09:00 Generator
# 2024-08-12 13:14:46.963595+09:00 ----------
# 私が飼っている猫は今、
# 2024-08-12 13:14:46.963672+09:00 ==========

# 3.1GB
