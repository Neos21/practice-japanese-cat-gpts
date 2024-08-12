import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# - https://huggingface.co/cyberagent/open-calm-7b
# - https://zenn.dev/fusic/articles/try-various-llms

model_name = 'cyberagent/open-calm-1b'
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

# 2024-08-12 12:02:33.347169+09:00 Start : Model Name [ cyberagent/open-calm-1b ]
# 2024-08-12 12:02:33.347229+09:00 Tokenizer
# 2024-08-12 12:02:33.753460+09:00 Model
# 2024-08-12 12:02:36.168059+09:00 Is Cuda Available?
# 2024-08-12 12:02:36.168134+09:00 Use Cuda
# 2024-08-12 12:02:36.170568+09:00 Prompt : [ 私が飼っている猫は今、 ]
# 2024-08-12 12:02:36.170604+09:00 Prompt Tokenizer
# 2024-08-12 12:02:36.172090+09:00 Model Generate
# 2024-08-12 12:02:37.648805+09:00 Decode
# 2024-08-12 12:02:37.649086+09:00 ----------
# 私が飼っている猫は今、オスで16歳。
# そしてもう1匹の猫はメスで8歳と高齢ですが2人とも病気知らず!
# 2024-08-12 12:02:37.649150+09:00 ==========

# 2.8GB
