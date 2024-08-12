import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# - https://huggingface.co/cyberagent/open-calm-3b
# - https://zenn.dev/fusic/articles/try-various-llms

model_name = 'cyberagent/open-calm-3b'
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

# 2024-08-12 12:08:41.900855+09:00 Start : Model Name [ cyberagent/open-calm-3b ]
# 2024-08-12 12:08:41.900912+09:00 Tokenizer
# 2024-08-12 12:08:42.558588+09:00 Model
# 2024-08-12 12:08:46.279977+09:00 Is Cuda Available?
# 2024-08-12 12:08:46.280055+09:00 Use Cuda
# 2024-08-12 12:08:46.283720+09:00 Prompt : [ 私が飼っている猫は今、 ]
# 2024-08-12 12:08:46.283783+09:00 Prompt Tokenizer
# 2024-08-12 12:08:46.285905+09:00 Model Generate
# 2024-08-12 12:08:49.134185+09:00 Decode
# 2024-08-12 12:08:49.134493+09:00 ----------
# 私が飼っている猫は今、生後3ヶ月半。 まだ乳歯が生えそろってなくて歯がムズムズしてるみたいだけど...
# 今朝は8時に起きて9時半から仕事をしたんだけどね...。 あー眠い( ́Д`)
# 2024-08-12 12:08:49.134561+09:00 ==========

# 5.4GB
