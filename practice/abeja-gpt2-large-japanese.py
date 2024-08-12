import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# - https://huggingface.co/abeja/gpt2-large-japanese

model_name = 'abeja/gpt2-large-japanese'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Start : Model Name [', model_name, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Tokenizer')
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, clean_up_tokenization_spaces=True)  # ワーニングが出るから legacy と clean_up_tokenization_spaces を入れた
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
        max_new_tokens=256,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id
    )

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Decode')
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '----------')
print(output_text)
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '==========')

# 2024-08-12 13:34:43.928899+09:00 Start : Model Name [ abeja/gpt2-large-japanese ]
# 2024-08-12 13:34:43.928962+09:00 Tokenizer
# 2024-08-12 13:34:44.687209+09:00 Model
# 2024-08-12 13:34:46.250134+09:00 Is Cuda Available?
# 2024-08-12 13:34:47.149705+09:00 Use Cuda
# 2024-08-12 13:34:49.213801+09:00 Prompt : [ 私が飼っている猫は今、 ]
# 2024-08-12 13:34:49.213875+09:00 Prompt Tokenizer
# 2024-08-12 13:34:49.215528+09:00 Model Generate
# 2024-08-12 13:34:58.342287+09:00 Decode
# 2024-08-12 13:34:58.342989+09:00 ----------
# 私が飼っている猫は今、猫風邪という病気になっていて 口の中がかなり痛いらしく口臭もあるようで 辛そうに 咳を繰り返しているのですが... 猫風邪で口臭?? 口が痛いのは辛いですね。 私は猫を飼っていないので知りませんでした。 私も昔は猫を飼っていましたが、 猫風邪にかかったことがありませんでしたので。 犬は猫風邪にかかりますか? (>_<) 犬の口の中を掃除してあげた方が良いのでしょうか? こんにちは。 猫風邪?ってどんな病気なのですか? (>_<) また、猫風邪にかかった場合、 どのような処置をすると、 良いのでしょうか? よろしくお願いします。(_ _)m 猫の口臭と嘔吐について。 猫を飼っていて猫風邪を引いてしまいました。 猫は、1匹で生活してるのですが、口臭がかなりきついようで1日1回口を洗っていたのですが、臭い口の中を見たとき、 気持ち悪そうで吐きそうになっていました。(; ́д`)... 口臭
# 2024-08-12 13:34:58.343099+09:00 ==========

# 2.9GB
