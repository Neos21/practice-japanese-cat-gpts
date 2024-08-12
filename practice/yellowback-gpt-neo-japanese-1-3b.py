import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
#from transformers import pipeline

# - https://huggingface.co/yellowback/gpt-neo-japanese-1.3B

# ワーニングが消えないので以下の最小コードは止めた
#generator = pipeline('text-generation', model='yellowback/gpt-neo-japanese-1.3B', device=0, clean_up_tokenization_spaces=False)
#output = generator('私が飼っている猫は今、', do_sample=True, max_length=50, num_return_sequences=1, truncation=True)
#print(output)

model_name = 'yellowback/gpt-neo-japanese-1.3B'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Start : Model Name [', model_name, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Tokenizer')  # ワーニングが出るから legacy と clean_up_tokenization_spaces を入れた・float16 は高速化のため
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
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
        max_length=100
    )

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Decode')
output_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '----------')
print(output_text)
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '==========')

# 2024-08-12 14:11:32.456212+09:00 Start : Model Name [ yellowback/gpt-neo-japanese-1.3B ]
# 2024-08-12 14:11:32.456268+09:00 Tokenizer
# 2024-08-12 14:11:32.811457+09:00 Model
# 2024-08-12 14:11:34.332567+09:00 Is Cuda Available?
# 2024-08-12 14:11:35.291572+09:00 Use Cuda
# 2024-08-12 14:11:37.858940+09:00 Prompt : [ 私が飼っている猫は今、 ]
# 2024-08-12 14:11:37.859011+09:00 Prompt Tokenizer
# 2024-08-12 14:11:37.860898+09:00 Model Generate
# 2024-08-12 14:11:41.463371+09:00 Decode
# 2024-08-12 14:11:41.463786+09:00 ----------
# 私が飼っている猫は今、円がないと眠れない家になります。料金がSNSで寒い冬には『猫バンバン』と言っていました。安くを冬場に動かすときは慎重にならないとだめですね。
# 小さいころに買ってもらった効果といえば指が透けて見えるような化繊の位が普通だったと思うのですが、日本に古くからある施術は木だの竹だの丈夫な素材で医療脱毛ができているため、観光用の大きな凧は
# 2024-08-12 14:11:41.463866+09:00 ==========

# 4.9GB
