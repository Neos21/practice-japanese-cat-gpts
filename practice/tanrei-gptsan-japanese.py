import datetime

from transformers import AutoModel, AutoTokenizer, trainer_utils

# - https://huggingface.co/Tanrei/GPTSAN-japanese
# - https://github.com/tanreinama/GPTSAN

device = 'cuda'
model_name = 'Tanrei/GPTSAN-japanese'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Start : Model Name [', model_name, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Tokenizer')
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)  # ワーニング解消のため clean_up_tokenization_spaces を入れた
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Model')
model = AutoModel.from_pretrained(model_name).to(device)

input_text = '私が飼っている猫は今、'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Prompt : [', input_text, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Prompt Tokenizer')
x_token = tokenizer(input_text, return_tensors='pt')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Model Generate')
trainer_utils.set_seed(30)
input_ids = x_token.input_ids.to(device)
gen_token = model.generate(input_ids, max_new_tokens=50)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Decode')
output_text = tokenizer.decode(gen_token[0])

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '----------')
print(output_text)
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '==========')

# 2024-08-12 11:52:59.572467+09:00 Start : Model Name [ Tanrei/GPTSAN-japanese ]
# 2024-08-12 11:52:59.572579+09:00 Tokenizer
# 2024-08-12 11:53:00.098307+09:00 Model
# 2024-08-12 11:53:14.838555+09:00 Prompt : [ 私が飼っている猫は今、 ]
# 2024-08-12 11:53:14.841870+09:00 Prompt Tokenizer
# 2024-08-12 11:53:14.883515+09:00 Model Generate
# 2024-08-12 11:53:23.330516+09:00 Decode
# 2024-08-12 11:53:23.338153+09:00 ----------
# 私が飼っている猫は今、猫IIS10号を毎日のように追い掛け回しており、私も追い掛け回す犬の気持ちが分かるからこそ、毎日彼の言葉が頭に浮かんできたりするのだが
# 2024-08-12 11:53:23.338269+09:00 ==========

# 5.2GB
