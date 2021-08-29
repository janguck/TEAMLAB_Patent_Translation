from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

english = "The system further generates the optimum energy pulse based on the determined amplitude and duration, and provides the optimum energy pulse to the heating element (34)."
korean = "시스템은 또한 결정된 진폭과 지속 시간을 기초로 하여 최적 에너지 펄스를 발생시키고, 가열 요소에 최적 에너지 펄스를 제공한다."



tokenizer.src_lang = "en_XX"
encoded_hi = tokenizer(korean, return_tensors="pt")
# generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"])
# result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


print(encoded_hi)
# print(result)
