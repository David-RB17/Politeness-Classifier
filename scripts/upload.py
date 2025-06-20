from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = r"G:\Python Projects\politeness-classifier-jp\models\bert-finetunedv2"
HF_REPO = "japanese-formality-classifier"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)