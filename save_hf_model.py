# %% import required libraries
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

model_path = "models/transformers/"  # Created automatically if does not exists

# Download and save the model to local directory
model_name = "bert-base-multilingual-uncased-sentiment"

model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
classifier.save_pretrained(model_path)

# Load model from local directory if it works
model = TFAutoModelForSequenceClassification.from_pretrained(
    model_path, local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

classifier(["good"])
