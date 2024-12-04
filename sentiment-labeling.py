from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

df = pd.read_csv("voz_complete_data.csv")
print(df.head())  
check_point = "mr4/phobert-base-vi-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(check_point)
model = AutoModelForSequenceClassification.from_pretrained(check_point)

model