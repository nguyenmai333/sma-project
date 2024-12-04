from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

df = pd.read_csv("voz_complete_data.csv")
print(df.head())  
check_point = "mr4/phobert-base-vi-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(check_point)
model = AutoModelForSequenceClassification.from_pretrained(check_point)


print(model)
target = df['Reply_Content'].tolist()[:10]
inputs = tokenizer(target, padding=True,
                   truncation=True, return_tensors="pt")

outputs = model(**inputs)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)


print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
for i, prediction in enumerate(predictions):
    print(target[i])
    for j, value in enumerate(prediction):
        print(
            "    " + model.config.id2label[j] + ": " + str(value.item()))
print("<<<<<<<<<<<<<<<<<<<<<<<<<<")