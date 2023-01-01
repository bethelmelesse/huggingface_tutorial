from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

print()
classifier = pipeline("sentiment-analysis")  # default checkpoint for this pipline is distilbert-base-uncased-finetuned-sst-2-english 
example_1 = classifier(
    [
        "I've been waiting for a huggingFace course my whole life.",
        "I hate this so much"
    ]
)
print(example_1) 
print()

''' PREPROCESSING '''
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
 
raw_inputs = [
        "I've been waiting for a huggingFace course my whole life.",
        "I hate this so much"
    ]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)     
print()

''' MODEL '''
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

outputs = model(**inputs)

print(outputs.logits.shape)  
print(outputs.logits)

predicitions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predicitions)

pred_label = model.config.id2label
print(pred_label)