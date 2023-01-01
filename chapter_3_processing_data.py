import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
print()

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!"
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

batch["labels"] = torch.tensor([1,1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()

# (Microsoft Research Paraphrase Corpus
# 5,801 pairs of sentences, with a label indicating if they are paraphrases or not 

'''Loading the dataset from the Hub'''
#  GLUE benchmark, which is an academic benchmark that is used to measure the performance of ML models across 10 different text classification tasks.


raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
print()
# This command downloads and caches the dataset, by default in ~/.cache/huggingface/datasets. 

# We can access each pair of sentences in our raw_datasets object by indexing
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
print()

# We can see the labels are already integers, so we won’t have to do any preprocessing there. 
# To know which integer corresponds to which label, we can inspect the features of our raw_train_dataset. 
print(raw_train_dataset.features)
print()

# Behind the scenes, label is of type ClassLabel, and the mapping of integers to label name is stored in the names folder. 
# 0 corresponds to not_equivalent, and 1 corresponds to equivalent.

'''Preprocessing a Dataset'''
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

# print(tokenized_sentences_1)
# print(tokenized_sentences_2)
# print()

inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)
print()

decode = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
print(decode)
print()

tokenized_dataset = tokenizer(
    raw_datasets["train"]['sentence1'],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True
)
# This works well, but it has the disadvantage of returning a dictionary (with our keys, input_ids, attention_mask, and token_type_ids, and values that are lists of lists).
# To keep the data as a dataset, we will use the Dataset.map() method. 
# This also allows us some extra flexibility, if we need more preprocessing done than just tokenization. 

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

#  This function takes a dictionary (like the items of our dataset) and returns a new dictionary with the keys input_ids, attention_mask, and token_type_ids.
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)
print()

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
samples = tokenized_datasets["train"][:8]
samples = {k:v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]
print()

batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}