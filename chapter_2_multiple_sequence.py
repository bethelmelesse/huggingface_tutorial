# In the previous exercise you saw how sequences get translated into lists of numbers. Let’s convert this list of numbers to a tensor and send it to the model:

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)    
# ['i', "'", 've', 'been', 'waiting', 'for', 'a', 'hugging', '##face', 'course', 'my', 'whole', 'life', '.']
ids = tokenizer.convert_tokens_to_ids(tokens) 
# [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]
input_ids = torch.tensor(ids)
# tensor([ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012])
# This line will fail.
# model(input_ids)
# The problem is that we sent a single sequence to the model, whereas 🤗 Transformers models expect multiple sentences by default.

#  the tokenizer didn’t just convert the list of input IDs into a tensor, it added a dimension on top of it:
tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])
print()
'''trying again'''
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])       # changed here
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

# Batching is the act of sending multiple sentences through the model, all at once. 
# If you only have one sentence, you can just build a batch with a single sequence:

batched_ids = [ids, ids]
# Batching allows the model to work when you feed it multiple sentences. Using multiple sequences is just as simple as building a batch with a single sequence.
# When you’re trying to batch together two (or more) sentences, they might be of different lengths. 
# If you’ve ever worked with tensors before, you know that they need to be of rectangular shape, so you won’t be able to convert the list of input IDs into a tensor directly. 
# To work around this problem, we usually pad the inputs.

'''Padding the inputs'''
batched_ids = [
    [200, 200, 200],
    [200, 200]
]
# we’ll use padding to make our tensors have a rectangular shape. 
# Padding makes sure all our sentences have the same length by adding a special word called the padding token to the sentences with fewer values.
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id]
]
# The padding token ID can be found in tokenizer.pad_token_id.

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)

# There’s something wrong with the logits in our batched predictions: the second row should be the same as the logits for the second sentence, but we’ve got completely different values!

# This is because the key feature of Transformer models is attention layers that contextualize each token. 
# These will take into account the padding tokens since they attend to all of the tokens of a sequence. 
# To get the same result when passing individual sentences of different lengths through the model or when passing a batch with the same sentences and padding applied, we need to tell those attention layers to ignore the padding tokens. 
# This is done by using an attention mask.

# Attention masks are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s: 1s indicate the corresponding tokens should be attended to, and 0s indicate the corresponding tokens should not be attended to (i.e., they should be ignored by the attention layers of the model).
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]
outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)

# With Transformer models, there is a limit to the lengths of the sequences we can pass the models. 
# Most models handle sequences of up to 512 or 1024 tokens, and will crash when asked to process longer sequences.
# Use a model with a longer supported sequence length.
# Truncate your sequences.
# sequence = sequence[:max_sequence_length] for trancating sequences