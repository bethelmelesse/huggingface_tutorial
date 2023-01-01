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
# Transformer models only accept tensors as input

# step 1: Once we have the tokenizer, we can directly pass our sentences to it and we’ll get back a dictionary that’s ready to feed to our model!
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
 
# step 2: convert the list of input IDs to tensors.
 
raw_inputs = [
        "I've been waiting for a huggingFace course my whole life.",
        "I hate this so much"
    ]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)        # you will get a list of list as a result
print()

# input_ids - contains two rows of integers (one for each sentence) that are the unique identifiers of the tokens in each sentence
# atttention_mask - explained later


''' MODEL '''
model = AutoModel.from_pretrained(checkpoint)
# This architecture contains only the base Transformer module: given some inputs, it outputs what we’ll call hidden states, also known as features. 
# For each model input, we’ll retrieve a high-dimensional vector representing the contextual understanding of that input by the Transformer model.
# While these hidden states can be useful on their own, they’re usually inputs to another part of the model, known as the head.

''' high-dimensional vector? '''
# The vector output by the Transformer module is usually large. It generally has three dimensions:
# 1- Batch size: The number of sequences processed at a time (2 in our example).
# 2- Sequence Length: The length of the numerical representation of the sequence (16 in our example).
# 3- Hidden size:  The vector dimension of each model input.
# It is said to be “high dimensional” because of the last value. The hidden size can be very large (768 is common for smaller models, and in larger models this can reach 3072 or more).

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
print()

# Note that the outputs of 🤗 Transformers models behave like namedtuples or dictionaries. 
# You can access the elements by attributes (like we did) or by key (outputs["last_hidden_state"]), or even by index if you know exactly where the thing you are looking for is (outputs[0]).

''' MODEL HEADS'''
model_1 = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs_1 = model_1(**inputs)
print(outputs_1.logits.shape)   # Since we have just two sentences and two labels, the result we get from our model is of shape 2 x 2.

''' POSTPROCESSING THE OUTPUT'''
print(outputs_1.logits)   #  Those are not probabilities but logits, the raw, unnormalized scores outputted by the last layer of the model.

# To be converted to probabilities, they need to go through a SoftMax layer (all 🤗 Transformers models output the logits, as the loss function for training will generally fuse the last activation function, such as SoftMax, with the actual loss function, such as cross entropy)

predicitions = torch.nn.functional.softmax(outputs_1.logits, dim=-1)
print(predicitions)
# To get the labels corresponding to each position, we can inspect the id2label attribute of the model config 
pred_label = model.config.id2label
print(pred_label)