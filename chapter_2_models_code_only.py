from transformers import BertConfig, BertModel
import torch

print()

config = BertConfig()
model_1 = BertModel(config)     

print(config)

'''Different loading methods'''
model = BertModel.from_pretrained("bert-base-cased")

''' Saving Methods'''
model.save_pretrained("directory_on_my_computer")

'''Using a Transformer model for inference'''
sequences = ["Hello!", "Cool.", "Nice!"]
encoded_sequences = [                 
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

model_inputs = torch.tensor(encoded_sequences)
output = model(model_inputs)

print()
