# We’ll use the AutoModel class, which is handy when you want to instantiate any model from a checkpoint.
# The AutoModel class and all of its relatives are actually simple wrappers over the wide variety of models available in the library. It’s a clever wrapper as it can automatically guess the appropriate model architecture for your checkpoint, and then instantiates a model with this architecture.
# However, if you know the type of model you want to use, you can use the class that defines its architecture directly

from transformers import BertConfig, BertModel
import torch

print()
'''Creating Transformer'''
# The first thing we’ll need to do to initialize a BERT model is load a configuration object
# building the config
config = BertConfig()

# builing the model from the config
model_1 = BertModel(config)     # mdoel is randomly initialised
# the hidden_size attribute defines the size of the hidden_states vector, and 
# num_hidden_layers defines the number of layers the Transformer model has
print(config)

'''Different loading methods'''
# The model can be used in this state, but it will output gibberish; it needs to be trained first. 
model = BertModel.from_pretrained("bert-base-cased")
# In the code sample above we didn’t use BertConfig, and instead loaded a pretrained model via the bert-base-cased identifier.
# This model is now initialized with all the weights of the checkpoint. It can be used directly for inference on the tasks it was trained on, and it can also be fine-tuned on a new task. 
# The weights have been downloaded and cached (so future calls to the from_pretrained() method won’t re-download them) in the cache folder, which defaults to ~/.cache/huggingface/transformers. 
# You can customize your cache folder by setting the HF_HOME environment variable.

''' Saving Methods'''
model.save_pretrained("directory_on_my_computer")
# This saves two files to your disk:
# ls directory_on_my_computer
# config.json pytorch_model.bin
# If you take a look at the config.json file, you’ll recognize the attributes necessary to build the model architecture. 
# The pytorch_model.bin file is known as the state dictionary; it contains all your model’s weights.

'''Using a Transformer model for inference'''
sequences = ["Hello!", "Cool.", "Nice!"]
# The tokenizer converts these to vocabulary indices which are typically called input IDs. Each sequence is now a list of numbers!
# The resulting output is:

encoded_sequences = [                 # This is a list of encoded sequences: a list of lists. 
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

model_inputs = torch.tensor(encoded_sequences)
output = model(model_inputs)

print()
