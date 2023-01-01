#  tokenizers serve one purpose: 
# to translate text into data that can be processed by the model. 
# Models can only process numbers, so tokenizers need to convert our text inputs to numerical data.
# In NLP tasks, the data that is generally processed is raw text.
# The goal is to find the most meaningful representation — that is, the one that makes the most sense to the model — and, if possible, the smallest representation.
from transformers import BertTokenizer, AutoTokenizer

'''Word-based'''
#  the goal is to split the raw text into words and find a numerical representation for each of them
# step 1: split 
# There are different ways to split the text. 
# For example, we could could use whitespace to tokenize the text into words by applying Python’s split() function:
print()
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)
print()

# step 2: build vocabulary
# With this kind of tokenizer, we can end up with some pretty large “vocabularies,” where a vocabulary is defined by the total number of independent tokens that we have in our corpus.
# Each word gets assigned an ID, starting from 0 and going up to the size of the vocabulary. 
# The model uses these IDs to identify each word.

# step 3
# Finally, we need a custom token to represent words that are not in our vocabulary. 
# This is known as the “unknown” token, often represented as ”[UNK]” or ””. 
# It’s generally a bad sign if you see that the tokenizer is producing a lot of these tokens, as it wasn’t able to retrieve a sensible representation of a word and you’re losing information along the way. 
# The goal when crafting the vocabulary is to do it in such a way that the tokenizer tokenizes as few words as possible into the unknown token.


'''Character-based'''
# Character-based tokenizers split the text into characters, rather than words. 
# This has two primary benefits:
# 1 - The vocabulary is much smaller.
# 2 - There are much fewer out-of-vocabulary (unknown) tokens, since every word can be built from characters.
# Since the representation is now based on characters rather than words, one could argue that, intuitively, it’s less meaningful: each character doesn’t mean a lot on its own, whereas that is the case with words. 
# also we’ll end up with a very large amount of tokens to be processed by our model: whereas a word would only be a single token with a word-based tokenizer, it can easily turn into 10 or more tokens when converted into characters.

'''Subword tokenisation'''
# Subword tokenization algorithms rely on the principle that frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords.

'''Loading and Saving'''
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Similar to AutoModel, the AutoTokenizer class will grab the proper tokenizer class in the library based on the checkpoint name, and can be used directly with any checkpoint:
tokenizer_1 = AutoTokenizer.from_pretrained("bert-base-cased")

input = tokenizer("Using a Transformer network is simple")
input_1 = tokenizer_1("Using a Transformer network is simple")
print(input)
print()
print(input_1)
print()

tokenizer.save_pretrained("directory_on_my_computer")
tokenizer_1.save_pretrained("directory_on_my_computer")

# First, let’s see how the input_ids are generated. 
# To do this, we’ll need to look at the intermediate methods of the tokenizer.
'''Encoding'''
# Translating text to numbers is known as encoding. 
# Encoding is done in a two-step process: the tokenization, followed by the conversion to input IDs.

#  the first step is to split the text into words (or parts of words, punctuation symbols, etc.), usually called tokens. There are multiple rules that can govern that process, which is why we need to instantiate the tokenizer using the name of the model, to make sure we use the same rules that were used when the model was pretrained.

# The second step is to convert those tokens into numbers, so we can build a tensor out of them and feed them to the model. To do this, the tokenizer has a vocabulary, which is the part we download when we instantiate it with the from_pretrained() method. Again, we need to use the same vocabulary used when the model was pretrained.

'''Tokenization'''
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)    # ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

print(tokens)
print()

'''From tokens to input IDs'''
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
print()


'''Decoding'''
# Decoding is going the other way around: from vocabulary indices, we want to get a string. 
# This can be done with the decode() method as follows:
decoding_string = tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])
print(decoding_string)

# Note that the decode method not only converts the indices back to tokens, but also groups together the tokens that were part of the same words to produce a readable sentence.