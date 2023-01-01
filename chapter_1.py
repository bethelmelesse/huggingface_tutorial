from transformers import pipeline

print()
classifier = pipeline("sentiment-analysis")
example_1 = classifier("I've been waiting for a huggingface course my whole life.")
example_2 = classifier(["I've been waiting for a huggingface course my whole life.", "I hate this so much"])
print(example_1)
print(example_2)
print()

zero_shot = pipeline("zero-shot-classification")
example_3 = zero_shot("This is a course about the Transformers library.", candidate_labels=["education", "politics", "business"]) 
print(example_3) 
print()

generator = pipeline("text-generation")
example_4 = generator("In this course, we will teach you how to")
example_5 = generator("In this course, we will teach you how to", num_return_sequences=2, max_length=15)
print(example_4) 
print(example_5) 
print()

generator_2 = pipeline("text-generation", model="distilgpt2")
example_6 = generator("In this course, we will teach you how to", max_length=15, num_return_sequences=2)
print(example_6) 
print()

unmasker = pipeline("fill-mask")
example_7 = unmasker("In this course, we will teach you all about <mask> models.", top_k=2)
print(example_7) 
print()

ner = pipeline("ner", grouped_entities=True)
example_8 = ner("My name is Sylvain and I work at Hugging Face in Brooklyn")
print(example_8) 
print()

q_a = pipeline("question-answering")
example_9= q_a(
    question="where do i work",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(example_9) 
print()

summeraizer = pipeline("summarization")
example_10 = summeraizer("""
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
""")
print(example_10) 
print()

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
example_11 = translator("Ce cours est produit par Hugging Face.")
print(example_11) 
print()