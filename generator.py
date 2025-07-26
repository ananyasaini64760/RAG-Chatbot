from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_answer(query, context_list):
    context = " ".join(context_list)
    input_text = f"Context: {context}\n\nQuestion: {query}"
    result = generator(input_text, max_length=256, do_sample=True)
    return result[0]['generated_text']
