import os
hf_token = os.environ.get("HF_TOKEN")
print("Loaded HF token:", bool(hf_token))

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-es")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-es")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-es")

english_sentence = "I have a million dollars."

# Encode the English sentence
input_ids = tokenizer(english_sentence, return_tensors="pt").input_ids

# Generate the Spanish translation
output_ids = model.generate(input_ids)[0]

# Decode the generated output
spanish_translation = tokenizer.decode(output_ids, skip_special_tokens=True)

print(f"English: {english_sentence}")
print(f"Spanish: {spanish_translation}") #Last test made it at least this far

from comet import download_model, load_from_checkpoint

# Load a reference-free COMET model
model_path = download_model("Unbabel/wmt22-cometkiwi-da")
comet_model = load_from_checkpoint(model_path)

# Example data without reference translations
data = [
    {
        "src": "I have a million dollar.",
        "mt": "Tengo un millón de dólares.",
    },
    {
        "src": "The weather is nice today.",
        "mt": "El clima está agradable hoy.",
    }
]

model_output = comet_model.predict(data, batch_size=4, num_workers=4, gpus=1)
print (model_output)

# Import the Python SDK
import google.generativeai as genai#Last test made it at least this far
# Used to securely store your API key

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

genai.configure(api_key = GOOGLE_API_KEY)

print("Loaded GOOGLE API KEY", bool(GOOGLE_API_KEY))


#Now you can use the OpenAI API to perform translations. Here's an example:

if GOOGLE_API_KEY:
    try:
        # Initialize the Gemini API
        gemini_model = genai.GenerativeModel('gemini-flash-latest') # Using gemini-flash-latest

        response = gemini_model.generate_content("Translate the following English text to Spanish: I have a million dollar.")
        translated_text = response.text
        print(f"Original English: I have a million dollar.")
        print(f"Translated Spanish: {translated_text}")
    except Exception as e:
        print(f"An error occurred during translation: {e}")
else:
    print("Google API key not set. Cannot perform translation.")

# First, get a translation from the Helsinki-NLP model
english_sentence_to_evaluate = "This is a sentence to be evaluated."

# Ensure tokenizer and model for Helsinki-NLP are loaded (from cells ov35wJXaz0l4 or SqIzJTMeHRTk)
try:
    tokenizer # Check if tokenizer is defined
    model # Check if model is defined (Helsinki-NLP model)
except NameError:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-es")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-es")


input_ids_eval = tokenizer(english_sentence_to_evaluate, return_tensors="pt").input_ids
helsinki_translation = model.generate(input_ids_eval)[0]
helsinki_translation_text = tokenizer.decode(helsinki_translation, skip_special_tokens=True)

print(f"Original English: {english_sentence_to_evaluate}")
print(f"Helsinki-NLP Translation: {helsinki_translation_text}")

# Now, use the Gemini model to evaluate the Helsinki-NLP translation
if GOOGLE_API_KEY:
    try:
        # Ensure gemini_model is loaded (from cell c2fcd0cc)
        try:
            gemini_model
        except NameError:
             gemini_model = genai.GenerativeModel('gemini-flash-latest') # Or your chosen Gemini model

        evaluation_prompt = f"""Evaluate the following English to Spanish translation. Provide feedback on its accuracy, fluency, and grammar.

English Original: {english_sentence_to_evaluate}
Translation to Evaluate: {helsinki_translation_text}

Provide your evaluation and feedback."""

        gemini_evaluation_response = gemini_model.generate_content(evaluation_prompt)
        print("\nGemini Evaluation:")
        print(gemini_evaluation_response.text)

    except Exception as e:
        print(f"An error occurred during Gemini evaluation: {e}")
else:
    print("Google API key not set. Cannot perform Gemini evaluation.")

# Example of a poor translation
english_sentence_poor = "The book on the table it is red."
poor_spanish_translation = "El libro en la mesa es rojo. It is red." # Intentionally poor translation

print(f"Original English: {english_sentence_poor}")
print(f"Poor Spanish Translation: {poor_spanish_translation}")

# Use the Gemini model to evaluate the poor translation and suggest improvements
if GOOGLE_API_KEY:
    try:
        # Ensure gemini_model is loaded
        try:
            gemini_model
        except NameError:
             import google.generativeai as genai
             gemini_model = genai.GenerativeModel('gemini-flash-latest') # Or your chosen Gemini model

        evaluation_and_improvement_prompt = f"""Evaluate the following English to Spanish translation and provide specific feedback on its errors, fluency, and grammar. Then, suggest improvements to make it a correct and natural Spanish translation.

English Original: {english_sentence_poor}
Translation to Evaluate: {poor_spanish_translation}

Provide your evaluation, feedback, and suggested improvements."""

        gemini_response = gemini_model.generate_content(evaluation_and_improvement_prompt)
        print("\nGemini Evaluation and Improvements:")
        print(gemini_response.text)

    except Exception as e:
        print(f"An error occurred during Gemini evaluation: {e}")
else:
    print("Google API key not set. Cannot perform Gemini evaluation.")
