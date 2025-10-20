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
