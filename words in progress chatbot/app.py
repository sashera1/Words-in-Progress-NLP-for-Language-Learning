import streamlit as st
import requests
import os
from dotenv import load_dotenv
import re

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Set it in your environment or .env file.")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

st.set_page_config(page_title="Words in Progress", page_icon="📝")


def extract_sentence(text: str) -> str:
    # first try to extract from quotes
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return quoted[-1].strip()

    # otherwise use last line
    lines = text.strip().split("\n")
    return lines[-1].strip()


# session state defaults
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Beginner"


def call_gemini(prompt: str) -> str:
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


# HF: load Helsinki model
@st.cache_resource
def load_helsinki_model(model_name="Helsinki-NLP/opus-mt-tc-big-en-es"):
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except Exception as e:
        raise RuntimeError("Install transformers and torch.") from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 0 if (torch.cuda.is_available()) else -1
    return tokenizer, model, device

def translate_with_helsinki(text: str):
    tokenizer, model, device = load_helsinki_model()
    import torch

    inputs = tokenizer(text, return_tensors="pt")
    if device != -1:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=256)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


# COMET model
@st.cache_resource
def load_comet_model():
    try:
        from comet.comet import download_model, load_from_checkpoint
    except Exception:
        from comet import download_model, load_from_checkpoint

    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)
    return comet_model

def score_with_comet(src: str, mt: str, ref: str):
    try:
        comet_model = load_comet_model()
    except Exception as e:
        st.error(f"COMET load error: {e}")
        return None

    data = [{"src": src, "mt": mt, "ref": ref}]
    try:
        comet_output = comet_model.predict(data, batch_size=8, gpus=1)
        comet_score = float(comet_output.system_score)
        return comet_score
    except Exception as e:
        print("COMET prediction error:", e)
        return None


# ui: sidebar
st.sidebar.header("⚙️ Settings")
difficulty = st.sidebar.selectbox("Difficulty level:", ["Beginner", "Intermediate", "Advanced"])
st.session_state.difficulty = difficulty

# sentence generation
if st.sidebar.button("✨ New Sentence"):
    prompt = (
        f"Give me one {difficulty.lower()}-level English sentence for a student to translate "
        "into Spanish. Keep it short, simple, and enclosed in quotes."
    )

    raw_gemini_response = call_gemini(prompt)
    clean_sentence = extract_sentence(raw_gemini_response)

    # store clean sentence only
    st.session_state.current_prompt = clean_sentence

    st.session_state.chat_history.append(
        ("Gemini", f"Translate this sentence:\n\n**{clean_sentence}**")
    )
    st.rerun()

st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox("🔧 Developer Debug Mode", value=False)
st.session_state.debug = debug_mode


# title + chat history
st.title("📝 Words in Progress")
st.caption("Practice translating sentences to Spanish with guided feedback.")

for role, msg in st.session_state.chat_history:
    if role == "You":
        with st.chat_message("user"):
            st.write(msg)
    elif role == "Gemini":
        with st.chat_message("assistant"):
            st.write(msg)
    elif role == "Debug" and st.session_state.debug:
        with st.expander("🔧 Debug Info", expanded=False):
            st.markdown(msg)


# user input (Spanish translation)
user_input = st.chat_input("Enter your Spanish translation of the sentence, or ask for help…")

if user_input:
    st.session_state.chat_history.append(("You", user_input))
    user_translation = user_input.strip()

    if st.session_state.current_prompt:
        english_original = st.session_state.current_prompt

        # 1) Helsinki reference translation
        try:
            helsinki_ref = translate_with_helsinki(english_original)
        except Exception as e:
            helsinki_ref = None
            st.session_state.chat_history.append(("Gemini", f"⚠️ Helsinki error: {e}"))

        # 2) COMET scoring
        comet_score = None
        if helsinki_ref:
            comet_score = score_with_comet(
                src=english_original,
                mt=user_translation,
                ref=helsinki_ref
            )

        # 3) feedback prompt to Gemini
        gemini_eval_prompt = (
            "You are a friendly but detailed Spanish tutor.\n\n"
            f"English original: {english_original}\n"
            f"User translation: {user_translation}\n"
        )

        if helsinki_ref:
            gemini_eval_prompt += f"Reference (hidden): {helsinki_ref}\n"
        if comet_score is not None:
            gemini_eval_prompt += f"COMET quality score (0-1): {comet_score:.4f}\n\n"

        gemini_eval_prompt += (
            "Please:\n"
            "- Evaluate the user's translation for accuracy and fluency.\n"
            "- Provide a corrected version if needed.\n"
            "- Explain mistakes simply and clearly.\n"
            "- Provide one short improvement tip.\n"
        )

        ai_response = call_gemini(gemini_eval_prompt)
        st.session_state.chat_history.append(("Gemini", ai_response))

        # debug info
        if st.session_state.debug:
            debug_text = (
                "### 🔍 Developer Debug\n"
                f"- **English source:** {english_original}\n"
                f"- **User translation:** {user_translation}\n"
                f"- **Helsinki reference:** {helsinki_ref}\n"
                f"- **COMET score:** {comet_score}\n"
            )
            st.session_state.chat_history.append(("Debug", debug_text))

    else:
        # no sentence active — regular tutor chat
        open_prompt = (
            f"You are a helpful Spanish tutor. The student says: {user_input}\n"
            "Please reply helpfully."
        )
        ai_response = call_gemini(open_prompt)
        st.session_state.chat_history.append(("Gemini", ai_response))

    st.rerun()
