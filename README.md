# Words-in-Progress-NLP-for-Language-Learning
Final Project for CS 4774 - Natural Language Processing. We apply natural language processing (NLP) techniques to support language learning by providing feedback on student translations.

## To run the Streamlit chatbot: 
- Create an .env file with a valid Google API token
- Run `pip install streamlit` & `pip install streamlit requests python-dotenv`
- Configure a valid $HF_token in the terminal by running `huggingface-cli login --token $HF_TOKEN`
- Run `streamlit run app.py`
