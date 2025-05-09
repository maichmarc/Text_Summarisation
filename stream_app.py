import streamlit as st
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from src.pipeline.prediction import PredictionPipeline

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained('maichmarc/textS')
    model = AutoModelForSeq2SeqLM.from_pretrained("maichmarc/textS")
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Summarize')
button = st.button("Summarise")


if user_input and button :
 
    obj = PredictionPipeline()
    text = obj.predict(user_input)

    st.write("Summary: ", text)