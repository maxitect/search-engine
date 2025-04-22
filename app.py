import os

import requests
import streamlit as st
import wandb
from huggingface_hub._login import _login
from pydantic import BaseModel, Field

from eval_hn import get_model_features, setup

with open("/run/secrets/HF_TOKEN") as f:
    HF_TOKEN = f.read().strip()

with open("/run/secrets/WANDB_API_KEY") as f:
    WANDB_API_KEY = f.read().strip()

_login(token=HF_TOKEN, add_to_git_credential=False)
wandb.login(key=WANDB_API_KEY)


@st.cache_resource
def setup_inference():

    # Create a database session object that points to the URL.
    model, preprocessor, tokeniser = setup()
    return model, preprocessor, tokeniser

model, preprocessor, tokeniser = setup_inference()

st.title("Hacker News Upvotes Prediction Classifier")

st.write("This is a simple app to predict the number of upvotes a Hacker News post will get, using a backend API.")

# Entry form
with st.form("entry_form"):
    title = st.text_input("Title")
    url = st.text_input("URL")
    week_of_year = st.number_input("Week of year", min_value=1, max_value=52, value=52)
    day_of_week = st.number_input("Day of week", min_value=0, max_value=6, value=6)
    hour_of_day = st.number_input("Hour of day", min_value=0, max_value=23, value=13)
    submit_button = st.form_submit_button("Predict Upvotes")

# Submit button
if submit_button:
    # --- Prepare data payload for FastAPI ---
    # This dictionary structure MUST match what your FastAPI endpoint expects
    payload = {
        "title": title,
        "url": url,
        "week_of_year": week_of_year,
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
    }


    try:
        model, preprocessor, tokeniser = setup_inference()
        prediction_probability = get_model_features(title, week_of_year, day_of_week, hour_of_day, url, preprocessor, tokeniser)

        # --- Display the result ---
        if prediction_probability is not None:
            st.write(f"Predicted Probability of being upvoted: {prediction_probability:.4f}")
        else:
            st.warning("Error")
    except Exception as e:
        st.error(f"An error occurred: {e}")
