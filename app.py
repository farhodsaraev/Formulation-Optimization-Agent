# ==============================================================================
# Final Application: Formulation Optimization Agent
# ==============================================================================

import streamlit as st
import joblib
import pandas as pd
from groq import Groq
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Formulation Optimization Agent",
    page_icon="🧪",
    layout="wide"
)

# --- Groq API Client Initialization ---
# The API key is securely accessed from Streamlit's secrets management.
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("Groq API key not found. Please add it to your Streamlit Secrets.")
    st.stop()


# --- Caching the Model and Preprocessors ---
@st.cache_resource
def load_artifacts():
    """
    Loads the saved machine learning model, TF-IDF vectorizer, and label encoder.
    """
    try:
        model = joblib.load('random_forest_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        encoder = joblib.load('label_encoder.joblib')
        return model, vectorizer, encoder
    except FileNotFoundError:
        st.error("Model artifacts not found. Please ensure the .joblib files are in the repository root.")
        return None, None, None

# Load the artifacts
rf_model, tfidf_vectorizer, label_encoder = load_artifacts()

# --- Header and Title ---
st.title("🧪 Formulation Optimization Agent")
st.write("""
Welcome! This tool helps cosmetic chemists brainstorm new formulations. 
Start by providing a list of ingredients, and our AI will predict the product category 
and then generate a concept formulation.
""")
st.markdown("---")

# --- Prediction and Formulation Interface ---
st.header("1. Input Your Ingredients")
st.write("Provide a list of ingredients you are considering, separated by commas or on new lines.")

with st.form(key='ingredient_form'):
    ingredients_input = st.text_area(
        "Ingredient List",
        height=200,
        placeholder="e.g., Water, Glycerin, Hyaluronic Acid, Cetearyl Alcohol, ...",
        help="The more ingredients you provide, the better the context for the AI."
    )
    submit_button = st.form_submit_button(label='Predict & Generate Formulation ✨')

# --- Main Logic Block ---
if submit_button and ingredients_input:
    if rf_model is not None:
        # --- Stage 1: Predictive Model ---
        with st.spinner("Analyzing ingredients and predicting category..."):
            input_series = pd.Series([ingredients_input])
            input_tfidf = tfidf_vectorizer.transform(input_series)
            prediction_encoded = rf_model.predict(input_tfidf)
            prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

        st.header("2. AI Analysis Result")
        st.success(f"Predicted Product Category: **{prediction_label}**")

        # --- Stage 2: Generative Model ---
        st.header("3. AI-Generated Concept Formulation")
        
        # Construct a detailed prompt for the LLM
        prompt = f"""
        Act as a senior cosmetic chemist. Your task is to create a sample formulation based on a user's ingredient list and a predicted product category.

        **Product Category:** {prediction_label}
        **User's Key Ingredients:** {ingredients_input}

        **Instructions:**
        1.  Create a complete, professional-looking formulation table.
        2.  The table must include columns for: Phase, Ingredient Name (INCI), Percentage (%), and Function.
        3.  Incorporate the user's key ingredients into the formulation logically.
        4.  Fill in the rest of the formulation with common, appropriate ingredients to make a complete and stable product.
        5.  Ensure the total percentage adds up to 100%.
        6.  The formulation should be divided into logical phases (e.g., Water Phase, Oil Phase, Cool-Down Phase).
        7.  Provide a brief "Chemist's Note" at the end, explaining the role of the key ingredients and the overall rationale behind the formulation.

        Generate the response in Markdown format.
        """

        try:
            # Use st.write_stream to display the response as it's generated
            with st.spinner("Generating formulation with Groq's high-speed LLM..."):
                stream = client.chat.completions.create(
                    model="llama3-70b-8192", # A powerful and fast model
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                st.write_stream(stream)

        except Exception as e:
            st.error(f"An error occurred with the Groq API: {e}")

    else:
        st.warning("Cannot perform prediction as model artifacts are not loaded.")

st.markdown("---")
st.write("Built with ❤️ by a collaborative Human-AI team.")
