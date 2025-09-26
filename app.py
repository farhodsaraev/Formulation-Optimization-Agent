# ==============================================================================
# Phase 2: Advanced Functionality & User Experience
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
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("Groq API key not found. Please add it to your Streamlit Secrets.")
    st.stop()

# --- Caching the Model and Preprocessors ---
@st.cache_resource
def load_artifacts():
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

# --- UI: Sidebar for User Inputs ---
st.sidebar.title("🔬 R&D Parameters")
st.sidebar.write("Define your product concept here.")

# Get a list of our trained categories for the dropdown
product_categories = ["Auto-detect"] + sorted(list(label_encoder.classes_)) if label_encoder else ["Auto-detect"]

# Input widgets in the sidebar
product_type = st.sidebar.selectbox(
    "Select Product Type",
    options=product_categories,
    help="Choose a product category or let the AI detect it from your ingredients."
)

price_point = st.sidebar.selectbox(
    "Target Price Point",
    options=["Mass-market", "Prestige", "Luxury"],
    help="This will influence the AI's choice of supporting ingredients."
)

constraints = st.sidebar.multiselect(
    "Formulation Constraints",
    options=["Silicone-free", "Paraben-free", "Sulfate-free", "Fragrance-free", "Vegan", "Clean Beauty Compliant"],
    help="Select any 'free-of' claims the final formulation should adhere to."
)

st.sidebar.markdown("---")
ingredients_input = st.sidebar.text_area(
    "Enter Key Ingredients",
    height=200,
    placeholder="e.g., Water, Glycerin, Hyaluronic Acid, Squalane, Niacinamide...",
    help="List the core ingredients you want to feature in the formulation."
)

submit_button = st.sidebar.button(label='✨ Generate Formulation')

# --- Main App Interface ---
st.title("🧪 Formulation Optimization Agent")
st.write("Your AI partner for creating innovative cosmetic formulations. Define your parameters in the sidebar and click generate.")
st.markdown("---")


# --- Main Logic Block ---
if submit_button and ingredients_input:
    if rf_model is not None:
        
        # --- Stage 1: Determine Product Category ---
        st.header("1. AI Analysis")
        predicted_label = ""
        
        with st.spinner("Analyzing ingredients..."):
            if product_type == "Auto-detect":
                input_series = pd.Series([ingredients_input])
                input_tfidf = tfidf_vectorizer.transform(input_series)
                prediction_encoded = rf_model.predict(input_tfidf)
                predicted_label = label_encoder.inverse_transform(prediction_encoded)[0]
                st.success(f"AI Detected Product Category: **{predicted_label}**")
            else:
                predicted_label = product_type
                st.info(f"User-Selected Product Category: **{predicted_label}**")

        # --- Stage 2: Advanced Prompt Engineering & Generation ---
        st.header("2. AI-Generated Concept Formulation")
        
        constraints_text = ", ".join(constraints) if constraints else "None"

        # A more sophisticated prompt that incorporates all user inputs
        prompt_v2 = f"""
        Act as a world-class cosmetic chemist. Your task is to create a sample formulation based on a user's detailed R&D brief.

        **R&D Brief:**
        - **Product Category:** {predicted_label}
        - **User's Key Ingredients:** {ingredients_input}
        - **Target Price Point:** {price_point}
        - **Formulation Constraints:** {constraints_text}

        **Your Instructions:**
        1.  **Create a Professional Formulation Table:** The table must be in Markdown format and include columns for: Phase, Ingredient Name (INCI), Percentage (%), and Function.
        2.  **Incorporate Key Ingredients:** Logically integrate the user's key ingredients into the correct phases.
        3.  **Complete the Formula:** Fill in the rest of the formulation with common, appropriate ingredients to create a complete and stable product that respects the specified constraints.
        4.  **Respect Price Point:** Your choice of supporting ingredients (emulsifiers, emollients, actives) should reflect the target price point. For 'Luxury', you can use more exotic or high-tech ingredients. For 'Mass-market', stick to common, cost-effective options.
        5.  **Adhere to Constraints:** Ensure the final formula is, for example, 'Silicone-free' or 'Paraben-free' if specified.
        6.  **Total 100%:** The percentages must add up to exactly 100%.
        7.  **Provide a Chemist's Note:** After the table, write a brief rationale explaining your choices, how you met the constraints, and why the formulation is well-suited for the target product and price point.
        """

        try:
            with st.spinner("Generating formulation with Groq's high-speed LLM..."):
                def stream_generator(stream):
                    for chunk in stream:
                        if content := chunk.choices[0].delta.content:
                            yield content

                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt_v2}],
                    stream=True,
                )
                
                st.write_stream(stream_generator(stream))

        except Exception as e:
            st.error(f"An error occurred with the Groq API: {e}")

    else:
        st.warning("Cannot perform prediction as model artifacts are not loaded.")
else:
    st.info("Please provide your R&D parameters in the sidebar to begin.")
