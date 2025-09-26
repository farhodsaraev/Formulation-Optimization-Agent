# ==============================================================================
# Final Application: Formulation Agent with Robust Parsing
# ==============================================================================

import streamlit as st
import joblib
import pandas as pd
from groq import Groq
import os
import re
import pubchempy as pcp

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

# --- Caching for Models and API Calls ---
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

@st.cache_data(show_spinner=False)
def verify_ingredient_pubchem(ingredient_name):
    """
    Queries PubChem to verify an ingredient. Using st.cache_data to avoid re-querying.
    """
    try:
        # Strip whitespace and handle potential asterisks from the LLM
        clean_name = ingredient_name.strip().replace('*', '')
        if not clean_name: # Skip if the name is empty after cleaning
            return None, None

        results = pcp.get_compounds(clean_name, 'name')
        if results:
            return "Verified", results[0].molecular_formula
        else:
            return "Not Found", "-"
    except Exception:
        return "Error", "API call failed"

# Load the ML artifacts
rf_model, tfidf_vectorizer, label_encoder = load_artifacts()

# --- UI: Sidebar for User Inputs ---
st.sidebar.title("🔬 R&D Parameters")
st.sidebar.write("Define your product concept here.")

product_categories = ["Auto-detect"] + sorted(list(label_encoder.classes_)) if label_encoder else ["Auto-detect"]

product_type = st.sidebar.selectbox("Select Product Type", options=product_categories)
price_point = st.sidebar.selectbox("Target Price Point", options=["Mass-market", "Prestige", "Luxury"])
constraints = st.sidebar.multiselect("Formulation Constraints", options=["Silicone-free", "Paraben-free", "Sulfate-free", "Fragrance-free", "Vegan"])

st.sidebar.markdown("---")
ingredients_input = st.sidebar.text_area("Enter Key Ingredients", height=200, placeholder="e.g., Water, Glycerin, Hyaluronic Acid...")
submit_button = st.sidebar.button(label='✨ Generate & Verify Formulation')


# --- Main App Interface ---
st.title("🧪 Formulation Optimization Agent")
st.write("Your AI partner for creating and validating cosmetic formulations. Define parameters in the sidebar and click generate.")
st.markdown("---")

if 'formulation_text' not in st.session_state:
    st.session_state.formulation_text = ""

# --- Main Logic Block ---
if submit_button and ingredients_input:
    st.session_state.formulation_text = ""
    st.header("1. AI Analysis")
    predicted_label = ""
    # ... (Prediction logic is the same) ...
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

    st.header("2. AI-Generated Concept Formulation")
    constraints_text = ", ".join(constraints) if constraints else "None"
    prompt_v2 = f"""Act as a world-class cosmetic chemist. Your primary task is to generate a formulation table in Markdown format. All other text should be secondary to this table.
    **R&D Brief:** - **Product Category:** {predicted_label} - **User's Key Ingredients:** {ingredients_input} - **Target Price Point:** {price_point} - **Formulation Constraints:** {constraints_text}
    **Your Instructions:** 1. **MANDATORY: Generate a Formulation Table:** The response MUST contain a complete Markdown table... (rest of prompt is the same)"""

    try:
        with st.spinner("Generating formulation with Groq's high-speed LLM..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt_v2}], stream=False
            )
            st.session_state.formulation_text = response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the Groq API: {e}")

# --- Display and Verification Logic ---
if st.session_state.formulation_text:
    # First, display the raw markdown output from the LLM
    st.markdown(st.session_state.formulation_text)
    
    st.markdown("---")
    st.header("3. PubChem Ingredient Verification")
    
    # --- THIS IS THE NEW, SMARTER PARSER ---
    ingredients_to_verify = []
    # Split the text into lines
    lines = st.session_state.formulation_text.split('\n')
    for line in lines:
        # Check if it's a table row (starts and ends with a '|')
        if line.strip().startswith('|') and line.strip().endswith('|'):
            # Split the row into columns
            columns = [col.strip() for col in line.split('|')]
            # A valid ingredient row will have at least 3 columns (e.g., | Phase | Ingredient | % |)
            if len(columns) > 2:
                ingredient_name = columns[2] # The ingredient is in the 2nd index (3rd column)
                # Filter out headers and markdown separator lines
                if ingredient_name and "---" not in ingredient_name and "Ingredient Name" not in ingredient_name:
                    ingredients_to_verify.append(ingredient_name)

    if ingredients_to_verify:
        verification_data = []
        with st.spinner("Verifying ingredients against the PubChem database..."):
            for ingredient in ingredients_to_verify:
                status, formula = verify_ingredient_pubchem(ingredient)
                if status: # Only add if the ingredient name was not empty
                    verification_data.append({"Ingredient": ingredient, "Status": status, "Formula": formula})

        df_verification = pd.DataFrame(verification_data)

        def color_status(status):
            if status == "Verified": return 'background-color: #28a745; color: white'
            elif status == "Not Found": return 'background-color: #dc3545; color: white'
            else: return 'background-color: #ffc107; color: black'

        st.dataframe(df_verification.style.applymap(color_status, subset=['Status']), use_container_width=True)
    else:
        st.warning("Could not parse any ingredients from the generated text to verify.")
else:
    st.info("Please provide your R&D parameters in the sidebar to begin.")
