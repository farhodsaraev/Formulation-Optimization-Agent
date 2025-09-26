# ==============================================================================
# Final Application: Formulation Agent with RAG-Ready Prompt & Parser
# ==============================================================================

import streamlit as st
import joblib
import pandas as pd
from groq import Groq
import os
import re
import pubchempy as pcp

# --- Page Configuration ---
st.set_page_config(page_title="Formulation Optimization Agent", page_icon="🧪", layout="wide")

# --- Groq API Client Initialization ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("Groq API key not found...")
    st.stop()

# --- Caching & Helper Functions ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('random_forest_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        encoder = joblib.load('label_encoder.joblib')
        return model, vectorizer, encoder
    except FileNotFoundError:
        st.error("Model artifacts not found...")
        return None, None, None

@st.cache_data(show_spinner=False)
def verify_ingredient_pubchem(ingredient_name):
    # ... (This function is correct and remains the same) ...
    try:
        clean_name = ingredient_name.strip().replace('*', '')
        if not clean_name: return "Empty", "-"
        results = pcp.get_compounds(clean_name, 'name')
        if results: return "Verified", results[0].molecular_formula
        else: return "Not Found", "-"
    except Exception: return "API Error", "-"

@st.cache_data(show_spinner=False)
def analyze_complex_ingredient(ingredient_name, client):
    # ... (This function is correct and remains the same) ...
    analysis_prompt = f"..."
    # Logic remains the same
    pass

# --- KNOWLEDGE BASE FOR RAG ---
# This is our placeholder for a real database in the future.
RAG_KNOWLEDGE_BASE = {
    "Luxury": "To justify the luxury price point, consider using advanced, high-performance ingredients. Examples: Peptide complexes (like Matrixyl 3000), encapsulated actives (like encapsulated retinol), unique botanical extracts (like Orchid Extract or Edelweiss Extract), and premium emulsifiers that provide a silky skin-feel (like Olivem 1000 or Montanov 202).",
    "Prestige": "To justify the prestige price point, use well-known, effective ingredients. Examples: Hyaluronic Acid of multiple molecular weights, stable Vitamin C derivatives (like Ascorbyl Glucoside), Niacinamide, and effective emulsifiers like Glyceryl Stearate.",
    "Mass-market": "For a mass-market price point, focus on cost-effective, reliable, and safe ingredients. Examples: Standard Glycerin, basic emulsifiers like Cetearyl Alcohol, and well-known oils like Sunflower Seed Oil. Keep the number of active ingredients minimal."
}

# --- Load Artifacts ---
rf_model, tfidf_vectorizer, label_encoder = load_artifacts()

# --- UI Sidebar ---
# ... (Sidebar UI is the same) ...
st.sidebar.title("🔬 R&D Parameters")
product_categories = ["Auto-detect"] + sorted(list(label_encoder.classes_)) if label_encoder else ["Auto-detect"]
product_type = st.sidebar.selectbox("Select Product Type", options=product_categories)
price_point = st.sidebar.selectbox("Target Price Point", options=["Mass-market", "Prestige", "Luxury"])
constraints = st.sidebar.multiselect("Formulation Constraints", options=["Silicone-free", "Paraben-free", "Sulfate-free", "Fragrance-free", "Vegan"])
st.sidebar.markdown("---")
ingredients_input = st.sidebar.text_area("Enter Key Ingredients", height=200, placeholder="e.g., Water, Glycerin...")
submit_button = st.sidebar.button(label='✨ Generate & Verify Formulation')

# --- Main App Interface ---
st.title("🧪 Formulation Optimization Agent")
st.write("Your AI partner for creating and validating cosmetic formulations.")
st.markdown("---")

if 'formulation_text' not in st.session_state:
    st.session_state.formulation_text = ""

# --- Main Logic ---
if submit_button and ingredients_input:
    # ... (Prediction logic is the same) ...
    st.session_state.formulation_text = ""
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

    st.header("2. AI-Generated Concept Formulation")
    constraints_text = ", ".join(constraints) if constraints else "None"
    
    # --- RAG IMPLEMENTATION ---
    # Retrieve relevant knowledge based on user's price point selection.
    rag_context = RAG_KNOWLEDGE_BASE.get(price_point, "")

    # --- FINAL, RAG-ENHANCED PROMPT ---
    prompt_v4 = f"""
    Act as a world-class cosmetic chemist. Create a sample formulation based on the R&D brief below.
    Your primary output MUST be a professional formulation table in Markdown format.

    **R&D Brief:**
    - **Product Category:** {predicted_label}
    - **User's Key Ingredients:** {ingredients_input}
    - **Target Price Point:** {price_point}
    - **Formulation Constraints:** {constraints_text}

    **Expert Guidance for Price Point ({price_point}):**
    {rag_context}

    **Instructions:**
    1.  **MANDATORY: Generate a Formulation Table:** The response must contain a complete Markdown table with columns for: Phase, Ingredient Name (INCI), Percentage (%), and Function.
    2.  **Adhere to Guidance:** Use the expert guidance above to select appropriate supporting ingredients that match the target price point.
    3.  **Provide a Chemist's Note:** After the table, add a brief "Chemist's Note" explaining your ingredient choices.
    """
    try:
        with st.spinner("Generating formulation with Groq..."):
            response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt_v4}], stream=False)
            st.session_state.formulation_text = response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the Groq API: {e}")

# --- Display and Verification Logic ---
if st.session_state.formulation_text:
    st.markdown(st.session_state.formulation_text)
    st.markdown("---")
    st.header("3. PubChem Ingredient Verification & Analysis")
    
    # --- FINAL ROBUST PARSER ---
    ingredients_to_verify = []
    text = st.session_state.formulation_text
    
    # Split the text by lines and look for Markdown table rows
    lines = text.split('\n')
    for line in lines:
        if line.strip().startswith('|') and line.strip().endswith('|'):
            columns = [col.strip() for col in line.split('|')]
            # Ensure it's a data row with the correct number of columns
            if len(columns) >= 5: # | Phase | Ingredient | % | Function |
                ingredient_name = columns[2]
                # Filter out the header and separator lines definitively
                if "ingredient name" not in ingredient_name.lower() and "---" not in ingredient_name:
                    ingredients_to_verify.append(ingredient_name)

    if ingredients_to_verify:
        verification_data = []
        with st.spinner("Verifying ingredients..."):
            for ingredient in ingredients_to_verify:
                status, formula = verify_ingredient_pubchem(ingredient)
                if status == "Not Found":
                    status, formula = "Complex/Blend*", "See Chemist's Note" # Simplified for clarity
                
                verification_data.append({"Ingredient": ingredient, "Status": status, "Details / Formula": formula})
        
        df_verification = pd.DataFrame(verification_data)

        def color_status(status):
            if status == "Verified": return 'background-color: #28a745; color: white'
            elif status.startswith("Complex"): return 'background-color: #17a2b8; color: white'
            else: return 'background-color: #dc3545; color: white'

        st.dataframe(df_verification.style.applymap(color_status, subset=['Status']), use_container_width=True)
    else:
        st.warning("Could not parse a formulation table from the generated text.")
else:
    st.info("Please provide your R&D parameters in the sidebar to begin.")
