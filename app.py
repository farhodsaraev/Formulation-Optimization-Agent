# ==============================================================================
# Final Application: Formulation Agent with Hybrid Parser (Final Version)
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

# --- Caching for Models and API Calls ---
@st.cache_resource
def load_artifacts():
    # ... (function is the same as before) ...
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
    # ... (function is the same as before) ...
    try:
        clean_name = ingredient_name.strip().replace('*', '')
        if not clean_name: return "Empty", "-"
        results = pcp.get_compounds(clean_name, 'name')
        if results: return "Verified", results[0].molecular_formula
        else: return "Not Found", "-"
    except Exception: return "API Error", "-"

@st.cache_data(show_spinner=False)
def analyze_complex_ingredient(ingredient_name):
    # ... (function is the same as before) ...
    analysis_prompt = f"""As a cosmetic chemist, analyze "{ingredient_name}". Is it a polymer, blend, extract, or trade name? What are its primary chemical components? Respond ONLY in this format:
ANALYSIS: [Brief analysis]
COMPONENTS: [Component 1, Component 2]"""
    try:
        response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": analysis_prompt}], stream=False)
        content = response.choices[0].message.content
        analysis = re.search(r"ANALYSIS: (.*)", content).group(1)
        components_str = re.search(r"COMPONENTS: (.*)", content).group(1)
        components = [c.strip() for c in components_str.split(',')]
        verified_components = []
        for comp in components:
            status, _ = verify_ingredient_pubchem(comp)
            verified_components.append(f"{comp} ({status})")
        return f"{analysis}: " + ", ".join(verified_components), "Complex"
    except Exception: return "Analysis Failed", "Complex"


# --- Load Artifacts ---
rf_model, tfidf_vectorizer, label_encoder = load_artifacts()

# --- UI Sidebar ---
st.sidebar.title("🔬 R&D Parameters")
# ... (Sidebar UI is the same as before) ...
product_categories = ["Auto-detect"] + sorted(list(label_encoder.classes_)) if label_encoder else ["Auto-detect"]
product_type = st.sidebar.selectbox("Select Product Type", options=product_categories)
price_point = st.sidebar.selectbox("Target Price Point", options=["Mass-market", "Prestige", "Luxury"])
constraints = st.sidebar.multiselect("Formulation Constraints", options=["Silicone-free", "Paraben-free", "Sulfate-free", "Fragrance-free", "Vegan"])
st.sidebar.markdown("---")
ingredients_input = st.sidebar.text_area("Enter Key Ingredients", height=200, placeholder="e.g., Water, Glycerin, Hyaluronic Acid...")
submit_button = st.sidebar.button(label='✨ Generate & Verify Formulation')


# --- Main App Interface ---
st.title("🧪 Formulation Optimization Agent")
st.write("Your AI partner for creating and validating cosmetic formulations.")
st.markdown("---")

if 'formulation_text' not in st.session_state:
    st.session_state.formulation_text = ""

# --- Main Logic ---
if submit_button and ingredients_input:
    st.session_state.formulation_text = ""
    st.header("1. AI Analysis")
    # ... (Prediction logic is the same) ...
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
    
    # --- SLIGHTLY RELAXED, BUT STILL CLEAR PROMPT ---
    prompt_v3 = f"""
    Act as a world-class cosmetic chemist. Create a sample formulation based on the R&D brief below.
    Your primary output should be a professional formulation table in Markdown format. You may also provide additional context like a Chemist's Note.

    **R&D Brief:**
    - **Product Category:** {predicted_label}
    - **User's Key Ingredients:** {ingredients_input}
    - **Target Price Point:** {price_point}
    - **Formulation Constraints:** {constraints_text}
    """
    try:
        with st.spinner("Generating formulation with Groq's high-speed LLM..."):
            response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt_v3}], stream=False)
            st.session_state.formulation_text = response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the Groq API: {e}")

# --- Display and Verification Logic ---
if st.session_state.formulation_text:
    st.markdown(st.session_state.formulation_text)
    st.markdown("---")
    st.header("3. PubChem Ingredient Verification & Analysis")
    
    # --- THIS IS THE NEW HYBRID PARSER ---
    ingredients_to_verify = []
    text = st.session_state.formulation_text

    # Parser 1: Look for Markdown table rows
    table_ingredients = re.findall(r"\|\s*([^|]+?)\s*\|", text)
    if table_ingredients:
        # If a table is found, we assume the second column is the ingredient
        # This logic is simplified for clarity; a real production system might be more complex
        rows = [row.strip() for row in text.split('\n') if row.strip().startswith('|')]
        for row in rows:
            cols = [col.strip() for col in row.split('|')]
            if len(cols) > 2:
                 # Check to filter out headers and separator lines
                if "---" not in cols[2] and "Ingredient" not in cols[2]:
                    ingredients_to_verify.append(cols[2])

    # Parser 2: If no table ingredients, look for numbered or bulleted lists
    if not ingredients_to_verify:
        # This pattern looks for lines like "1. Ingredient Name (1%):" or "- Ingredient Name:"
        list_ingredients = re.findall(r"^\s*(?:\d+\.|-)\s*([A-Za-z\s\(\)-]+)", text, re.MULTILINE)
        if list_ingredients:
            ingredients_to_verify = [name.strip() for name in list_ingredients]

    if ingredients_to_verify:
        verification_data = []
        with st.spinner("Performing deep verification on ingredients..."):
            for ingredient in ingredients_to_verify:
                status, formula = verify_ingredient_pubchem(ingredient)
                if status == "Not Found":
                    status, formula = analyze_complex_ingredient(ingredient)
                if status != "Empty":
                    verification_data.append({"Ingredient": ingredient, "Status": status, "Details / Formula": formula})

        df_verification = pd.DataFrame(verification_data)

        def color_status(status):
            if status == "Verified": return 'background-color: #28a745; color: white'
            elif status.startswith("Complex") or status.startswith("Analysis"): return 'background-color: #17a2b8; color: white'
            else: return 'background-color: #dc3545; color: white'

        st.dataframe(df_verification.style.applymap(color_status, subset=['Status']), use_container_width=True)
    else:
        st.warning("Could not parse a formulation table or list from the generated text.")

else:
    st.info("Please provide your R&D parameters in the sidebar to begin.")
