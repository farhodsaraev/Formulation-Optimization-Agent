# ==============================================================================
# Final Application: Formulation Agent with Advanced Ingredient Analysis
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
        st.error("Model artifacts not found...")
        return None, None, None

@st.cache_data(show_spinner=False)
def verify_ingredient_pubchem(ingredient_name):
    try:
        clean_name = ingredient_name.strip().replace('*', '')
        if not clean_name: return "Empty", "-"
        
        results = pcp.get_compounds(clean_name, 'name')
        if results:
            return "Verified", results[0].molecular_formula
        else:
            return "Not Found", "-"
    except Exception:
        return "API Error", "-"

@st.cache_data(show_spinner=False)
def analyze_complex_ingredient(ingredient_name):
    """
    Uses an LLM to break down a complex ingredient and verifies its components.
    """
    analysis_prompt = f"""
    As a cosmetic chemist, analyze the ingredient name "{ingredient_name}".
    Is it a polymer, a commercial blend, a natural extract, or a trade name?
    Based on your analysis, what are its most likely primary chemical components?
    
    Provide your response in the following format ONLY:
    ANALYSIS: [Your brief analysis, e.g., A common emulsifier blend]
    COMPONENTS: [List the primary components separated by commas, e.g., Cetearyl Alcohol, Olive Oil Fatty Acids]
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": analysis_prompt}],
            stream=False,
        )
        content = response.choices[0].message.content
        
        # Parse the LLM's response
        analysis = re.search(r"ANALYSIS: (.*)", content).group(1)
        components_str = re.search(r"COMPONENTS: (.*)", content).group(1)
        components = [c.strip() for c in components_str.split(',')]
        
        # Verify each component
        verified_components = []
        for comp in components:
            status, _ = verify_ingredient_pubchem(comp)
            if status == "Verified":
                verified_components.append(f"{comp} (Verified)")
            else:
                verified_components.append(f"{comp} (Mixture/Polymer)")

        return f"{analysis}: " + ", ".join(verified_components), "Complex"

    except Exception:
        return "Analysis Failed", "Complex"


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
    # ... (Stage 1 & 2 logic is identical to before) ...
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
    prompt_v2 = f"""Act as a world-class cosmetic chemist... (Use the strengthened prompt from the previous step)"""
    try:
        with st.spinner("Generating formulation with Groq's high-speed LLM..."):
            response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt_v2}], stream=False)
            st.session_state.formulation_text = response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the Groq API: {e}")

# --- Display and Verification Logic ---
if st.session_state.formulation_text:
    st.markdown(st.session_state.formulation_text)
    st.markdown("---")
    st.header("3. PubChem Ingredient Verification & Analysis")

    ingredients_to_verify = []
    lines = st.session_state.formulation_text.split('\n')
    for line in lines:
        if line.strip().startswith('|') and line.strip().endswith('|'):
            columns = [col.strip() for col in line.split('|')]
            if len(columns) > 2:
                ingredient_name = columns[2]
                if ingredient_name and "---" not in ingredient_name and "Ingredient Name" not in ingredient_name:
                    ingredients_to_verify.append(ingredient_name)

    if ingredients_to_verify:
        verification_data = []
        with st.spinner("Performing deep verification on ingredients..."):
            for ingredient in ingredients_to_verify:
                status, formula = verify_ingredient_pubchem(ingredient)
                if status == "Not Found":
                    # --- THIS IS THE NEW LOGIC ---
                    # If not found, run the advanced analysis.
                    status, formula = analyze_complex_ingredient(ingredient)
                
                verification_data.append({"Ingredient": ingredient, "Status": status, "Details / Formula": formula})

        df_verification = pd.DataFrame(verification_data)

        def color_status(status):
            if status == "Verified": return 'background-color: #28a745; color: white'
            elif status.startswith("Complex") or status.startswith("Analysis"): return 'background-color: #17a2b8; color: white'
            else: return 'background-color: #dc3545; color: white'

        st.dataframe(df_verification.style.applymap(color_status, subset=['Status']), use_container_width=True)
    else:
        st.warning("Could not parse ingredients from the generated text.")
else:
    st.info("Please provide your R&D parameters in the sidebar to begin.")
