# ==============================================================================
# Final Application: Formulation Agent with PubChem Verification
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
        results = pcp.get_compounds(ingredient_name.strip(), 'name')
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

# Initialize session state to hold the formulation text
if 'formulation_text' not in st.session_state:
    st.session_state.formulation_text = ""

# --- Main Logic Block ---
if submit_button and ingredients_input:
    st.session_state.formulation_text = "" # Clear previous results
    
    # ... (Stage 1 and 2: Prediction and Generation) ...
    # This part remains the same
    
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

    # A more sophisticated and robust prompt that prioritizes the table format
    prompt_v2 = f"""
    Act as a world-class cosmetic chemist. Your primary task is to generate a formulation table in Markdown format. All other text should be secondary to this table.

    **R&D Brief:**
    - **Product Category:** {predicted_label}
    - **User's Key Ingredients:** {ingredients_input}
    - **Target Price Point:** {price_point}
    - **Formulation Constraints:** {constraints_text}

    **Your Instructions:**
    1.  **MANDATORY: Generate a Formulation Table:** The response MUST contain a complete Markdown table. This is the most critical part of your task. The table must include columns for: Phase, Ingredient Name (INCI), Percentage (%), and Function.
    2.  **Incorporate Key Ingredients:** Logically integrate the user's key ingredients into the correct phases within the table.
    3.  **Complete the Formula:** Fill in the rest of the formulation table with common, appropriate ingredients to create a complete and stable product that respects the specified constraints.
    4.  **Respect Price Point:** Your choice of supporting ingredients in the table should reflect the target price point. 'Luxury' implies more advanced or unique ingredients, while 'Mass-market' implies cost-effective, standard ingredients.
    5.  **Adhere to Constraints:** The final list of ingredients in the table MUST be, for example, 'Silicone-free' or 'Paraben-free' if specified.
    6.  **Total 100%:** The percentages in the table must add up to exactly 100%.
    7.  **Provide a Chemist's Note:** AFTER the mandatory Markdown table, you MAY add a brief "Chemist's Note" to explain your choices and the formulation's rationale.
    """

    try:
        with st.spinner("Generating formulation with Groq's high-speed LLM..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt_v2}],
                stream=False, # We need the full text before we can parse it
            )
            st.session_state.formulation_text = response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the Groq API: {e}")

# --- Display Generated Formulation and Verification Table ---
if st.session_state.formulation_text:
    st.markdown(st.session_state.formulation_text)
    
    st.markdown("---")
    st.header("3. PubChem Ingredient Verification")
    
    # Use regex to find all ingredients from the markdown table
    # This pattern looks for text between '|' characters, ignoring whitespace
    ingredients_to_verify = re.findall(r"\|\s*(.*?)\s*\|", st.session_state.formulation_text)
    
    # Filter out table headers and separators
    headers = ["Ingredient Name (INCI)", "Phase", "Percentage (%)", "Function", "---"]
    ingredients_to_verify = [
        name.strip() for name in ingredients_to_verify 
        if name.strip() not in headers and "%" not in name and "Phase" not in name
    ]

    if ingredients_to_verify:
        verification_data = []
        with st.spinner("Verifying ingredients against the PubChem database..."):
            for ingredient in ingredients_to_verify:
                status, formula = verify_ingredient_pubchem(ingredient)
                verification_data.append({
                    "Ingredient": ingredient,
                    "Status": status,
                    "Formula": formula
                })

        df_verification = pd.DataFrame(verification_data)

        # Color-coding the status column for better readability
        def color_status(status):
            if status == "Verified":
                return 'background-color: #28a745; color: white'
            elif status == "Not Found":
                return 'background-color: #dc3545; color: white'
            else:
                return 'background-color: #ffc107; color: black'

        st.dataframe(df_verification.style.applymap(color_status, subset=['Status']), use_container_width=True)
    else:
        st.warning("Could not parse any ingredients from the generated text to verify.")
else:
    st.info("Please provide your R&D parameters in the sidebar to begin.")

