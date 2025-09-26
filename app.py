# ==============================================================================
# Step 7: Streamlit Application with Integrated Predictive Model
# ==============================================================================

import streamlit as st
import joblib
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Formulation Optimization Agent",
    page_icon="🧪",
    layout="wide"
)

# --- Caching the Model and Preprocessors ---
# Use @st.cache_resource to load these heavy objects only once.
@st.cache_resource
def load_artifacts():
    """
    Loads the saved machine learning model, TF-IDF vectorizer, and label encoder.
    The @st.cache_resource decorator ensures this function runs only once.
    """
    try:
        model = joblib.load('random_forest_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        encoder = joblib.load('label_encoder.joblib')
        return model, vectorizer, encoder
    except FileNotFoundError:
        st.error("Model artifacts not found. Please ensure the .joblib files are in the root of the repository.")
        return None, None, None

# Load the artifacts
rf_model, tfidf_vectorizer, label_encoder = load_artifacts()

# --- Header and Title ---
st.title("🧪 Formulation Optimization Agent")
st.subheader("Predict Product Category from Ingredients")
st.write("""
This tool uses a machine learning model to predict the most likely product category 
based on a list of ingredients. This is the first step in our R&D process.
""")
st.markdown("---")

# --- Prediction Interface ---
st.header("1. Enter Ingredients")
st.write("Provide the ingredient list, one ingredient per line or separated by commas.")

# Use a form to group user input and the submit button
with st.form(key='ingredient_form'):
    # Text area for user to input ingredients
    ingredients_input = st.text_area(
        "Ingredient List",
        height=200,
        placeholder="e.g., Water, Glycerin, Hyaluronic Acid, Cetearyl Alcohol, ...",
        help="Separate ingredients with commas or new lines."
    )
    
    # Submit button for the form
    submit_button = st.form_submit_button(label='Predict Category ✨')

# --- Prediction Logic ---
# This block runs only when the submit button is pressed
if submit_button and ingredients_input:
    if rf_model is not None:
        # 1. Preprocess the input text (must be in a list or pd.Series)
        input_series = pd.Series([ingredients_input])
        
        # 2. Transform the text using the loaded TF-IDF vectorizer
        input_tfidf = tfidf_vectorizer.transform(input_series)
        
        # 3. Make a prediction using the loaded model
        prediction_encoded = rf_model.predict(input_tfidf)
        
        # 4. Decode the prediction back to the original label
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
        
        # 5. Display the result
        st.header("2. AI Prediction Result")
        st.success(f"The model predicts this is a **{prediction_label}**.")
        st.write("This prediction is based on statistical patterns found in thousands of commercial products.")
    else:
        st.warning("Cannot perform prediction as model artifacts are not loaded.")

# Add a footer or some other information
st.markdown("---")
st.write("Developed by an AI Engineer and their AI partner.")
