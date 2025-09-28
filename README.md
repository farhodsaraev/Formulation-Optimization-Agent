# Formulation-Optimization-Agent
# 🧪 Formulation Optimization Agent

A sophisticated, end-to-end AI application designed to assist cosmetics R&D professionals in brainstorming and developing new product formulations. This tool leverages a multi-stage AI pipeline to predict a product's category from an ingredient list and then generate a complete, professional concept formulation.

**Live Application:** [Link to your Streamlit App URL]

---

## Core Features

*   **Predictive Analysis:** Utilizes a `scikit-learn` Random Forest model trained on a dataset of over 1,400 real-world cosmetic products to predict a product category (e.g., Moisturizer, Serum, Cleanser) from a given list of ingredients.
*   **Generative Formulation:** Employs a high-speed Large Language Model (LLM) via the Groq API (`llama-3.3-70b-versatile`) to take the predicted category and user-input ingredients as context.
*   **Professional Output:** The LLM generates a complete, professional formulation table, including phases, ingredient names (INCI), realistic percentages, and chemical functions. It also provides a "Chemist's Note" explaining the formulation's rationale.
*   **Interactive UI:** A clean, user-friendly web interface built with Streamlit allows for easy input and clear visualization of the results.

## Project Architecture & Tech Stack

This project was built with a focus on performance, scalability, and a zero-dollar budget, using robust and widely-adopted technologies.

*   **Development Environment:** Google Colab
*   **Data Storage:** Google Drive
*   **Predictive Model:** Python, Pandas, Scikit-learn (Random Forest Classifier)
*   **Feature Engineering:** TF-IDF Vectorization for ingredient text processing
*   **Generative LLM:** Groq API (llama-3.3-70b-versatile)
*   **UI Framework:** Streamlit
*   **Deployment:** Streamlit Community Cloud
*   **Version Control:** GitHub

## How It Works: The AI Pipeline

1.  **User Input:** The user provides a list of desired ingredients.
2.  **Preprocessing:** The raw text is vectorized using a pre-trained TF-IDF model to convert the ingredient list into a numerical format.
3.  **Prediction:** The trained Random Forest model predicts the most likely product category.
4.  **Prompt Engineering:** A detailed prompt is constructed, providing the LLM with the predicted category, the user's ingredients, and instructions to act as a senior cosmetic chemist.
5.  **Generation:** The prompt is sent to the Groq API, which generates the final formulation in real-time.
6.  **Rendering:** The Streamlit app displays the predictive result and streams the generative output to the user.

## How to Use the Application

1.  Navigate to the live application URL.
2.  Enter a list of cosmetic ingredients into the text area, separated by commas or new lines.
3.  Click the "Predict & Generate Formulation ✨" button.
4.  View the AI's analysis and the generated concept formulation.

---

