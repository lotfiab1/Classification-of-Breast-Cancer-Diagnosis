import streamlit as st
import numpy as np
import joblib

import os
print("Chemin actuel :", os.getcwd())

# Charger les modèles
models = {
    "Régression Logistique": joblib.load("models/logistic_model.pkl"),
    "KNN": joblib.load("models/knn_model.pkl"),
    "Arbre de Décision": joblib.load("models/decision_tree_model.pkl")
}

# Liste des caractéristiques (sans la cible)
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

st.title("Prédiction du diagnostic du cancer du sein")

st.write("Entrez les valeurs cliniques des mesures de la biopsie:")

# Formulaire de saisie
with st.form(key="input_form"):
    input_data = []
    for feature in features:
        val = st.number_input(label=feature, format="%.5f")
        input_data.append(val)
    model_choice = st.selectbox("Choisir le modèle", options=list(models.keys()))
    submit_button = st.form_submit_button(label="Prédire")

if submit_button:
    input_array = np.array(input_data).reshape(1, -1)
    model = models[model_choice]
    
    prediction = model.predict(input_array)[0]
    
    # Vérifier si predict_proba existe (ex: KNN et logistic regression oui, decision tree parfois)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_array)[0][prediction]
        proba_text = f" avec une probabilité de {proba:.2f}"
    else:
        proba_text = ""
    
    diagnosis = "Maligne (M)" if prediction == 1 else "Bénigne (B)"
    
    st.success(f"Diagnostic prédit: **{diagnosis}**{proba_text}")
