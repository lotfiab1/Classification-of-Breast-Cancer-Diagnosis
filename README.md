# Breast Cancer Diagnosis Prediction

This project is a Streamlit app for predicting breast cancer diagnosis using machine learning models trained on the Breast Cancer Wisconsin dataset.

## Project contents

- `app.py` - Streamlit application for entering biopsy measurement values and predicting whether a tumor is benign or malignant.
- `Breast Cancer Wisconsin.csv` - Dataset used for training and analysis.
- `Project Classification of Breast Cancer Diagnosis.ipynb` - Jupyter notebook with exploration, model training, and evaluation.
- `Projet Machine Learning.pdf` - Project report.
- `logistic_model.pkl` - Saved logistic regression model.
- `knn_model.pkl` - Saved KNN model.
- `decision_tree_model.pkl` - Saved decision tree model.
- `scaler.pkl` - Saved scaler (if used for preprocessing).

## Requirements

Install dependencies before running the app.

```bash
pip install streamlit numpy joblib
```

If you want to run the notebook, also install:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Running the app

From the project directory, run:

```bash
streamlit run app.py
```

Then open the local URL shown in your browser.

## How to use

1. Enter the biopsy measurement values for all features shown in the form.
2. Select one of the available models:
   - Régression Logistique
   - KNN
   - Arbre de Décision
3. Click **Prédire** to get a prediction.

The app will display whether the diagnosis is predicted as:

- `Maligne (M)`
- `Bénigne (B)`

and, when available, a probability score.

## Notes

- The model files must remain in the same directory as `app.py` for the app to load them correctly.
- The app expects all 30 numeric features from the dataset.

## License

This project is provided as-is for educational purposes.
