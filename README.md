# Gallstone Prediction

Predicting gallstone status from clinical and bioimpedance measurements, using the UCI Gallstone dataset. The project compares four machine-learning models and serves the best one in a small web app that explains each prediction.

> **Disclaimer:** educational project only. It is not a medical device and must not be used to diagnose anyone.

## What it does
- Trains and compares **Logistic Regression, Random Forest, SVM and XGBoost**
- Tunes each model with grid search and 5-fold cross-validation
- Selects the best model by cross-validated macro-F1, then evaluates on a held-out test set (stratified 80/20 split)
- Provides a **Gradio web app**: enter a patient's values, get the predicted class, its probability, and the top factors behind the prediction (**LIME**)

## Project files
| File | Purpose |
|---|---|
| `train.py` | Trains the models, prints the results, saves the best model to `artifacts/` |
| `app.py` | Gradio web app |
| `requirements.txt` | Python packages |
| `dataset-uci.csv` | UCI Gallstone dataset |
| `notebooks/` | Original experiments (Decision Tree, SHAP and LIME plots) |

## How to run
```
pip install -r requirements.txt
python train.py dataset-uci.csv
python app.py
```
Then open the local link printed in the terminal.

**Google Colab:** upload the files, run `!pip install -q gradio lime xgboost`, then `!python train.py dataset-uci.csv`. Change the last line of `app.py` to `demo.launch(share=True)` and run `!python app.py`. Open the `gradio.live` link.

## Results
| Model | CV F1 (macro) | Test accuracy | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|
Logistic Regression |0.771|0.797|0.788|0.812|0.8|
Random Forest|0.786|0.812|0.833|0.781|0.806|
SVM|0.787|0.828|0.818|0.844|0.831|
XGBoost|0.763|0.797|0.852|0.719|0.78|

## Data
UCI Machine Learning Repository, Gallstone dataset. Please check the dataset's licence and cite the original authors.

## Author
Your name, course name, year.
