"""Run:  python app.py   (after python train.py)"""
import json, random, joblib, pandas as pd, gradio as gr
from lime.lime_tabular import LimeTabularExplainer

model = joblib.load("artifacts/model.joblib")
meta = json.load(open("artifacts/meta.json"))
bg = pd.read_csv("artifacts/background.csv")
FEATURES, CLASSES = meta["features"], meta["class_names"]

explainer = LimeTabularExplainer(bg.values, feature_names=FEATURES, class_names=CLASSES,
                                 mode="classification", random_state=42)
_predict = lambda a: model.predict_proba(pd.DataFrame(a, columns=FEATURES))

def predict(*vals):
    row = pd.DataFrame([vals], columns=FEATURES)
    for f in FEATURES:                                   # empty box -> typical value
        if pd.isna(row.at[0, f]):
            row.at[0, f] = meta["medians"][f]
    probs = model.predict_proba(row)[0]
    idx = int(probs.argmax())
    exp = explainer.explain_instance(row.values[0], _predict, labels=(idx,), num_features=8)
    table = [[cond, f"{w:+.3f}"] for cond, w in exp.as_list(label=idx)]
    return {c: float(p) for c, p in zip(CLASSES, probs)}, table

def random_patient():
    r = bg.sample(1).iloc[0]
    return [float(r[f]) for f in FEATURES]

with gr.Blocks(title="Gallstone prediction") as demo:
    gr.Markdown("# Gallstone prediction\nEnter a patient's values and get a prediction with an explanation.\n\n"
                "⚠️ Educational project, **not a medical device**. Do not use it to diagnose anyone.")
    with gr.Tab("Predict"):
        inputs = []
        with gr.Row():
            columns = [gr.Column() for _ in range(3)]
        for i, f in enumerate(FEATURES):
            with columns[i % 3]:
                inputs.append(gr.Number(label=f, value=round(meta["medians"][f], 3),
                                        info=f"range in data: {meta['mins'][f]:g} to {meta['maxs'][f]:g}"))
        with gr.Row():
            btn = gr.Button("Predict", variant="primary")
            rnd = gr.Button("Load a random patient from the data")
        label = gr.Label(label="Prediction")
        table = gr.Dataframe(headers=["Feature condition", "Effect on predicted class"],
                             label="Top factors for this prediction (LIME)")
        btn.click(predict, inputs, [label, table])
        rnd.click(random_patient, None, inputs)
    with gr.Tab("Model performance"):
        gr.Markdown(f"Best model by cross-validation: **{meta['best_model']}**")
        gr.Dataframe(pd.DataFrame(meta["results"]))

if __name__ == "__main__":
    demo.launch()
