
# ---------------------------------------------------
import io
import json
import pickle
from typing import Any, List
import os
import glob

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Pickle Model Predictor", layout="centered")


def load_model_from_path(path: str) -> Any:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load pickle from path: {e}")
        return None


def model_summary(model: Any) -> str:
    try:
        return repr(model)
    except Exception:
        return str(model)


def build_feature_list(model: Any) -> List[str]:
    # Prefer explicit feature names
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass

    # Fallback to n_features_in_
    if hasattr(model, "n_features_in_"):
        try:
            n = int(model.n_features_in_)
            return [f"f{i}" for i in range(n)]
        except Exception:
            pass

    # Try to infer from coef_ shape (linear models)
    if hasattr(model, "coef_"):
        try:
            coef = np.asarray(model.coef_)
            if coef.ndim == 1:
                n = coef.shape[0]
            else:
                n = coef.shape[-1]
            return [f"f{i}" for i in range(n)]
        except Exception:
            pass

    return []


def predict_with_model(model: Any, X: pd.DataFrame) -> dict:
    out = {}
    try:
        preds = model.predict(X)
        out["prediction"] = preds.tolist()
    except Exception as e:
        out["prediction_error"] = str(e)

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            out["probabilities"] = np.round(probs, 6).tolist()
    except Exception as e:
        out["probabilities_error"] = str(e)

    return out


def main():
    st.title("Pickle model — Streamlit predictor")

    

    # find pickle files in project folder
    pickles = glob.glob(os.path.join(os.getcwd(), "*.pkl")) + glob.glob(os.path.join(os.getcwd(), "*.pickle"))
    pickles = [os.path.basename(p) for p in pickles]

    if not pickles:
        st.error("No pickle files found in the project folder. Place your model file (e.g., model.pkl) here and reload the app.")
        return

    if len(pickles) == 1:
        model_file = pickles[0]
        
    else:
        model_file = st.selectbox("Choose a pickle model to load", pickles)

    model = load_model_from_path(os.path.join(os.getcwd(), model_file))
    if model is None:
        st.error("Failed to load model.")
        return

    

    # Build feature inputs
    feature_names = build_feature_list(model)

    st.subheader("Enter input features")

    input_vals = {}
    if feature_names:
        st.write("Model feature names detected. You can enter values manually or apply an example preset:")

        # build simple example presets
        examples = {}
        examples["Example - zeros"] = {f: 0.0 for f in feature_names}
        examples["Example - ones"] = {f: 1.0 for f in feature_names}
        examples["Example - sequence"] = {f: float(i + 1) for i, f in enumerate(feature_names)}

        with st.expander("Examples (preview)"):
            ex_df = pd.DataFrame(examples).T
            st.dataframe(ex_df)
            ex_choice = st.selectbox("Pick an example to apply", ["-- none --"] + list(examples.keys()))
            if st.button("Apply example"):
                if ex_choice and ex_choice != "-- none --":
                    chosen = examples[ex_choice]
                    for fname, val in chosen.items():
                        st.session_state[f"input_{fname}"] = float(val)

        cols = st.columns(2)
        for i, fname in enumerate(feature_names):
            key = f"input_{fname}"
            default = float(st.session_state.get(key, 0.0))
            col = cols[i % 2]
            input_vals[fname] = col.number_input(fname, value=default, format="%f", key=key)
    else:
        st.write("No feature names detected. Enter a JSON object for a single row, e.g. {\"f0\": 1, \"f1\": 2}")
        text = st.text_area("Input JSON (single row)", value="{\"f0\": 0}")
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                input_vals = obj
            else:
                st.error("Please enter a JSON object representing a single row.")
                return
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            return

    # Build DataFrame
    try:
        X = pd.DataFrame([input_vals])
    except Exception as e:
        st.error(f"Failed to build input DataFrame: {e}")
        return

    st.write("Input preview:")
    st.dataframe(X)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            result = predict_with_model(model, X)

        st.subheader("Result")
        if "prediction" in result:
            st.success("Prediction:")
            st.write(result["prediction"])
        if "probabilities" in result:
            st.write("Probabilities:")
            st.write(result["probabilities"])
        if "prediction_error" in result:
            st.error(result["prediction_error"])
        if "probabilities_error" in result:
            st.error(result["probabilities_error"])


if __name__ == "__main__":
    main()
