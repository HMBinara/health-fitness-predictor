import streamlit as st
import numpy as np
import pickle
import glob
import os


# -------------------------------------------------------
# Load your model (NO CHANGES TO PICKLE FILE)
# - If multiple pickles exist in the folder, let the user pick one.
# - Show clear error messages instead of crashing.
# -------------------------------------------------------
def find_pickles():
    picks = glob.glob(os.path.join(os.getcwd(), "*.pkl")) + glob.glob(os.path.join(os.getcwd(), "*.pickle"))
    return [os.path.basename(p) for p in picks]


pickles = find_pickles()
model = None
model_load_error = None
if not pickles:
    model_load_error = "No .pkl or .pickle files found in the project folder. Please place your model file here."
else:
    # if only one, use it; otherwise allow user to select
    chosen = pickles[0] if len(pickles) == 1 else st.selectbox("Pick a model file", pickles)
    try:
        with open(chosen, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        model_load_error = f"Failed to load pickle '{chosen}': {e}"

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.title("💪 Health & Fitness Prediction App")
st.write("Please enter all input values below. After filling all fields, click **Predict**.")

st.markdown("### 📝 Enter Input Features")

# If model failed to load, show error and stop early
if model is None:
    if model_load_error:
        st.error(model_load_error)
    else:
        st.error("Model not loaded.")
    st.stop()

# -------------------------------------------------------
# Determine feature names (use model metadata when available)
# -------------------------------------------------------
def get_feature_names(mdl):
    if hasattr(mdl, "feature_names_in_"):
        try:
            return list(mdl.feature_names_in_)
        except Exception:
            pass
    if hasattr(mdl, "n_features_in_"):
        try:
            n = int(mdl.n_features_in_)
            return [f"Feature {i+1}" for i in range(n)]
        except Exception:
            pass
    if hasattr(mdl, "coef_"):
        try:
            arr = np.asarray(mdl.coef_)
            n = arr.shape[-1] if arr.ndim > 1 else arr.shape[0]
            return [f"Feature {i+1}" for i in range(n)]
        except Exception:
            pass
    # default to 12 features
    return [f"Feature {i+1}" for i in range(12)]


feature_names = get_feature_names(model)


st.markdown("---")

# render inputs dynamically
inputs = []
cols = st.columns(2)
for i, name in enumerate(feature_names):
    default = 0.0
    key = f"input_{i}"
    inputs.append(cols[i % 2].number_input(name, value=default, key=key))

st.markdown("---")

# Prediction
if st.button("Predict"):
    X = np.array([inputs])
    try:
        prediction = model.predict(X)
        st.success(f"📌 Prediction Result: **{prediction[0]}**")
        if hasattr(model, "predict_proba"):
            st.write("Probabilities:", model.predict_proba(X).tolist())
    except Exception as e:
        st.error("❌ ERROR: Failed to run prediction.")
        st.error(str(e))
