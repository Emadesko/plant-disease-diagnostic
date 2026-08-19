import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Diagnostic maladies foliaires", page_icon="🌿")

@st.cache_resource
def load():
    model = tf.keras.models.load_model("plantvillage_mobilenet_finetuned.keras")
    class_names = json.load(open("class_names.json"))
    return model, class_names

model, class_names = load()
IMG_SIZE = (224, 224)
SEUIL = 0.60

st.title("🌿 Diagnostic des maladies foliaires — PlantVillage")
st.write("Uploadez une photo de feuille → diagnostic + confiance + conseil. Outil d'aide au dépistage, non un diagnostic définitif.")

file = st.file_uploader("Photo de la feuille", type=["jpg", "jpeg", "png", "webp"])
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, width=300)
    arr = np.expand_dims(np.array(image.resize(IMG_SIZE)).astype("float32"), 0)
    proba = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(proba)); conf = float(proba[idx]); label = class_names[idx]

    st.subheader("Diagnostic (top 3)")
    for i in np.argsort(proba)[-3:][::-1]:
        st.write(f"- **{class_names[i]}** : {proba[i]:.0%}")

    if conf < SEUIL:
        st.warning(f"⚠️ Diagnostic incertain ({conf:.0%}). Reprenez une photo nette ou consultez un expert.")
    elif "healthy" in label:
        st.success(f"✅ {label} — plante saine, aucun traitement nécessaire.")
    else:
        st.error(f"🩺 {label} détecté. Isolez la plante, retirez les feuilles atteintes, consultez un conseiller phytosanitaire.")