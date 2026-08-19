import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Diagnostic des feuilles", page_icon="🌿", layout="centered")

# ---------- Style ----------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 640px;}
.big-card {border-radius: 16px; padding: 18px 20px; margin: 10px 0; color: #fff;}
.green {background: linear-gradient(135deg,#16a34a,#22c55e);}
.orange{background: linear-gradient(135deg,#d97706,#f59e0b);}
.gray  {background: linear-gradient(135deg,#475569,#64748b);}
.big-card h2 {margin:0; font-size:1.4rem;}
.big-card p {margin:.4rem 0 0; font-size:1rem; opacity:.95;}
.chip {display:inline-block; background:#f1f5f9; color:#0f172a; border-radius:10px;
       padding:6px 12px; margin:4px 6px 0 0; font-size:.9rem;}
.tips {background:#ecfdf5; border-left:4px solid #16a34a; padding:12px 14px;
       border-radius:8px; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load():
    model = tf.keras.models.load_model("plantvillage_mobilenet_finetuned.keras")
    class_names = json.load(open("class_names.json"))
    return model, class_names

model, class_names = load()
IMG_SIZE = (224, 224)
SEUIL = 0.60

def joli(nom):
    return nom.replace("___", " — ").replace("_", " ")

def card(css, titre, texte):
    st.markdown(f'<div class="big-card {css}"><h2>{titre}</h2><p>{texte}</p></div>', unsafe_allow_html=True)

# ---------- En-tête ----------
st.title("🌿 Diagnostic des maladies foliaires")
st.caption("Aide au dépistage de première ligne — ce n'est pas un diagnostic définitif.")
st.markdown('<div class="tips">📸 <b>Pour un meilleur résultat :</b> cadrez <b>une seule feuille bien en évidence</b>, '
            'nette, bien éclairée, sur un fond simple. Évitez les photos floues ou de loin.</div>', unsafe_allow_html=True)

# ---------- Entrée : photo ou upload ----------
tab_cam, tab_up = st.tabs(["📷 Prendre une photo", "📤 Uploader une photo"])
with tab_cam:
    img_cam = st.camera_input("Prenez la feuille en photo")
with tab_up:
    img_up = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png", "webp"])

fichier = img_cam or img_up

# ---------- Prédiction ----------
if fichier:
    image = Image.open(fichier).convert("RGB")
    st.image(image, caption="Image analysée", width=280)

    arr = np.expand_dims(np.array(image.resize(IMG_SIZE)).astype("float32"), 0)
    proba = model.predict(arr, verbose=0)[0]

    ordre = np.argsort(proba)[::-1]
    best = int(ordre[0]); conf = float(proba[best]); label = class_names[best]
    autres = [(class_names[i], float(proba[i])) for i in ordre[1:3]]

    st.write("")
    if conf >= SEUIL:
        if "healthy" in label:
            card("green", "✅ Plante saine", f"Aucun signe de maladie détecté — confiance {conf:.0%}.")
        else:
            card("orange", f"🩺 {joli(label)}",
                 f"Maladie détectée (confiance {conf:.0%}). Isolez la plante, retirez les feuilles atteintes "
                 f"et rapprochez-vous d'un conseiller phytosanitaire.")
        st.markdown("**Ça ressemble aussi un peu à :**")
        st.markdown("".join(f'<span class="chip">{joli(n)} · {p:.0%}</span>' for n, p in autres), unsafe_allow_html=True)
    else:
        card("gray", "🤔 Diagnostic incertain",
             f"Je ne suis pas assez sûr pour trancher (meilleure hypothèse {conf:.0%}). "
             f"Reprenez une photo plus nette, feuille bien cadrée et éclairée.")
        st.markdown("**Cela pourrait ressembler à :**")
        cands = [(label, conf)] + autres
        st.markdown("".join(f'<span class="chip">{joli(n)} · {p:.0%}</span>' for n, p in cands), unsafe_allow_html=True)
        st.info("💡 Astuce : approchez-vous de la feuille, sur fond clair, sans flou → le diagnostic sera bien plus fiable.")