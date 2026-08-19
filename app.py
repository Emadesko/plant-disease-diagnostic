import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Diagnostic des feuilles", page_icon="🌿", layout="centered")

st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/remixicon@4.5.0/fonts/remixicon.css" rel="stylesheet">
<style>
.block-container {padding: 1.5rem 1.1rem 3rem; max-width: 560px;}
h1 {font-size: 1.7rem !important; font-weight: 800; color:#14321f; margin-bottom:.1rem;}
.sub {color:#5b6b60; margin:0 0 16px; font-size:.95rem;}
.card {border-radius:16px; padding:15px 16px; margin:10px 0; border:1px solid #dfe6e2;}
.tips {background:#e9f5ee; border-color:#c2e3d0; color:#245c3a; font-size:.9rem;
       display:flex; gap:11px; align-items:flex-start;}
.tips b{color:#14321f;}
.amber{background:#fdf1dd; border-color:#f2d49a;}
.greenc{background:#e9f6ed; border-color:#bfe3cd;}
.grayc{background:#eef1f0; border-color:#d6ddda;}
.rhead{display:flex; gap:12px; align-items:flex-start;}
.ricon{width:42px; height:42px; border-radius:11px; display:flex; align-items:center;
       justify-content:center; font-size:1.3rem; color:#fff; flex:0 0 42px;}
.i-amber{background:#f59e0b;} .i-green{background:#22a45a;} .i-gray{background:#9aa8a0;}
.i-tip{background:#22a45a; width:34px; height:34px; border-radius:9px; font-size:1.05rem; flex:0 0 34px;}
.rlabel{font-size:.72rem; letter-spacing:.07em; font-weight:700; color:#7a8a80; text-transform:uppercase;}
.rtitle{font-size:1.35rem; font-weight:800; color:#14321f; line-height:1.15;}
.rsub{color:#5b6b60; font-size:.9rem; margin-top:2px;}
.confrow{display:flex; justify-content:space-between; margin:15px 0 6px; font-size:.9rem; color:#374151;}
.confrow b{font-size:1.05rem; color:#14321f;}
.bar{background:#e5e7eb; border-radius:9px; height:9px; overflow:hidden;}
.fill{height:9px; border-radius:9px;}
.f-amber{background:#f59e0b;} .f-green{background:#22a45a;} .f-gray{background:#9aa8a0;}
.subcard{background:#fff; border:1px solid rgba(0,0,0,.06); border-radius:12px; padding:12px 14px;
         margin-top:14px; font-size:.92rem; color:#374151;}
.subcard b{color:#14321f;} .subcard ul{margin:.45rem 0 0; padding-left:1.1rem;} .subcard li{margin:.28rem 0;}
.sec{font-weight:700; color:#14321f; margin:16px 0 2px; font-size:.95rem;}
.chip{display:inline-block; background:#f1f5f4; border:1px solid #e0e8e4; color:#14321f; border-radius:11px;
      padding:7px 13px; margin:6px 6px 0 0; font-size:.88rem;} .chip b{color:#245c3a;}
.disc{background:#f5f7f6; border:1px solid #e6ebe8; border-radius:12px; padding:11px 14px; margin-top:14px;
      color:#6b7280; font-size:.82rem; display:flex; gap:9px; align-items:flex-start;}
button[kind="primary"]{background:#22a45a !important; border:none !important;}
button[kind="primary"]:hover{background:#1c8f4e !important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load():
    return tf.keras.models.load_model("plantvillage_mobilenet_finetuned.keras"), json.load(open("class_names.json"))

model, class_names = load()
IMG_SIZE = (224, 224); SEUIL = 0.60
def joli(n): return n.replace("___", " — ").replace("_", " ")
def espece(n): return joli(n.split("___")[0])
def chips(items): return "".join(f'<span class="chip">{joli(n)} <b>{p:.0%}</b></span>' for n, p in items)

st.markdown("<h1>Diagnostic des maladies foliaires</h1>", unsafe_allow_html=True)
st.markdown('<div class="sub">Aide au dépistage — ce n\'est pas un diagnostic définitif.</div>', unsafe_allow_html=True)
st.markdown('<div class="card tips"><div class="ricon i-tip"><i class="ri-camera-line"></i></div>'
            '<div><b>Conseils pour une bonne photo.</b> Cadrez une seule feuille bien visible, nette, '
            'bien éclairée, sur un fond simple.</div></div>', unsafe_allow_html=True)

st.markdown('<div class="sec">Ajouter une photo de feuille</div>', unsafe_allow_html=True)
mode = st.segmented_control("mode", ["Prendre une photo", "Uploader une photo"],
                            default="Prendre une photo", label_visibility="collapsed")
if mode == "Uploader une photo":
    fichier = st.file_uploader("img", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
else:
    fichier = st.camera_input("img", label_visibility="collapsed")

if fichier:
    image = Image.open(fichier).convert("RGB")
    st.image(image, use_container_width=True)
    if st.button("Analyser la photo", type="primary", use_container_width=True):
        arr = np.expand_dims(np.array(image.resize(IMG_SIZE)).astype("float32"), 0)
        proba = model.predict(arr, verbose=0)[0]
        o = np.argsort(proba)[::-1]
        best = int(o[0]); conf = float(proba[best]); label = class_names[best]
        autres = [(class_names[i], float(proba[i])) for i in o[1:3]]
        pct = f"{conf*100:.0f}%"

        if conf >= SEUIL and "healthy" in label:
            st.markdown(f'''<div class="card greenc"><div class="rhead">
              <div class="ricon i-green"><i class="ri-checkbox-circle-line"></i></div><div>
              <div class="rlabel">Plante saine</div><div class="rtitle">Plante saine — {espece(label)}</div>
              <div class="rsub">Aucun signe de maladie détecté.</div></div></div>
              <div class="confrow"><span>Indice de confiance</span><b>{pct}</b></div>
              <div class="bar"><div class="fill f-green" style="width:{pct}"></div></div>
              <div class="subcard">Continuez la surveillance habituelle et gardez un bon espacement pour l'aération du feuillage.</div></div>''', unsafe_allow_html=True)
            st.markdown('<div class="sec">Cela ressemble aussi à…</div>', unsafe_allow_html=True)
            st.markdown(chips(autres), unsafe_allow_html=True)
        elif conf >= SEUIL:
            st.markdown(f'''<div class="card amber"><div class="rhead">
              <div class="ricon i-amber"><i class="ri-error-warning-line"></i></div><div>
              <div class="rlabel">Maladie détectée</div><div class="rtitle">{joli(label)}</div>
              <div class="rsub">Culture : {espece(label)}</div></div></div>
              <div class="confrow"><span>Indice de confiance</span><b>{pct}</b></div>
              <div class="bar"><div class="fill f-amber" style="width:{pct}"></div></div>
              <div class="subcard"><b>Que faire maintenant ?</b><ul>
              <li>Isolez la plante des cultures voisines.</li>
              <li>Retirez et détruisez les feuilles atteintes.</li>
              <li>Consultez un conseiller pour un traitement adapté.</li></ul></div></div>''', unsafe_allow_html=True)
            st.markdown('<div class="sec">Cela ressemble aussi à…</div>', unsafe_allow_html=True)
            st.markdown(chips(autres), unsafe_allow_html=True)
        else:
            st.markdown(f'''<div class="card grayc"><div class="rhead">
              <div class="ricon i-gray"><i class="ri-question-line"></i></div><div>
              <div class="rlabel">Diagnostic incertain</div><div class="rtitle">Reprenez une photo plus nette</div>
              <div class="rsub">Aucune hypothèse n'atteint 60 % de confiance. Une meilleure photo affinera le résultat.</div></div></div>
              <div style="margin-top:12px;"><span class="chip">Confiance la plus élevée <b>{pct}</b></span></div></div>''', unsafe_allow_html=True)
            st.markdown('<div class="sec">Hypothèses les plus proches</div>', unsafe_allow_html=True)
            st.markdown(chips([(label, conf)] + autres), unsafe_allow_html=True)
            st.markdown('<div class="card tips"><div class="ricon i-tip"><i class="ri-camera-line"></i></div>'
                        '<div><b>Astuce photo.</b> Approchez-vous d\'une seule feuille, à la lumière du jour, '
                        'sur un fond uni (main, papier, sol nu).</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="disc"><i class="ri-information-line"></i><div>Outil d\'aide au dépistage — '
                'ce n\'est pas un diagnostic définitif. En cas de doute, consultez un conseiller agricole.</div></div>',
                unsafe_allow_html=True)