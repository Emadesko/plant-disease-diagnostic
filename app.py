import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Diagnostic des feuilles", page_icon="🌿", layout="centered")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 560px;}
h1 {font-size: 1.7rem !important; font-weight: 800; color:#14321f;}
.sub {color:#5b6b60; margin:-6px 0 14px; font-size:.95rem;}
.tips {background:#e9f5ee; border-radius:14px; padding:14px 16px; margin-bottom:8px;
       color:#245c3a; font-size:.9rem; display:flex; gap:10px;}
.tips b{color:#14321f;}
.rcard{border-radius:18px; padding:18px; margin:8px 0;}
.amber{background:#fdf1dd;} .greenc{background:#e9f6ed;} .grayc{background:#eef1f0;}
.rhead{display:flex; gap:12px; align-items:flex-start;}
.ricon{width:40px; height:40px; border-radius:10px; display:flex; align-items:center;
       justify-content:center; font-size:1.2rem; color:#fff; flex:0 0 40px;}
.i-amber{background:#f59e0b;} .i-green{background:#22a45a;} .i-gray{background:#94a3b8;}
.rlabel{font-size:.72rem; letter-spacing:.08em; font-weight:700; color:#6b7280; text-transform:uppercase;}
.rtitle{font-size:1.35rem; font-weight:800; color:#14321f; line-height:1.15;}
.rsub{color:#5b6b60; font-size:.9rem;}
.confrow{display:flex; justify-content:space-between; margin:14px 0 6px; font-size:.9rem; color:#374151;}
.confrow b{font-size:1.05rem; color:#14321f;}
.bar{background:#e5e7eb; border-radius:9px; height:9px; overflow:hidden;}
.fill{height:9px; border-radius:9px;}
.f-amber{background:#f59e0b;} .f-green{background:#22a45a;} .f-gray{background:#94a3b8;}
.subcard{background:rgba(255,255,255,.75); border-radius:12px; padding:12px 14px; margin-top:14px;
         font-size:.92rem; color:#374151;}
.subcard b{color:#14321f;} .subcard ul{margin:.4rem 0 0; padding-left:1.1rem;} .subcard li{margin:.25rem 0;}
.chip{display:inline-block; background:#f1f5f4; color:#14321f; border-radius:11px;
      padding:7px 13px; margin:5px 6px 0 0; font-size:.88rem;} .chip b{color:#245c3a;}
.disc{background:#f5f7f6; border-radius:12px; padding:11px 14px; margin-top:12px;
      color:#6b7280; font-size:.82rem;}
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
st.markdown('<div class="tips">📷 <div><b>Conseils pour une bonne photo.</b> Cadrez une seule feuille '
            'bien visible, nette, bien éclairée, sur un fond simple.</div></div>', unsafe_allow_html=True)

tab_cam, tab_up = st.tabs(["Prendre une photo", "Uploader une photo"])
with tab_cam: img_cam = st.camera_input("Prendre la feuille en photo")
with tab_up:  img_up = st.file_uploader("Choisir une image", type=["jpg","jpeg","png","webp"])
fichier = img_cam or img_up

if fichier:
    image = Image.open(fichier).convert("RGB")
    st.image(image, caption="Aperçu de la feuille", width=260)
    if st.button("Analyser la photo", type="primary", use_container_width=True):
        arr = np.expand_dims(np.array(image.resize(IMG_SIZE)).astype("float32"), 0)
        proba = model.predict(arr, verbose=0)[0]
        o = np.argsort(proba)[::-1]
        best = int(o[0]); conf = float(proba[best]); label = class_names[best]
        autres = [(class_names[i], float(proba[i])) for i in o[1:3]]
        pct = f"{conf*100:.0f}%"

        if conf >= SEUIL and "healthy" in label:
            st.markdown(f'''<div class="rcard greenc"><div class="rhead">
              <div class="ricon i-green">✓</div><div>
              <div class="rlabel">Plante saine</div><div class="rtitle">Plante saine — {espece(label)}</div>
              <div class="rsub">Aucun signe de maladie détecté.</div></div></div>
              <div class="confrow"><span>Indice de confiance</span><b>{pct}</b></div>
              <div class="bar"><div class="fill f-green" style="width:{pct}"></div></div>
              <div class="subcard">Continuez la surveillance habituelle et gardez un bon espacement pour l'aération du feuillage.</div>
              </div>''', unsafe_allow_html=True)
            st.markdown("**Cela ressemble aussi à…**"); st.markdown(chips(autres), unsafe_allow_html=True)

        elif conf >= SEUIL:
            st.markdown(f'''<div class="rcard amber"><div class="rhead">
              <div class="ricon i-amber">⚠</div><div>
              <div class="rlabel">Maladie détectée</div><div class="rtitle">{joli(label)}</div>
              <div class="rsub">Culture : {espece(label)}</div></div></div>
              <div class="confrow"><span>Indice de confiance</span><b>{pct}</b></div>
              <div class="bar"><div class="fill f-amber" style="width:{pct}"></div></div>
              <div class="subcard"><b>Que faire maintenant ?</b><ul>
              <li>Isolez la plante des cultures voisines.</li>
              <li>Retirez et détruisez les feuilles atteintes.</li>
              <li>Consultez un conseiller pour un traitement adapté.</li></ul></div>
              </div>''', unsafe_allow_html=True)
            st.markdown("**Cela ressemble aussi à…**"); st.markdown(chips(autres), unsafe_allow_html=True)

        else:
            st.markdown(f'''<div class="rcard grayc"><div class="rhead">
              <div class="ricon i-gray">?</div><div>
              <div class="rlabel">Diagnostic incertain</div><div class="rtitle">Reprenez une photo plus nette</div>
              <div class="rsub">Aucune hypothèse n'atteint 60 % de confiance. Une meilleure photo affinera le résultat.</div></div></div>
              <div style="margin-top:12px;"><span class="chip">Confiance la plus élevée <b>{pct}</b></span></div>
              </div>''', unsafe_allow_html=True)
            st.markdown("**Hypothèses les plus proches**")
            st.markdown(chips([(label, conf)] + autres), unsafe_allow_html=True)
            st.markdown('<div class="tips">📷 <div><b>Astuce photo.</b> Approchez-vous d\'une seule feuille, '
                        'à la lumière du jour, sur un fond uni (main, papier, sol nu).</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="disc">ⓘ Outil d\'aide au dépistage — ce n\'est pas un diagnostic définitif. '
                'En cas de doute, consultez un conseiller agricole.</div>', unsafe_allow_html=True)