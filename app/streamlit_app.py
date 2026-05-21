"""
streamlit_app.py — Application web de détection d'incendie.

Lancement :
    cd app
    streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Ajouter le dossier app au path
sys.path.insert(0, os.path.dirname(__file__))
from prediction import load_model, predict, make_gradcam

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="🔥 Fire Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: 800;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center; color: #888; margin-bottom: 2rem;
    }
    .result-fire {
        background: linear-gradient(135deg, #ff4b2b, #ff416c);
        color: white; border-radius: 12px; padding: 1.5rem;
        text-align: center; font-size: 1.8rem; font-weight: 700;
    }
    .result-safe {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white; border-radius: 12px; padding: 1.5rem;
        text-align: center; font-size: 1.8rem; font-weight: 700;
    }
    .metric-box {
        background: #1e1e2e; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .stProgress > div > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/fire.png", width=80)
    st.title("⚙️ Configuration")

    st.subheader("Modèle")
    model_choice = st.selectbox(
        "Choisir le modèle",
        ["CNN Simple", "ResNet50", "EfficientNetB0"],
        index=2,
        help="EfficientNetB0 = meilleur compromis précision/vitesse"
    )

    model_files = {
        "CNN Simple":      "../cnn_best.keras",
        "ResNet50":        "../resnet_p2_best.keras",
        "EfficientNetB0":  "../efficientnet_best.keras",
    }

    st.subheader("Options")
    show_gradcam  = st.checkbox("Afficher Grad-CAM", value=True,
                                 help="Visualise les zones activées par le modèle")
    show_probs    = st.checkbox("Afficher les probabilités détaillées", value=True)
    threshold     = st.slider("Seuil de détection feu", 0.1, 0.9, 0.5, 0.05,
                               help="En dessous de ce seuil → Non-feu")

    st.divider()
    st.subheader("📊 Infos modèle")
    model_info = {
        "CNN Simple":     {"acc": "~92%", "params": "650K",  "speed": "⚡⚡⚡"},
        "ResNet50":       {"acc": "~97%", "params": "25.6M", "speed": "⚡"},
        "EfficientNetB0": {"acc": "~96%", "params": "5.3M",  "speed": "⚡⚡"},
    }
    info = model_info[model_choice]
    st.metric("Accuracy", info["acc"])
    st.metric("Paramètres", info["params"])
    st.write(f"Vitesse : {info['speed']}")

# ── Titre principal ────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔥 Détection d\'Incendie</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Classification d\'images par Deep Learning — CNN / ResNet50 / EfficientNetB0</div>',
            unsafe_allow_html=True)

# ── Chargement du modèle ──────────────────────────────────────────────────────
@st.cache_resource
def get_model(choice):
    path = model_files[choice]
    model = load_model(path)
    return model

model = get_model(model_choice)

if model is None:
    st.warning(
        f"⚠️ Modèle **{model_choice}** non trouvé (`{model_files[model_choice]}`).\n\n"
        "Entraînez d'abord le modèle avec le notebook final, "
        "ou placez le fichier `.keras` dans le dossier racine du projet.\n\n"
        "**Mode démo** : les prédictions seront simulées."
    )

# ── Zone d'upload ─────────────────────────────────────────────────────────────
st.subheader("📤 Charger une image")

tab1, tab2 = st.tabs(["📁 Uploader une image", "🖼️ Utiliser un exemple"])

with tab1:
    uploaded = st.file_uploader(
        "Glissez une image (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with tab2:
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    samples = []
    if os.path.exists(sample_dir):
        samples = [f for f in os.listdir(sample_dir)
                   if f.lower().endswith(('.jpg','.jpeg','.png'))]

    if samples:
        selected = st.selectbox("Choisir un exemple", samples)
        if st.button("Utiliser cet exemple"):
            uploaded = open(os.path.join(sample_dir, selected), "rb")
    else:
        st.info("Ajoutez des images dans `app/sample_images/` pour les tester ici.")

# ── Prédiction ────────────────────────────────────────────────────────────────
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    st.divider()
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🖼️ Image originale")
        st.image(image, use_container_width=True)
        st.caption(f"Taille : {image.size[0]}×{image.size[1]} px")

    with col2:
        st.subheader("🤖 Résultat de l'analyse")

        with st.spinner("Analyse en cours..."):
            if model is not None:
                result = predict(model, image)
                # Appliquer le seuil personnalisé
                is_fire = result["fire_prob"] / 100 >= threshold
                label = "Feu 🔥" if is_fire else "Non-feu ✅"
                confidence = result["fire_prob"] if is_fire else result["safe_prob"]
            else:
                # Mode démo — simulation
                import random
                fire_prob = random.uniform(0.1, 0.9)
                is_fire = fire_prob >= threshold
                label = "Feu 🔥" if is_fire else "Non-feu ✅"
                confidence = fire_prob * 100 if is_fire else (1 - fire_prob) * 100
                result = {
                    "fire_prob": round(fire_prob * 100, 2),
                    "safe_prob": round((1 - fire_prob) * 100, 2),
                }

        # Afficher le résultat
        css_class = "result-fire" if is_fire else "result-safe"
        st.markdown(f'<div class="{css_class}">{label}<br><small>{confidence:.1f}% confiance</small></div>',
                    unsafe_allow_html=True)

        st.write("")

        if show_probs:
            st.write("**Probabilités détaillées :**")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("🔥 Feu", f"{result['fire_prob']}%")
                st.progress(int(result["fire_prob"]))
            with c2:
                st.metric("✅ Non-feu", f"{result['safe_prob']}%")
                st.progress(int(result["safe_prob"]))

        st.write(f"*Seuil appliqué : {threshold}*")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    if show_gradcam and model is not None:
        st.divider()
        st.subheader("🔍 Grad-CAM — Zones d'attention du modèle")

        with st.spinner("Calcul Grad-CAM..."):
            heatmap, overlay = make_gradcam(model, image)

        if overlay is not None:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(image.resize((128,128)), caption="Original", use_container_width=True)
            with c2:
                st.image(heatmap, caption="Heatmap Grad-CAM", use_container_width=True)
            with c3:
                st.image(overlay, caption="Superposition", use_container_width=True)
            st.caption("🔴 Rouge = zones très activées | 🔵 Bleu = zones peu activées")
        else:
            st.info("Grad-CAM non disponible pour ce modèle (transfer learning).")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#666; font-size:0.85rem;'>
    Projet Deep Learning — Fire Detection CNN &nbsp;|&nbsp;
    Dataset : 999 images &nbsp;|&nbsp;
    Modèles : CNN · ResNet50 · EfficientNetB0
</div>
""", unsafe_allow_html=True)
