"""Interactive demo: upload a pet photo, see the predicted breed and *why* (Grad-CAM).

streamlit run demo/streamlit_app.py -- --checkpoint artifacts/resnet50/model.pt
"""

from __future__ import annotations

import argparse
import sys

import streamlit as st

from petcls.predict import Predictor

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="artifacts/resnet50/model.pt")
args, _ = parser.parse_known_args(sys.argv[1:])

st.set_page_config(page_title="Pet Breed Classifier", page_icon="🐾", layout="wide")
st.title("🐾 Pet breed classifier with explanations")
st.caption(
    "ResNet-50 fine-tuned on Oxford-IIIT Pet (37 breeds). The heatmap shows which pixels "
    "drove the prediction (Grad-CAM)."
)


@st.cache_resource
def load(path: str) -> Predictor:
    return Predictor(path)


predictor = load(args.checkpoint)
upload = st.file_uploader("Upload a cat or dog photo", type=["jpg", "jpeg", "png", "webp"])
if upload:
    img = Predictor.load_image(upload.getvalue())
    pred = predictor.predict(img, k=5)
    left, mid, right = st.columns(3)
    left.image(img, caption="Input", use_container_width=True)
    target = right.selectbox("Explain which class?", [p["label"] for p in pred.top_k])
    mid.image(
        predictor.explain(img, target), caption=f"Grad-CAM for '{target}'", use_container_width=True
    )
    right.metric("Prediction", pred.label, f"{pred.confidence:.1%} confidence")
    right.bar_chart({p["label"]: p["probability"] for p in pred.top_k}, horizontal=True)
    if pred.confidence < 0.5:
        st.warning(
            "Low confidence: this may be a mixed breed, an unusual angle, or not one of "
            "the 37 supported breeds."
        )
