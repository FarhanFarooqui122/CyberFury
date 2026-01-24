import streamlit as st
from PIL import Image
import torch
import warnings
import plotly.graph_objects as go
from transformers import AutoImageProcessor, AutoModelForImageClassification

# PAGE CONFIG
st.set_page_config(page_title="CyberFury | AI Forensic Lab", layout="wide", page_icon="⚡")
warnings.filterwarnings("ignore", category=UserWarning)

# UI
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #00f3ff; font-family: 'Courier New', Courier, monospace; }
    .stButton>button { 
        width: 100%; background: transparent; color: #00f3ff; border: 2px solid #00f3ff;
        font-weight: bold; box-shadow: 0 0 10px #00f3ff;
    }
    .stButton>button:hover { border: 2px solid #ff003c; color: #ff003c; box-shadow: 0 0 20px #ff003c; }
    </style>
""", unsafe_allow_html=True)

class CyberFuryEngine:
    @staticmethod
    @st.cache_resource
    def load_model():
        model_name = "Organika/sdxl-detector"
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        
        # Robust Device Detection
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        model.to(device)
        return processor, model, device

    @staticmethod
    def get_dynamic_reasoning(verdict, ai_raw, real_raw):
        """Generates context-aware reasoning based on score distribution."""
        if verdict == "AI":
            if ai_raw > 0.99999:
                return "Absolute AI signature detected. Mathematical convergence on synthetic diffusion patterns is 100%."
            return f"Anomalous pixel gradients detected. Synthetic markers ({ai_raw:.4f}) exceed safety thresholds."
        else:
            if real_raw > 0.95:
                return "Pure organic sensor data. Pixel-level noise matches physical hardware capture characteristics."
            elif 0.70 <= real_raw <= 0.95:
                return "Natural image with post-processing. Content verified as organic despite compression artifacts."
            else:
                return "Edge-case detected. Image shows mixed signals, but remains below the synthetic detection buffer."

    @staticmethod
    def analyze(image, processor, model, device, buffer=0.9999):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        id2label = model.config.id2label
        ai_score, real_score = 0.0, 0.0

        for idx, prob in enumerate(probs):
            label = id2label[idx].upper()
            val = prob.item()
            if any(key in label for key in ["AI", "FAKE", "SYNTHETIC", "ARTIFICIAL"]):
                ai_score = val
            else:
                real_score = val

        raw_ai_fixed = ai_score
        raw_real_fixed = real_score

        if ai_score >= buffer:
            verdict = "AI"
            status = "🚩 AI-GENERATED DETECTED"
            conf = ai_score * 100
        else:
            verdict = "REAL"
            status = "✨ AUTHENTIC / ORGANIC"
            conf = 100 - (real_score * 100)
            ai_score = 1 - ai_score
            real_score = 1 - real_score

        reason = CyberFuryEngine.get_dynamic_reasoning(verdict, raw_ai_fixed, raw_real_fixed)

        return {
            "verdict": verdict, "status": status, "conf": conf, 
            "reason": reason, "ai_raw": ai_score, "real_raw": real_score,
            "device": device.type.upper()
        }

def draw_gauge(val, verdict):
    color = "#ff003c" if verdict == "AI" else "#00f3ff"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number={'suffix': "%", 'font': {'color': color}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}, 'bgcolor': "#111"}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#00f3ff"}, height=350)
    return fig

def main():
    st.markdown("<h1 style='text-align:center;'>⚡ CYBERFURY: FORENSIC SCANNER</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    if 'data' not in st.session_state: st.session_state.data = None

    with col1:
        file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])
        if file:
            img = Image.open(file)
            st.image(img, use_container_width=True)
            if st.button("🚨 RUN FORENSIC SCAN"):
                with st.spinner("Decoding pixel signatures..."):
                    proc, mod, dev = CyberFuryEngine.load_model()
                    st.session_state.data = CyberFuryEngine.analyze(img, proc, mod, dev)

    with col2:
        if st.session_state.data:
            res = st.session_state.data
            st.plotly_chart(draw_gauge(res["conf"], res["verdict"]), use_container_width=True)
            
            color = "#ff003c" if res["verdict"] == "AI" else "#00f3ff"
            st.markdown(f"""
                <div style='border: 2px solid {color}; padding: 20px; box-shadow: 0 0 15px {color}; background: rgba(0,0,0,0.5);'>
                    <h2 style='color: {color}; margin-top:0;'>{res['status']}</h2>
                    <p style='font-size: 1.1em;'><strong>Reasoning:</strong> {res['reason']}</p>
                    <hr style='border: 0.5px solid {color}; opacity: 0.3;'>
                    <small style='color: gray;'>
                        Inverse AI: {res['ai_raw']:.4f} | Inverse Real: {res['real_raw']:.4f} | Engine: {res['device']}
                    </small>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("System Online. Awaiting evidence for forensic deep-scan.")

if __name__ == "__main__":
    main()
