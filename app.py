# =============================================================================
# Network Attack Forecasting — Streamlit Dashboard
# =============================================================================
"""
Cybersecurity command-center dashboard for the AI-based network attack
forecasting platform.  Provides six views: Data Input, Network Overview,
Attack Forecast, Attack Progression, Explainability, and Flagged Behaviour.
"""

import os, sys, io, tempfile
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.express as px
import streamlit as st

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.helpers import load_config, set_seed, get_device, load_json
from src.data.dataset_loader import load_csv_dataset
from src.data.feature_extractor import FeatureExtractor
from src.data.preprocessor import Preprocessor
from src.data.time_windowing import TimeWindowGenerator
from src.attack_mapping.mitre_mapper import MitreMapper, ATTACK_STAGES


def _load_model_and_align(X_seq, config):
    """
    Load the trained world model and align the current data dimensions
    to match what the model was trained on.  Returns (model, aligned_X_seq, feature_names).
    """
    import torch
    from src.models.world_model import NetworkWorldModel

    model_path = os.path.join(_ROOT, "models", "saved", "best_world_model.pt")
    meta_path = os.path.join(_ROOT, "models", "saved", "feature_meta.json")
    has_trained = os.path.isfile(model_path)

    # Determine input_dim: prefer saved metadata over current data
    if has_trained and os.path.isfile(meta_path):
        meta = load_json(meta_path)
        train_dim = meta["input_dim"]
        train_features = meta.get("feature_columns", [])
    else:
        train_dim = X_seq.shape[2]
        train_features = []

    model = NetworkWorldModel.from_config(train_dim, config)

    if has_trained:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Align data dimensions: pad or trim to match train_dim
    current_dim = X_seq.shape[2]
    if current_dim < train_dim:
        pad = np.zeros((X_seq.shape[0], X_seq.shape[1], train_dim - current_dim), dtype=np.float32)
        X_seq = np.concatenate([X_seq, pad], axis=2)
    elif current_dim > train_dim:
        X_seq = X_seq[:, :, :train_dim]

    return model, X_seq, train_features, has_trained


def get_sih_risk_level(infiltration_prob: float) -> str:
    """Map infiltration probability to a risk label based on SIH thresholds (0-100% scale)."""
    p_pct = infiltration_prob * 100.0
    if p_pct <= 20.0:
        return "LOW"
    elif p_pct <= 50.0:
        return "MODERATE"
    elif p_pct <= 75.0:
        return "HIGH"
    else:
        return "CRITICAL"

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NetForecast AI", page_icon="🛡️", layout="wide")

CUSTOM_CSS = """
<style>
* { font-family: Arial, Helvetica, sans-serif; }
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%); color: #e2e8f0; }
[data-testid="stSidebar"] { background: #0d1321; border-right: 1px solid #1e293b; }
[data-testid="stHeader"] { background: transparent; }
.metric-card { background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; border-radius: 12px; padding: 20px; margin: 6px 0; }
.metric-card h3 { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 8px 0; }
.metric-card .value { font-size: 2rem; font-weight: 700; margin: 0; }
.risk-low { color: #22c55e; } .risk-medium { color: #eab308; }
.risk-high { color: #f97316; } .risk-critical { color: #ef4444; }
.stage-badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin: 2px 4px; }
.section-title { color: #e2e8f0; font-size: 1.3rem; font-weight: 600; margin: 24px 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #334155; }
div[data-testid="stMetric"] { background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; border-radius: 12px; padding: 16px; }
div[data-testid="stMetric"] label { color: #94a3b8 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: #1e293b; border-radius: 8px 8px 0 0; color: #94a3b8; border: 1px solid #334155; }
.stTabs [aria-selected="true"] { background: #0ea5e9 !important; color: #fff !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Arial"), margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _ss(): return st.session_state
def _init_state():
    for k, v in dict(df=None, metadata=None, scaled=None, states=None,
                     labels_atk=None, labels_bin=None, win_ts=None,
                     X_seq=None, y_state=None, y_atk=None, y_bin=None,
                     forecast_results=None, explain_results=None,
                     feature_cols=None, preprocessor=None, config=None,
                     analysis_done=False).items():
        if k not in st.session_state:
            st.session_state[k] = v
_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ NetForecast AI")
    st.markdown("*Temporal World Model for Network Attack Forecasting*")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📥 Data Input", "📊 Network Overview", "🔮 Attack Forecast",
        "⚔️ Attack Progression", "🧠 Explainability", "🚩 Flagged Behaviour",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("v1.0 · Runs fully offline")

# =========================================================================
# PAGE 1 — DATA INPUT
# =========================================================================
if page == "📥 Data Input":
    st.markdown("# 📥 Network Data Input")
    st.markdown("Upload a CSV network traffic dataset or use the built-in synthetic demo data.")

    col1, col2 = st.columns([2, 1])
    with col1:
        upload = st.file_uploader("Upload CSV", type=["csv"], help="CIC-IDS, CTU-13, or generic CSV")
    with col2:
        use_demo = st.button("🧪 Use Synthetic Demo Data", use_container_width=True)
        k_steps = st.slider("Forecast Steps (K)", 1, 10, 5)
        window_size = st.slider("Window Size (rows)", 5, 100, 20)

    if use_demo or upload:
        with st.spinner("Processing data..."):
            try:
                config = load_config(os.path.join(_ROOT, "config", "config.yaml"))
                config["windowing"]["window_size_seconds"] = window_size
                config["forecasting"]["default_k_steps"] = k_steps
                st.session_state.config = config
                set_seed(42)

                if use_demo:
                    demo_path = os.path.join(_ROOT, "data", "sample", "synthetic_traffic.csv")
                    if not os.path.isfile(demo_path):
                        from data.generate_synthetic import generate_synthetic_traffic
                        generate_synthetic_traffic(n_records=8000, output_path=demo_path)
                    csv_path = demo_path
                else:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                    tmp.write(upload.read()); tmp.close()
                    csv_path = tmp.name

                df, meta = load_csv_dataset(csv_path, config)
                extractor = FeatureExtractor(config)
                df, feat_meta = extractor.extract(df)
                preprocessor = Preprocessor(config)
                save_dir = config.get("training", {}).get("save_dir", "models/saved")
                scaler_path = os.path.join(_ROOT, save_dir, "scaler.pkl")
                if os.path.isfile(scaler_path):
                    preprocessor.load(os.path.join(_ROOT, save_dir))
                    df = preprocessor.parse_timestamps(df)
                    df = preprocessor.map_labels(df, config)
                    df, scaled = preprocessor.transform(df)
                else:
                    df = preprocessor.parse_timestamps(df)
                    df = preprocessor.map_labels(df, config)
                    df, scaled = preprocessor.fit_transform(df)
                feature_cols = preprocessor.feature_columns

                windower = TimeWindowGenerator(config)
                states, l_atk, l_bin, w_ts = windower.create_windows(df, scaled, feature_cols)
                X_seq, y_st, y_atk, y_bin = windower.create_sequences(states, l_atk, l_bin)

                # Store in session
                for k, v in dict(df=df, metadata=meta, scaled=scaled, states=states,
                                 labels_atk=l_atk, labels_bin=l_bin, win_ts=w_ts,
                                 X_seq=X_seq, y_state=y_st, y_atk=y_atk, y_bin=y_bin,
                                 feature_cols=feature_cols, preprocessor=preprocessor,
                                 analysis_done=True, forecast_results=None,
                                 explain_results=None).items():
                    st.session_state[k] = v

                st.success(f"✅ Data processed: {len(df)} flows → {len(states)} windows → {len(X_seq)} sequences")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Flows", f"{len(df):,}")
                c2.metric("Time Windows", f"{len(states):,}")
                c3.metric("Features", f"{len(feature_cols)}")
                c4.metric("Schema", meta.get("schema", "unknown"))

                st.markdown("**Label Distribution:**")
                lbl_counts = df["attack_stage"].value_counts().sort_index()
                mapper = MitreMapper(config)
                lbl_df = pd.DataFrame({
                    "Stage": [mapper.get_stage_name(i) for i in lbl_counts.index],
                    "Count": lbl_counts.values,
                    "Percentage": (lbl_counts.values / len(df) * 100).round(1),
                })
                st.dataframe(lbl_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback; st.code(traceback.format_exc())

# =========================================================================
# PAGE 2 — NETWORK OVERVIEW
# =========================================================================
elif page == "📊 Network Overview":
    st.markdown("# 📊 Network Overview")
    if not st.session_state.analysis_done:
        st.warning("⚠️ Upload data on the Data Input page first."); st.stop()

    df = st.session_state.df
    states = st.session_state.states
    l_bin = st.session_state.labels_bin
    config = st.session_state.config
    mapper = MitreMapper(config)

    suspicious = int(l_bin.sum())
    risk = suspicious / max(len(l_bin), 1)
    risk_lbl = "CRITICAL" if risk > 0.5 else "HIGH" if risk > 0.3 else "MEDIUM" if risk > 0.15 else "LOW"
    risk_cls = f"risk-{risk_lbl.lower()}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Flows", f"{len(df):,}")
    c2.metric("Suspicious Windows", f"{suspicious} / {len(l_bin)}")
    c3.metric("Current Risk", risk_lbl)
    c4.metric("Attack Ratio", f"{risk:.1%}")

    # Traffic volume over time (by window index)
    st.markdown('<div class="section-title">Traffic Volume Over Time</div>', unsafe_allow_html=True)
    if "total_bytes" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=states[:, 0] if states.shape[1] > 0 else np.zeros(len(states)),
            mode="lines", name="Feature 0 (scaled)",
            line=dict(color="#0ea5e9", width=2),
            fill="tozeroy", fillcolor="rgba(14,165,233,0.1)",
        ))
        # Highlight suspicious windows
        sus_idx = np.where(l_bin > 0)[0]
        if len(sus_idx) > 0:
            fig.add_trace(go.Scatter(
                x=sus_idx, y=states[sus_idx, 0] if states.shape[1] > 0 else np.zeros(len(sus_idx)),
                mode="markers", name="Suspicious", marker=dict(color="#ef4444", size=6),
            ))
        fig.update_layout(**PLOT_LAYOUT, title="Network State Feature (Window-level)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Protocol & Port distribution
    col_a, col_b = st.columns(2)
    with col_a:
        if "protocol" in df.columns:
            proto = df["protocol"].value_counts().head(10)
            proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
            fig = go.Figure(go.Bar(
                x=[proto_map.get(p, str(p)) for p in proto.index],
                y=proto.values, marker_color="#0ea5e9",
            ))
            fig.update_layout(**PLOT_LAYOUT, title="Protocol Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        if "dst_port" in df.columns:
            ports = pd.to_numeric(df["dst_port"], errors="coerce").dropna().astype(int)
            top_ports = ports.value_counts().head(10)
            fig = go.Figure(go.Bar(
                x=top_ports.index.astype(str), y=top_ports.values,
                marker_color="#8b5cf6",
            ))
            fig.update_layout(**PLOT_LAYOUT, title="Top Destination Ports", height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Attack stage distribution across windows
    st.markdown('<div class="section-title">Attack Stage Distribution (Windows)</div>', unsafe_allow_html=True)
    l_atk = st.session_state.labels_atk
    stage_counts = pd.Series(l_atk).value_counts().sort_index()
    colors = [mapper.get_stage_color(i) for i in stage_counts.index]
    fig = go.Figure(go.Bar(
        x=[mapper.get_stage_name(i) for i in stage_counts.index],
        y=stage_counts.values, marker_color=colors,
    ))
    fig.update_layout(**PLOT_LAYOUT, title="Windows per Attack Stage", height=300)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# PAGE 3 — ATTACK FORECAST
# =========================================================================
elif page == "🔮 Attack Forecast":
    st.markdown("# 🔮 Attack Forecast — K-Step Forward Simulation")
    if not st.session_state.analysis_done:
        st.warning("⚠️ Upload data first."); st.stop()

    config = st.session_state.config
    states = st.session_state.states
    X_seq = st.session_state.X_seq
    feature_cols = st.session_state.feature_cols
    mapper = MitreMapper(config)
    k = config.get("forecasting", {}).get("default_k_steps", 5)

    st.info("ℹ️ Running world model inference for K-step forecast...")

    # Load the trained World Model. Never present an untrained network as a prediction.
    try:
        from src.models.forecasting_engine import ForecastingEngine

        model, X_seq, train_features, has_trained = _load_model_and_align(X_seq, config)
        if not has_trained:
            st.error("No trained World Model checkpoint found at models/saved/best_world_model.pt. Train the model before running forecast inference.")
            st.stop()

        st.success("Trained GRU World Model loaded successfully.")
        engine = ForecastingEngine(model, config)
        # Use the last available sequence for forecasting
        last_seq = X_seq[-1]  # (seq_len, D)
        results = engine.forecast(last_seq, k_steps=k)
        st.session_state.forecast_results = results

        # Infiltration probability chart
        steps = [f"T+{r.step}" for r in results]
        probs = [r.infiltration_prob * 100 for r in results]
        stages = [mapper.get_stage_name(r.attack_stage) for r in results]
        
        # Calculate confidence using the decay factor configuration to be consistent with the progression tab
        decay_factor = config.get("forecasting", {}).get("confidence_decay", 0.95)
        confs = [max(0.0, min(100.0, (decay_factor ** r.step) * 100.0)) for r in results]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps, y=probs, mode="lines+markers+text",
            text=[f"{p:.1f}%" for p in probs], textposition="top center",
            textfont=dict(color="#e2e8f0", size=14),
            line=dict(color="#ef4444", width=3),
            marker=dict(size=12, color=[
                "#22c55e" if p <= 20 else "#eab308" if p <= 50 else "#f97316" if p <= 75 else "#ef4444"
                for p in probs
            ], line=dict(color="#fff", width=2)),
            name="Attack Probability",
        ))
        # Threshold lines matching the SIH risk levels
        for thr, clr, lbl in [(20, "#22c55e", "Low"), (50, "#eab308", "Moderate"), (75, "#f97316", "High")]:
            fig.add_hline(y=thr, line_dash="dot", line_color=clr, opacity=0.4,
                          annotation_text=lbl, annotation_position="right")
        fig.update_layout(**PLOT_LAYOUT, title="Attack Probability Forecast",
                          yaxis_title="Probability (%)", yaxis_range=[0, 105], height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        forecast_df = pd.DataFrame({
            "Forecast Step": steps,
            "Predicted Stage": stages,
            "Attack Probability": [f"{p:.1f}%" for p in probs],
            "Risk Level": [engine.get_risk_level(r.infiltration_prob) for r in results],
            "Model Confidence": [f"{c:.0f}%" for c in confs],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Forecast error: {e}")
        import traceback; st.code(traceback.format_exc())

# =========================================================================
# PAGE 4 — ATTACK PROGRESSION
# =========================================================================
elif page == "⚔️ Attack Progression":
    st.markdown("# ⚔️ Attack Progression Visualisation")
    if not st.session_state.analysis_done:
        st.warning("⚠️ Upload data first."); st.stop()

    config = st.session_state.config
    mapper = MitreMapper(config)
    l_atk = st.session_state.labels_atk
    results = st.session_state.forecast_results

    # Current observed stage (most recent window)
    current_stage = int(l_atk[-1]) if len(l_atk) > 0 else 0
    current_info = mapper.get_stage_info(current_stage)

    st.markdown("### 🔍 Current Observed State")
    if current_stage == 0:
        st.markdown(f"**NORMAL** 🟢")
        st.markdown("*No confirmed compromise currently detected.*")
    else:
        st.markdown(f"**{current_info['name'].upper()}** {current_info['icon']}")
        st.markdown(f"*{current_info['description']}*")
    st.markdown("---")

    if results:
        st.markdown("### Predicted Attack Progression")
        # Build progression: current → forecasted stages
        all_stages = [current_stage] + [r.attack_stage for r in results]
        all_labels = ["Current (Observed)"] + [f"T+{r.step} Forecast" for r in results]
        all_probs = [0] + [r.infiltration_prob for r in results]

        cols = st.columns(len(all_stages))
        for i, (stage_id, label, prob) in enumerate(zip(all_stages, all_labels, all_probs)):
            info = mapper.get_stage_info(stage_id)
            with cols[i]:
                if i == 0:
                    border = "2px solid #0ea5e9"
                    badge_clr = "#0ea5e9"
                    badge = "OBSERVED"
                    st.markdown(f"""<div style="background:#1e293b; border:{border}; border-radius:12px; padding:16px; text-align:center; min-height:220px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
<span style="background:{badge_clr}; color:#fff; padding:2px 10px; border-radius:10px; font-size:0.7rem; font-weight:600; text-transform:uppercase;">{badge}</span>
<div style="font-size:2.5rem; margin:12px 0;">{info['icon']}</div>
<div style="font-weight:700; font-size:1.1rem; color:#f1f5f9; text-transform:uppercase;">{info['name']}</div>
<div style="color:#94a3b8; font-size:0.8rem; margin-top:8px; font-weight:500;">{label}</div>
</div>""", unsafe_allow_html=True)
                else:
                    r = results[i-1]
                    border = f"2px solid {info['color']}"
                    badge_clr = info["color"]
                    badge = "FORECAST"
                    risk_lvl = get_sih_risk_level(r.infiltration_prob)
                    
                    # Risk level color matching
                    risk_colors = {
                        "LOW": "#22c55e",
                        "MODERATE": "#eab308",
                        "HIGH": "#f97316",
                        "CRITICAL": "#ef4444"
                    }
                    risk_clr = risk_colors.get(risk_lvl, "#94a3b8")
                    
                    # Confidence decay calculation
                    decay_factor = config.get("forecasting", {}).get("confidence_decay", 0.95)
                    decayed_conf = (decay_factor ** r.step) * 100.0
                    decayed_conf = max(0.0, min(100.0, decayed_conf))
                    
                    st.markdown(f"""<div style="background:#1e293b; border:{border}; border-radius:12px; padding:16px; text-align:center; min-height:220px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
<span style="background:{badge_clr}; color:#fff; padding:2px 10px; border-radius:10px; font-size:0.7rem; font-weight:600; text-transform:uppercase;">{badge}</span>
<div style="font-size:2.5rem; margin:12px 0;">{info['icon']}</div>
<div style="font-weight:700; font-size:1.1rem; color:#f1f5f9; text-transform:uppercase;">{info['name']}</div>
<div style="color:#94a3b8; font-size:0.8rem; margin-top:4px; font-weight:500; margin-bottom:12px;">{label}</div>
<div style="border-top:1px solid #334155; padding-top:8px; margin-top:8px; text-align:left;">
<div style="color:#94a3b8; font-size:0.75rem;">Attack Probability:</div>
<div style="color:#f1f5f9; font-weight:600; font-size:0.95rem;">{r.infiltration_prob:.1%}</div>
<div style="color:#94a3b8; font-size:0.75rem; margin-top:6px;">Risk Level:</div>
<div style="color:{risk_clr}; font-weight:700; font-size:0.95rem; text-transform:uppercase;">{risk_lvl}</div>
<div style="color:#94a3b8; font-size:0.75rem; margin-top:6px;">Confidence:</div>
<div style="color:#94a3b8; font-weight:500; font-size:0.9rem;">{decayed_conf:.0f}%</div>
</div>
</div>""", unsafe_allow_html=True)

        # Progression arrow timeline
        st.markdown("---")
        if results and results[-1].attack_stage > current_stage:
            pred_stage_name = mapper.get_stage_name(results[-1].attack_stage)
            st.warning(
                f"⚠️ **ESCALATION FORECAST**: Model predicts progression from "
                f"**{current_info['name']}** to **{pred_stage_name}** "
                f"within {len(results)} time windows."
            )

        # Narrative
        if results:
            narrative = mapper.generate_risk_narrative(
                current_stage, results[-1].attack_stage, results[-1].infiltration_prob
            )
            st.markdown("### Risk Narrative")
            st.code(narrative, language="text")
    else:
        st.info("Run the Attack Forecast page first to generate predictions.")

# =========================================================================
# PAGE 5 — EXPLAINABILITY
# =========================================================================
elif page == "🧠 Explainability":
    st.markdown("# 🧠 Prediction Explainability")
    if not st.session_state.analysis_done:
        st.warning("⚠️ Upload data first."); st.stop()

    config = st.session_state.config
    X_seq = st.session_state.X_seq
    feature_cols = st.session_state.feature_cols

    st.info("Computing feature attribution via perturbation analysis...")

    try:
        from src.explainability.explainer import Explainer

        model, X_seq, train_features, has_trained = _load_model_and_align(X_seq, config)
        if not has_trained:
            st.error("No trained World Model checkpoint found. Train the model before generating explanations.")
            st.stop()
        if train_features:
            feature_cols = train_features

        # Get forecasted stage (at T+1)
        forecast_results = st.session_state.get("forecast_results", [])
        pred_stage_id = forecast_results[0].attack_stage if forecast_results else 0
        mapper = MitreMapper(config)

        explainer = Explainer(model, feature_cols, config)
        last_seq = X_seq[-1]
        explanation = explainer.explain(last_seq, predicted_stage=pred_stage_id)
        st.session_state.explain_results = explanation

        st.markdown("### Top Risk-Driving Features")
        top = explanation["top_features"]

        # Horizontal bar chart
        feat_names = [f[0] for f in top]
        feat_vals = [f[1] for f in top]
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in feat_vals]

        fig = go.Figure(go.Bar(
            x=feat_vals, y=feat_names, orientation="h",
            marker_color=colors, text=[f"{v:+.2e}" for v in feat_vals],
            textposition="outside", textfont=dict(color="#e2e8f0"),
        ))
        fig.update_layout(**PLOT_LAYOUT, title="Feature Importance (Infiltration Risk)",
                          xaxis_title="Importance", height=max(300, len(top) * 35 + 100))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table with dynamic label for top driver and top mitigator
        pred_stage_name = mapper.get_stage_name(pred_stage_id).upper()
        directions = []
        
        # Find index of the first positive value (top risk driver)
        first_pos_idx = next((i for i, v in enumerate(feat_vals) if v > 0), -1)
        # Find index of the first negative value (top risk mitigator)
        first_neg_idx = next((i for i, v in enumerate(feat_vals) if v < 0), -1)
        
        for idx, v in enumerate(feat_vals):
            if v > 0:
                if idx == first_pos_idx:
                    directions.append(f"🔥 Top Driver for {pred_stage_name} (↑ Increased Risk)")
                else:
                    directions.append("↑ Increased Risk")
            else:
                if idx == first_neg_idx:
                    directions.append(f"🛡️ Top Mitigator for {pred_stage_name} (↓ Reduced Risk)")
                else:
                    directions.append("↓ Reduced Risk")

        imp_df = pd.DataFrame({
            "Feature": feat_names, "Importance": [f"{v:+.2e}" for v in feat_vals],
            "Direction": directions,
        })
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

        # Narrative
        st.markdown("### AI Explanation")
        st.markdown(f"*{explanation['narrative']}*")
        st.caption(f"Method: {explanation['method']} | Top {len(top)} features shown | Attribution values are relative perturbation effects, not probabilities")

    except Exception as e:
        st.error(f"Explainability error: {e}")
        import traceback; st.code(traceback.format_exc())

# =========================================================================
# PAGE 6 — FLAGGED BEHAVIOUR
# =========================================================================
elif page == "🚩 Flagged Behaviour":
    st.markdown("# 🚩 Flagged Network Behaviour")
    if not st.session_state.analysis_done:
        st.warning("⚠️ Upload data first."); st.stop()

    df = st.session_state.df
    config = st.session_state.config
    mapper = MitreMapper(config)

    # Build flagged table from attack flows
    flagged = df[df["is_attack"] == 1].copy()

    if len(flagged) == 0:
        st.info("No suspicious flows detected in the dataset.")
        st.stop()

    # Build a clean display table.
    # IMPORTANT: attack_stage is the observed/ground-truth stage stored in the
    # uploaded dataset. It is NOT a per-flow model prediction.
    display_cols = []
    for col in ["timestamp", "src_ip", "dst_ip", "dst_port", "protocol", "label"]:
        if col in flagged.columns:
            display_cols.append(col)

    flagged_display = flagged[display_cols].copy()

    # Show the dataset's attack stage with an explicit name so the dashboard
    # does not incorrectly claim that these values came from model inference.
    if "attack_stage" in flagged.columns:
        flagged_display["Observed Stage"] = flagged["attack_stage"].map(
            lambda x: mapper.get_stage_name(int(x))
        )

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        if "Observed Stage" in flagged_display.columns:
            stage_filter = st.multiselect(
                "Filter by Observed Stage",
                options=flagged_display["Observed Stage"].dropna().unique().tolist(),
                default=flagged_display["Observed Stage"].dropna().unique().tolist(),
            )
            flagged_display = flagged_display[flagged_display["Observed Stage"].isin(stage_filter)]
    with col2:
        if "protocol" in flagged_display.columns:
            proto_filter = st.multiselect(
                "Filter by Protocol",
                options=flagged_display["protocol"].unique().tolist(),
                default=flagged_display["protocol"].unique().tolist(),
            )
            flagged_display = flagged_display[flagged_display["protocol"].isin(proto_filter)]

    st.caption(
        "Observed Stage is derived from the attack_stage label in the uploaded dataset. "
        "Model-generated future stages are shown in the Attack Forecast and Attack Progression pages."
    )

    st.metric("Flagged Flows", f"{len(flagged_display):,}")
    st.dataframe(flagged_display.head(500), use_container_width=True, hide_index=True)
    st.caption(f"Showing up to 500 of {len(flagged_display)} flagged flows.")
