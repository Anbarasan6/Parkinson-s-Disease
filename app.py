import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import io
import time

import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's Assessment",
    page_icon="🧠",
    layout="centered",
)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "step": 1,
    "clinical": {},
    "typing_text": "",
    "typing_start": None,
    "voice_bytes": None,
    "features": {},
    "prediction": None,
    "probability": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Feature order expected by the model ──────────────────────────────────────
FEATURE_COLS = [
    "age", "gender", "family_history", "symptom_duration_years",
    "updrs_score", "avg_pitch", "jitter", "shimmer", "voice_tremor",
    "speech_rate", "tremor_frequency", "tremor_amplitude",
    "finger_tap_speed", "gait_speed", "balance_score",
]

FEATURE_META = {
    "age":                      ("Clinical", "years",    "Patient age"),
    "gender":                   ("Clinical", "0=F / 1=M","Biological sex"),
    "family_history":           ("Clinical", "0/1",      "Family Parkinson's history"),
    "symptom_duration_years":   ("Clinical", "years",    "Duration of symptoms"),
    "updrs_score":              ("Clinical", "score",    "Unified Parkinson's Rating Scale"),
    "gait_speed":               ("Clinical", "m/s",      "Walking speed"),
    "balance_score":            ("Clinical", "0–100",    "Postural stability score"),
    "finger_tap_speed":         ("Motor",    "chars/s",  "Finger dexterity (typing test)"),
    "avg_pitch":                ("Voice",    "Hz",       "Mean fundamental frequency"),
    "jitter":                   ("Voice",    "%",        "Cycle-to-cycle F0 variation"),
    "shimmer":                  ("Voice",    "%",        "Cycle-to-cycle amplitude variation"),
    "voice_tremor":             ("Voice",    "ratio",    "Tremor energy in voice signal"),
    "speech_rate":              ("Voice",    "syll/s",   "Speaking rate"),
    "tremor_frequency":         ("Voice",    "Hz",       "Dominant tremor modulation frequency"),
    "tremor_amplitude":         ("Voice",    "ratio",    "Amplitude of tremor modulation"),
}

# ── Model loader (auto-trains from CSV if pkl files absent) ───────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    mp = os.path.join(base, "model.pkl")
    sp = os.path.join(base, "scaler.pkl")
    if os.path.exists(mp) and os.path.exists(sp):
        return joblib.load(mp), joblib.load(sp)

    csv = os.path.join(base, "Parkinsons_Clinical_Voice_Motor_1500.csv")
    if not os.path.exists(csv):
        return None, None

    df = pd.read_csv(csv)
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])
    X = df.drop("parkinson_status", axis=1)
    y = df["parkinson_status"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xsc = sc.fit_transform(X_tr)
    mdl = XGBClassifier(random_state=42, eval_metric="logloss")
    mdl.fit(Xsc, y_tr)
    joblib.dump(mdl, mp)
    joblib.dump(sc, sp)
    return mdl, sc


model, scaler = load_model()


def predict(features: dict):
    if model is None or scaler is None:
        return None, None
    row = pd.DataFrame([[features.get(c, 0.0) for c in FEATURE_COLS]], columns=FEATURE_COLS)
    row_sc = scaler.transform(row)
    pred = int(model.predict(row_sc)[0])
    prob = float(model.predict_proba(row_sc)[0][1])
    return pred, prob


# ── Shared UI helpers ─────────────────────────────────────────────────────────
def show_progress():
    step = st.session_state.step
    labels = ["1 Clinical", "2 Typing", "3 Voice", "4 Features", "5 Result"]
    cols = st.columns(5)
    for i, (col, lbl) in enumerate(zip(cols, labels)):
        with col:
            if i + 1 < step:
                st.markdown(
                    f"<div style='text-align:center;color:#22c55e;font-weight:600'>✔ {lbl}</div>",
                    unsafe_allow_html=True,
                )
            elif i + 1 == step:
                st.markdown(
                    f"<div style='text-align:center;color:#3b82f6;font-weight:700'>▶ {lbl}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:center;color:#9ca3af'>○ {lbl}</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("---")


def back_button(key="back"):
    if st.button("← Back", key=key):
        st.session_state.step -= 1
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Clinical Information Form
# ═══════════════════════════════════════════════════════════════════════════════
def page_clinical():
    st.title("🧠 Parkinson's Disease Detection")
    show_progress()
    st.subheader("Step 1 — Clinical Information")
    st.caption("Complete the patient's clinical profile, then press **Next**.")

    prev = st.session_state.clinical

    with st.form("clinical_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age (years)", 1, 120, int(prev.get("age", 60)), step=1
            )
            gender = st.selectbox(
                "Gender", ["Male", "Female"],
                index=0 if prev.get("gender", 1) == 1 else 1,
            )
            family_history = st.radio(
                "Family history of Parkinson's?", ["No", "Yes"],
                index=int(prev.get("family_history", 0)), horizontal=True,
            )
            symptom_duration = st.number_input(
                "Symptom duration (years)", 0, 50,
                int(prev.get("symptom_duration_years", 0)), step=1,
            )

        with col2:
            updrs_score = st.number_input(
                "UPDRS Score", 0.0, 200.0,
                float(prev.get("updrs_score", 20.0)), step=0.5,
                help="Unified Parkinson's Disease Rating Scale — 0 = no disability",
            )
            gait_speed = st.number_input(
                "Gait Speed (m/s)", 0.0, 3.0,
                float(prev.get("gait_speed", 1.2)), step=0.01,
                help="Normal walking speed ≈ 1.2–1.4 m/s",
            )
            balance_score = st.number_input(
                "Balance Score (0–100)", 0.0, 100.0,
                float(prev.get("balance_score", 80.0)), step=0.5,
            )

        submitted = st.form_submit_button("Next →", type="primary")

    if submitted:
        st.session_state.clinical = {
            "age": age,
            "gender": 1 if gender == "Male" else 0,
            "family_history": 1 if family_history == "Yes" else 0,
            "symptom_duration_years": symptom_duration,
            "updrs_score": updrs_score,
            "gait_speed": gait_speed,
            "balance_score": balance_score,
        }
        st.session_state.step = 2
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Typing Test
# ═══════════════════════════════════════════════════════════════════════════════
_TARGET = "The quick brown fox jumps over the lazy dog near the riverbank"


def page_typing():
    st.title("🧠 Parkinson's Disease Detection")
    show_progress()
    st.subheader("Step 2 — Motor Typing Test")
    st.caption("Type the sentence below as naturally as possible.")

    # Start timer on first render of this step
    if st.session_state.typing_start is None:
        st.session_state.typing_start = time.time()

    st.info(f"**Type this sentence:**\n\n_{_TARGET}_")

    # Live JS stopwatch (purely cosmetic display)
    components.html(
        """
        <div style="font-family:monospace;font-size:14px;padding:6px 14px;
                    background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
                    display:inline-block;margin-bottom:4px;">
          ⏱ Elapsed: <span id="t">0.0</span>s
        </div>
        <script>
          const s = Date.now();
          setInterval(() => {
            document.getElementById("t").innerText =
              ((Date.now() - s) / 1000).toFixed(1);
          }, 100);
        </script>
        """,
        height=55,
    )

    typed = st.text_area(
        "Your input",
        value=st.session_state.typing_text,
        height=80,
        placeholder="Start typing here…",
        label_visibility="collapsed",
        key="typing_area",
    )
    st.session_state.typing_text = typed

    # Live metrics while typing
    if typed:
        elapsed = time.time() - (st.session_state.typing_start or time.time())
        wpm = len(typed.split()) / max(elapsed / 60, 1e-4)
        cps = len(typed) / max(elapsed, 1e-4)
        correct = sum(a == b for a, b in zip(typed, _TARGET))
        acc = correct / max(len(typed), 1) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Speed (WPM)", f"{wpm:.1f}")
        m2.metric("Chars / sec", f"{cps:.2f}")
        m3.metric("Accuracy", f"{acc:.0f}%")

    col_back, _, col_next = st.columns([1, 3, 1])
    with col_back:
        back_button("typing_back")
    with col_next:
        if st.button("Next →", key="typing_next", type="primary"):
            elapsed = time.time() - (st.session_state.typing_start or time.time())
            typed_text = st.session_state.typing_text
            cps = len(typed_text) / max(elapsed, 1.0)
            wpm = len(typed_text.split()) / max(elapsed / 60, 1e-4)
            st.session_state.features["finger_tap_speed"] = round(cps, 4)
            st.session_state.features.setdefault("speech_rate", round(wpm / 40, 4))
            st.session_state.typing_start = None
            st.session_state.step = 3
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Voice Recorder
# ═══════════════════════════════════════════════════════════════════════════════
def _extract_voice(audio_bytes: bytes) -> dict:
    if not LIBROSA_AVAILABLE:
        return {}
    try:
        y_a, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)

        # Fundamental frequency
        f0, voiced, _ = librosa.pyin(
            y_a,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        f0v = f0[voiced] if voiced is not None else f0[~np.isnan(f0)]
        f0v = f0v[~np.isnan(f0v)]
        avg_pitch = float(np.mean(f0v)) if len(f0v) > 0 else 150.0

        # Jitter
        if len(f0v) > 1:
            periods = 1.0 / f0v
            jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100)
        else:
            jitter = 0.5

        # Shimmer
        hop = 256
        rms = librosa.feature.rms(y=y_a, hop_length=hop)[0]
        shimmer = (
            float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9) * 100)
            if len(rms) > 1 else 0.5
        )

        # Tremor (4–12 Hz AM band)
        rms_sr = sr / hop
        if len(rms) > 10:
            freqs = np.fft.rfftfreq(len(rms), d=1.0 / rms_sr)
            fft_r = np.abs(np.fft.rfft(rms - np.mean(rms)))
            band = (freqs >= 4) & (freqs <= 12)
            if band.any():
                bf, ba = freqs[band], fft_r[band]
                pk = np.argmax(ba)
                tremor_frequency = float(bf[pk])
                total = np.sum(fft_r) + 1e-9
                tremor_amplitude = float(ba[pk] / total)
                voice_tremor = float(np.sum(ba) / total)
            else:
                tremor_frequency, tremor_amplitude, voice_tremor = 6.0, 0.05, 0.1
        else:
            tremor_frequency, tremor_amplitude, voice_tremor = 6.0, 0.05, 0.1

        # Speech rate
        dur = librosa.get_duration(y=y_a, sr=sr)
        speech_rate = float(np.sum(voiced) / max(dur, 1.0)) if voiced is not None else 3.0

        return dict(
            avg_pitch=round(avg_pitch, 3),
            jitter=round(jitter, 4),
            shimmer=round(shimmer, 4),
            voice_tremor=round(voice_tremor, 4),
            speech_rate=round(speech_rate, 4),
            tremor_frequency=round(tremor_frequency, 4),
            tremor_amplitude=round(tremor_amplitude, 6),
        )
    except Exception as exc:
        st.warning(f"Voice analysis error: {exc}")
        return {}


def _manual_voice():
    st.markdown("**Enter voice features manually:**")
    c1, c2 = st.columns(2)
    with c1:
        avg_pitch = st.number_input("Avg Pitch (Hz)", 50.0, 500.0, 150.0, step=1.0)
        jitter = st.number_input("Jitter (%)", 0.0, 10.0, 0.5, step=0.01)
        shimmer = st.number_input("Shimmer (%)", 0.0, 10.0, 0.5, step=0.01)
        voice_tremor = st.number_input("Voice Tremor (0–1)", 0.0, 1.0, 0.1, step=0.01)
    with c2:
        speech_rate = st.number_input("Speech Rate (syll/s)", 0.0, 10.0, 3.5, step=0.1)
        tremor_freq = st.number_input("Tremor Frequency (Hz)", 0.0, 20.0, 6.0, step=0.1)
        tremor_amp = st.number_input("Tremor Amplitude", 0.0, 1.0, 0.05, step=0.001, format="%.4f")
    return dict(
        avg_pitch=avg_pitch, jitter=jitter, shimmer=shimmer,
        voice_tremor=voice_tremor, speech_rate=speech_rate,
        tremor_frequency=tremor_freq, tremor_amplitude=tremor_amp,
    )


def page_voice():
    st.title("🧠 Parkinson's Disease Assessment")
    show_progress()
    st.subheader("Step 3 — Voice Recording")
    st.caption("Record yourself saying the phrase below for 5–10 seconds.")

    st.info("**Say aloud:** _Aaah… The quick brown fox jumps over the lazy dog_")

    voice_features: dict = {}

    if AUDIO_AVAILABLE:
        st.markdown("**Press the microphone icon to record:**")
        audio_bytes = audio_recorder(
            pause_threshold=3.0,
            sample_rate=22050,
            text="",
            icon_size="2x",
            key="voice_recorder",
        )

        if audio_bytes:
            st.session_state.voice_bytes = audio_bytes
            st.audio(audio_bytes, format="audio/wav")
            with st.spinner("Analysing voice features…"):
                voice_features = _extract_voice(audio_bytes)
            if voice_features:
                st.success("✅ Voice analysed")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Pitch (Hz)", f"{voice_features['avg_pitch']:.1f}")
                c2.metric("Jitter (%)", f"{voice_features['jitter']:.3f}")
                c3.metric("Shimmer (%)", f"{voice_features['shimmer']:.3f}")
                c4.metric("Tremor", f"{voice_features['voice_tremor']:.3f}")
        elif st.session_state.voice_bytes:
            st.info("Using previous recording.")
            voice_features = _extract_voice(st.session_state.voice_bytes)
        else:
            st.warning("No recording yet — press the microphone button above.")
    else:
        st.warning(
            "⚠️ Microphone recorder unavailable. Enter values manually."
        )
        voice_features = _manual_voice()

    col_back, _, col_next = st.columns([1, 3, 1])
    with col_back:
        back_button("voice_back")
    with col_next:
        if st.button("Next →", key="voice_next", type="primary"):
            # Merge voice features (defaults if analysis failed)
            merged = {
                "avg_pitch": 150.0, "jitter": 0.5, "shimmer": 0.5,
                "voice_tremor": 0.1, "speech_rate": 3.5,
                "tremor_frequency": 6.0, "tremor_amplitude": 0.05,
            }
            merged.update(voice_features)
            st.session_state.features.update(merged)
            st.session_state.features.update(st.session_state.clinical)
            st.session_state.step = 4
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Feature Results Table
# ═══════════════════════════════════════════════════════════════════════════════
def page_features():
    st.title("🧠 Parkinson's Disease Assessment")
    show_progress()
    st.subheader("Step 4 — Extracted Feature Summary")
    st.caption("Review all features before running the prediction. You can go back to adjust any step.")

    feats = st.session_state.features

    rows = []
    for col in FEATURE_COLS:
        meta = FEATURE_META.get(col, ("–", "–", col))
        val = feats.get(col, "–")
        rows.append({
            "Category": meta[0],
            "Feature": col,
            "Value": f"{val:.4f}" if isinstance(val, float) else str(val),
            "Unit": meta[1],
            "Description": meta[2],
        })

    df_feat = pd.DataFrame(rows)

    def _highlight(row):
        palette = {"Clinical": "#eff6ff", "Motor": "#f0fdf4", "Voice": "#fdf4ff"}
        bg = palette.get(row["Category"], "#ffffff")
        return [f"background-color:{bg}"] * len(row)

    st.dataframe(
        df_feat.style.apply(_highlight, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart (normalised values)
    numeric = {k: v for k, v in feats.items() if isinstance(v, (int, float)) and k in FEATURE_COLS}
    if numeric:
        fig, ax = plt.subplots(figsize=(10, 5))
        keys = list(numeric.keys())
        vals = np.array(list(numeric.values()), dtype=float)
        vals_norm = (vals - vals.min()) / ((vals.max() - vals.min()) + 1e-9)
        cat_color = {
            "Clinical": "#3b82f6",
            "Motor": "#22c55e",
            "Voice": "#a855f7",
        }
        bar_colors = [cat_color.get(FEATURE_META.get(k, ("–",))[0], "#94a3b8") for k in keys]
        bars = ax.barh(keys, vals_norm, color=bar_colors)
        ax.set_xlabel("Normalised value (0–1)")
        ax.set_xlim(0, 1.15)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(color=c, label=l) for l, c in cat_color.items()],
            loc="lower right", fontsize=9,
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    col_back, _, col_next = st.columns([1, 3, 1])
    with col_back:
        back_button("feat_back")
    with col_next:
        if st.button("Run Prediction →", key="feat_next", type="primary"):
            pred, prob = predict(st.session_state.features)
            st.session_state.prediction = pred
            st.session_state.probability = prob
            st.session_state.step = 5
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Prediction Result + Real-time Adjustment
# ═══════════════════════════════════════════════════════════════════════════════
def page_result():
    st.title("🧠 Parkinson's Disease Assessment")
    show_progress()
    st.subheader("Step 5 — Prediction Result")

    pred = st.session_state.prediction
    prob = st.session_state.probability

    if pred is None:
        st.error("No prediction available — please complete all previous steps.")
        if st.button("← Start over"):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()
        return

    # ── Result banner ─────────────────────────────────────────────────────────
    if pred == 1:
        st.markdown(
            f"""<div style="background:#fef2f2;border:2px solid #ef4444;border-radius:14px;
                            padding:24px;text-align:center;margin-bottom:12px;">
              <h2 style="color:#dc2626;margin:0">🔴 Parkinson's Disease Detected</h2>
              <p style="font-size:20px;color:#7f1d1d;margin-top:10px">
                Risk probability: <strong>{prob*100:.1f}%</strong>
              </p></div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style="background:#f0fdf4;border:2px solid #22c55e;border-radius:14px;
                            padding:24px;text-align:center;margin-bottom:12px;">
              <h2 style="color:#16a34a;margin:0">🟢 Normal — No Parkinson's Detected</h2>
              <p style="font-size:20px;color:#14532d;margin-top:10px">
                Confidence: <strong>{(1-prob)*100:.1f}%</strong>
              </p></div>""",
            unsafe_allow_html=True,
        )

    # Probability bar
    fig, ax = plt.subplots(figsize=(9, 1.2))
    ax.barh([""], [1 - prob], color="#22c55e", height=0.5, label="Normal")
    ax.barh([""], [prob], left=[1 - prob], color="#ef4444", height=0.5, label="Parkinson's")
    ax.axvline(0.5, color="#374151", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # ── Real-time feature adjustment ──────────────────────────────────────────
    st.markdown("### 🎛 Real-time Feature Adjustment")
    st.caption(
        "Move any slider to instantly re-run the model — useful for exploring how each "
        "feature influences the prediction."
    )

    live = dict(st.session_state.features)

    with st.expander("🏥 Clinical Features", expanded=True):
        a1, a2 = st.columns(2)
        with a1:
            live["age"] = st.slider("Age", 1, 100, int(live.get("age", 60)))
            live["updrs_score"] = st.slider(
                "UPDRS Score", 0.0, 200.0, float(live.get("updrs_score", 20.0)), step=0.5
            )
            live["symptom_duration_years"] = st.slider(
                "Symptom Duration (yrs)", 0, 30, int(live.get("symptom_duration_years", 0))
            )
        with a2:
            live["gait_speed"] = st.slider(
                "Gait Speed (m/s)", 0.0, 3.0, float(live.get("gait_speed", 1.2)), step=0.01
            )
            live["balance_score"] = st.slider(
                "Balance Score", 0.0, 100.0, float(live.get("balance_score", 80.0)), step=0.5
            )
            live["family_history"] = int(
                st.selectbox(
                    "Family History",
                    [0, 1],
                    index=int(live.get("family_history", 0)),
                    format_func=lambda x: "Yes" if x else "No",
                    key="fh_live",
                )
            )

    with st.expander("🖐 Motor Features"):
        live["finger_tap_speed"] = st.slider(
            "Finger Tap Speed (chars/s)", 0.0, 15.0,
            float(live.get("finger_tap_speed", 5.0)), step=0.1,
        )

    with st.expander("🎙 Voice Features"):
        v1, v2 = st.columns(2)
        with v1:
            live["avg_pitch"] = st.slider(
                "Avg Pitch (Hz)", 50.0, 500.0, float(live.get("avg_pitch", 150.0)), step=1.0
            )
            live["jitter"] = st.slider(
                "Jitter (%)", 0.0, 5.0, float(live.get("jitter", 0.5)), step=0.01
            )
            live["shimmer"] = st.slider(
                "Shimmer (%)", 0.0, 5.0, float(live.get("shimmer", 0.5)), step=0.01
            )
            live["voice_tremor"] = st.slider(
                "Voice Tremor", 0.0, 1.0, float(live.get("voice_tremor", 0.1)), step=0.01
            )
        with v2:
            live["speech_rate"] = st.slider(
                "Speech Rate (syll/s)", 0.0, 10.0, float(live.get("speech_rate", 3.5)), step=0.1
            )
            live["tremor_frequency"] = st.slider(
                "Tremor Frequency (Hz)", 0.0, 20.0,
                float(live.get("tremor_frequency", 6.0)), step=0.1,
            )
            live["tremor_amplitude"] = st.slider(
                "Tremor Amplitude", 0.0, 1.0,
                float(live.get("tremor_amplitude", 0.05)), step=0.001, format="%.3f",
            )

    # Live re-prediction (runs on every slider move via Streamlit rerun)
    live_pred, live_prob = predict(live)

    st.markdown("#### 🔮 Live Prediction")
    if live_pred is not None:
        lc1, lc2 = st.columns([3, 1])
        with lc1:
            if live_pred == 1:
                st.error(f"🔴 **Parkinson's Detected** — risk {live_prob*100:.1f}%")
            else:
                st.success(f"🟢 **Normal** — confidence {(1-live_prob)*100:.1f}%")
        with lc2:
            delta_pct = (live_prob - prob) * 100
            st.metric(
                "Risk probability",
                f"{live_prob*100:.1f}%",
                delta=f"{delta_pct:+.1f}% vs original",
                delta_color="inverse",
            )

    st.markdown("---")
    rc1, rc2 = st.columns(2)
    with rc1:
        if st.button("← Start Over", key="restart"):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()
    with rc2:
        if st.button("💾 Save adjusted values as final", key="save_live"):
            st.session_state.features = live
            st.session_state.prediction = live_pred
            st.session_state.probability = live_prob
            st.success("Saved.")


# ── Router ────────────────────────────────────────────────────────────────────
{
    1: page_clinical,
    2: page_typing,
    3: page_voice,
    4: page_features,
    5: page_result,
}[st.session_state.step]()
