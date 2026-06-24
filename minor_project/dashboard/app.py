import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.alerts import run_all_alerts
import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

from models.latent_ode import LatentODE

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Neural ODE Health Monitor", layout="wide")

# ── Load model once ──────────────────────────────────────────
@st.cache_resource
def load_model():
    model = LatentODE(input_dim=1, hidden_dim=64)
    model.load_state_dict(torch.load(
        "neural_ode_model.pth", map_location="cpu"
    ))
    model.eval()
    return model

model = load_model()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Prediction",
    "Signal Monitor",
    "Heart Rate Monitor",
    "ICU Monitor",        
    "ECG Heatmap",
    "Latent Space",
    "Alert System",
    "Model Performance",
    "System Info"
])


st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload ECG CSV", type="csv")

# ── Parse uploaded file ───────────────────────────────────────
signal_data = None
filename = None

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    data = df.select_dtypes(include=[np.number]).values.astype(float)
    if data.shape[1] >= 1:
        signal_raw = data[:500, 0]  # first 500 timesteps, first column
        signal_norm = (signal_raw - signal_raw.mean()) / (signal_raw.std() + 1e-5)
        signal_data = signal_norm
        filename = uploaded_file.name

def run_prediction(signal):
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    t = torch.linspace(0, 1, x.shape[1])
    
    import time
    start = time.time()
    with torch.no_grad():
        logits = model(x, t)
        probs = torch.softmax(logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item() * 100
    inference_time = (time.time() - start) * 1000  # in ms
    
    return pred, confidence, probs.numpy(), inference_time

# ══════════════════════════════════════════════════════════════
# PREDICTION PAGE
# ══════════════════════════════════════════════════════════════
if page == "Prediction":
    st.title("ECG Classification")

    if signal_data is None:
        st.info("Upload an ECG CSV file from the sidebar to begin.")
    else:
        pred, confidence, probs, inf_time = run_prediction(signal_data)
        label = "Normal" if pred == 0 else "Abnormal"

        st.subheader(f"File: `{filename}`")

        col1, col2, col3,col4 = st.columns(4)
        col1.metric("Prediction", label)
        col2.metric("Confidence", f"{confidence:.1f}%")
        col4.metric("Inference Time", f"{inf_time:.1f} ms")

        if pred == 0:
            st.success("✅ ECG classified as Normal")
        else:
            st.error("⚠️ ECG classified as Abnormal — Review recommended")

        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
    "Class": ["Normal", "Abnormal"],
    "Probability": [probs[0], probs[1]]
})
        st.bar_chart(prob_df.set_index("Class"))

        st.subheader("ECG Signal Preview")
        st.line_chart(pd.DataFrame({"ECG Signal": signal_data}))

# ══════════════════════════════════════════════════════════════
# SIGNAL MONITOR PAGE
# ══════════════════════════════════════════════════════════════
elif page == "Signal Monitor":
    st.title("Signal Analysis")

    if signal_data is None:
        st.info("Upload an ECG CSV file from the sidebar.")
    else:
        signal = signal_data
        st.line_chart(pd.DataFrame({"ECG": signal}))

        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{signal.mean():.4f}")
        col2.metric("Std Dev", f"{signal.std():.4f}")
        col3.metric("Max Amplitude", f"{np.max(np.abs(signal)):.4f}")

        # Heart rate estimate
        peaks, _ = find_peaks(signal, distance=30)
        sampling_rate = 250
        duration = len(signal) / sampling_rate
        heart_rate = int((len(peaks) / duration) * 60) if duration > 0 else 0

        col4, col5 = st.columns(2)
        col4.metric("Detected R-Peaks", len(peaks))
        if 30 < heart_rate < 220:
            col5.metric("Estimated Heart Rate", f"{heart_rate} BPM")
        else:
            col5.metric("Estimated Heart Rate", "Unreliable")

        # Alerts
        st.subheader("Signal Alerts")
        alerts = []
        if signal.std() > 0.5:
            alerts.append("High variance detected")
        if np.max(np.abs(np.diff(signal))) > 1.5:
            alerts.append("Sudden spike detected")
        anomalies = np.where(signal > signal.mean() + 2 * signal.std())[0]
        if len(anomalies) > 0:
            alerts.append(f"{len(anomalies)} anomalous points detected")

        if alerts:
            for a in alerts:
                st.warning(a)
        else:
            st.success("Signal stable — no alerts")

# ══════════════════════════════════════════════════════════════
# ECG HEATMAP PAGE
# ══════════════════════════════════════════════════════════════
elif page == "ECG Heatmap":
    st.title("ECG Anomaly Heatmap")

    if signal_data is None:
        st.info("Upload an ECG CSV file from the sidebar.")
    else:
        signal = signal_data
        anomaly_score = np.abs(signal - signal.mean())
        heat = anomaly_score.reshape(1, -1)

        fig, ax = plt.subplots(figsize=(14, 2))
        sns.heatmap(heat, cmap="Reds", ax=ax, cbar=True, xticklabels=False, yticklabels=False)
        ax.set_title("Anomaly Intensity Along ECG Signal")
        st.pyplot(fig)

        st.subheader("Anomaly Score Over Time")
        st.line_chart(pd.DataFrame({"Anomaly Score": anomaly_score}))

# ══════════════════════════════════════════════════════════════
# LATENT SPACE PAGE
# ══════════════════════════════════════════════════════════════
elif page == "Latent Space":
    st.title("Latent Space Visualization")
    st.caption("Latent trajectory extracted from GRU encoder hidden states")

    if signal_data is None:
        st.info("Upload an ECG CSV file from the sidebar.")
    else:
        signal = signal_data
        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            # Extract hidden states at each timestep
            gru_out, _ = model.encoder(x)
            hidden_states = gru_out.squeeze(0).numpy()  # (T, hidden_dim)

        # PCA to 2D for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(hidden_states)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            reduced[:, 0], reduced[:, 1],
            c=np.arange(len(reduced)),
            cmap="viridis", s=5
        )
        plt.colorbar(scatter, ax=ax, label="Timestep")
        ax.set_title("Latent Space Trajectory (PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)

        st.caption("Color gradient shows progression through time. Tight clusters indicate stable signal regions.")

# ══════════════════════════════════════════════════════════════
# MODEL PERFORMANCE PAGE
# ══════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Evaluation Results")
    st.caption("Results from evaluation on MIT-BIH test set (7548 samples, balanced)")

    from datasets.ecg_loader import ECGDataset
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    import numpy as np

    @st.cache_data
    def get_eval_results():
        dataset = ECGDataset("data/mitbih_kaggle/mitbih_test.csv")
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                t = torch.linspace(0, 1, x.shape[1])
                output = model(x, t)
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.numpy())
                all_labels.extend(y.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        return all_preds, all_labels

    with st.spinner("Running evaluation on test set..."):
        all_preds, all_labels = get_eval_results()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    report = classification_report(all_labels, all_preds,
                                   target_names=['Normal', 'Abnormal'],
                                   labels=[0, 1], output_dict=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("F1 Score", f"{f1:.4f}")
    col3.metric("Test Samples", "7,548")

    st.subheader("Per-Class Performance")
    perf_df = pd.DataFrame({
        "Class": ["Normal", "Abnormal"],
        "Precision": [report['Normal']['precision'], report['Abnormal']['precision']],
        "Recall":    [report['Normal']['recall'],    report['Abnormal']['recall']],
        "F1":        [report['Normal']['f1-score'],  report['Abnormal']['f1-score']]
    })
    st.dataframe(perf_df, use_container_width=True)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred Normal", "Pred Abnormal"],
                yticklabels=["True Normal", "True Abnormal"])
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
# ══════════════════════════════════════════════════════════════
# ALERT SYSTEM PAGE
# ══════════════════════════════════════════════════════════════
elif page == "Alert System":
    st.title("Smart Alert System")
    st.caption("Automated detection of spikes, flatlines, and irregular rhythms")

    if signal_data is None:
        st.info("Upload an ECG CSV file from the sidebar.")
    else:
        signal = signal_data
        alerts = run_all_alerts(signal)

        # Summary metrics
        critical = [a for a in alerts if a["severity"] == "CRITICAL"]
        high = [a for a in alerts if a["severity"] == "HIGH"]
        medium = [a for a in alerts if a["severity"] == "MEDIUM"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Alerts", len(alerts))
        col2.metric("🔴 Critical", len(critical))
        col3.metric("🟠 High", len(high))
        col4.metric("🟡 Medium", len(medium))

        st.markdown("---")

        if not alerts:
            st.success("✅ No alerts detected. Signal appears normal.")
        else:
            st.subheader("Active Alerts")
            for alert in alerts:
                if alert["severity"] == "CRITICAL":
                    st.error(f"🔴 **{alert['type']}** — {alert['message']}")
                elif alert["severity"] == "HIGH":
                    st.warning(f"🟠 **{alert['type']}** — {alert['message']}")
                else:
                    st.info(f"🟡 **{alert['type']}** — {alert['message']}")

        # Signal with spike overlay
        st.subheader("Signal with Anomaly Overlay")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(signal, color="steelblue", linewidth=0.8, label="ECG Signal")

        # Mark spike locations
        spike_alerts = [a for a in alerts if a["type"] == "SPIKE"]
        if spike_alerts:
            spike_idx = spike_alerts[0]["indices"]
            ax.scatter(spike_idx, signal[spike_idx],
                      color="red", s=20, zorder=5, label="Spikes")

        # Mark flatline regions
        flat_alerts = [a for a in alerts if a["type"] == "FLATLINE"]
        if flat_alerts:
            flat_idx = flat_alerts[0]["indices"]
            ax.axvspan(flat_idx[0], flat_idx[-1],
                      alpha=0.3, color="orange", label="Flatline")

        ax.set_title("ECG Signal — Anomaly Markers")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)

        # Alert summary table
        if alerts:
            st.subheader("Alert Summary")
            alert_df = pd.DataFrame([{
                "Type": a["type"],
                "Severity": a["severity"],
                "Message": a["message"]
            } for a in alerts])
            st.dataframe(alert_df, use_container_width=True)

            # Download alert report
            csv = alert_df.to_csv(index=False)
            st.download_button(
                "Download Alert Report",
                csv,
                "alert_report.csv",
                "text/csv"
            )
# ══════════════════════════════════════════════════════════════
# HEART RATE MONITOR PAGE
# ══════════════════════════════════════════════════════════════
elif page == "Heart Rate Monitor":
    st.title("Heart Rate Monitor")
    st.caption("Real BPM calculation from raw MIT-BIH ECG recordings")

    st.info("Upload a MIT-BIH .hea file (the .dat file must be in the same folder)")
    hea_file = st.file_uploader("Upload .hea file", type="hea", key="hea")

    import tempfile, shutil, wfdb

    if hea_file:
        # We need both .hea and .dat — save them to a temp directory
        st.warning("Also upload the matching .dat file below")
        dat_file = st.file_uploader("Upload matching .dat file", type="dat", key="dat")

        if dat_file:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save both files to temp dir
                rec_name = hea_file.name.replace(".hea", "")
                hea_path = os.path.join(tmpdir, hea_file.name)
                dat_path = os.path.join(tmpdir, dat_file.name)

                with open(hea_path, "wb") as f:
                    f.write(hea_file.read())
                with open(dat_path, "wb") as f:
                    f.write(dat_file.read())

                # Load record
                record = wfdb.rdrecord(os.path.join(tmpdir, rec_name))
                signal = record.p_signal[:, 0]  # first lead
                fs = record.fs  # sampling frequency (360 Hz for MIT-BIH)

                st.success(f"Loaded record: {rec_name} | Sampling rate: {fs} Hz | Duration: {len(signal)/fs:.1f}s")

                # Peak detection
                from scipy.signal import find_peaks
                min_distance = int(0.4 * fs)  # minimum 400ms between beats
                peaks, properties = find_peaks(
                    signal,
                    distance=min_distance,
                    height=np.mean(signal) + 0.3 * np.std(signal)
                )

                # BPM calculation
                if len(peaks) > 1:
                    rr_intervals = np.diff(peaks) / fs  # in seconds
                    mean_rr = np.mean(rr_intervals)
                    bpm = 60 / mean_rr
                    bpm_std = np.std(60 / rr_intervals)

                    # Classify
                    if bpm < 60:
                        rhythm = "Bradycardia"
                        color = "warning"
                    elif bpm > 100:
                        rhythm = "Tachycardia"
                        color = "error"
                    else:
                        rhythm = "Normal Sinus Rhythm"
                        color = "success"

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Heart Rate", f"{bpm:.1f} BPM")
                    col2.metric("Rhythm", rhythm)
                    col3.metric("R-Peaks Detected", len(peaks))
                    col4.metric("BPM Variability (±)", f"{bpm_std:.1f}")

                    if color == "success":
                        st.success(f"✅ {rhythm} — Heart rate within normal range")
                    elif color == "warning":
                        st.warning(f"⚠️ {rhythm} — Heart rate below 60 BPM")
                    else:
                        st.error(f"🔴 {rhythm} — Heart rate above 100 BPM")

                    # RR interval plot
                    st.subheader("RR Interval Trend")
                    rr_ms = rr_intervals * 1000
                    st.line_chart(pd.DataFrame({"RR Interval (ms)": rr_ms}))

                    # Signal with peaks — show first 10 seconds
                    st.subheader("ECG Signal with R-Peaks (first 10 seconds)")
                    show_samples = int(10 * fs)
                    sig_snippet = signal[:show_samples]
                    peaks_snippet = peaks[peaks < show_samples]

                    fig, ax = plt.subplots(figsize=(14, 4))
                    ax.plot(sig_snippet, color="steelblue", linewidth=0.8, label="ECG")
                    ax.scatter(peaks_snippet, sig_snippet[peaks_snippet],
                              color="red", s=40, zorder=5, label="R-Peaks")
                    ax.set_title(f"ECG Signal — {rec_name} (first 10s)")
                    ax.set_xlabel("Sample")
                    ax.set_ylabel("Amplitude (mV)")
                    ax.legend()
                    st.pyplot(fig)

                    # BPM over time (rolling window)
                    st.subheader("BPM Over Time")
                    window = 10  # beats per window
                    if len(peaks) > window:
                        bpm_over_time = []
                        for i in range(len(peaks) - window):
                            rr_win = np.diff(peaks[i:i+window]) / fs
                            bpm_over_time.append(60 / np.mean(rr_win))
                        st.line_chart(pd.DataFrame({"BPM": bpm_over_time}))
                else:
                    st.error("Not enough peaks detected. Try a different record.")
# ══════════════════════════════════════════════════════════════
# ICU MONITOR PAGE
# ══════════════════════════════════════════════════════════════
elif page == "ICU Monitor":
    st.title("ICU Real-Time ECG Monitor")
    st.caption("Simulated real-time ECG playback at original sampling rate")

    import wfdb, tempfile, time

    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        hea_icu = st.file_uploader("Upload .hea file", type="hea", key="icu_hea")
    with col_upload2:
        dat_icu = st.file_uploader("Upload .dat file", type="dat", key="icu_dat")

    WINDOW_SIZE = 360  # 1 second of data at 360Hz

    if hea_icu and dat_icu:
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_name = hea_icu.name.replace(".hea", "")
            with open(os.path.join(tmpdir, hea_icu.name), "wb") as f:
                f.write(hea_icu.read())
            with open(os.path.join(tmpdir, dat_icu.name), "wb") as f:
                f.write(dat_icu.read())

            record = wfdb.rdrecord(os.path.join(tmpdir, rec_name))
            signal = record.p_signal[:, 0]
            fs = record.fs

        # Controls
        col1, col2, col3 = st.columns(3)
        speed = col1.selectbox("Playback Speed", ["0.5x", "1x", "2x"], index=1)
        duration = col2.selectbox("Duration (seconds)", [10, 30, 60], index=0)
        start_btn = col3.button("▶ Start Monitoring")

        speed_map = {"0.5x": 2.0, "1x": 1.0, "2x": 0.5}
        delay = speed_map[speed] / fs

        total_samples = int(duration * fs)
        signal_segment = signal[:total_samples]

        # Live metrics placeholders
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        bpm_box = m1.empty()
        status_box = m2.empty()
        alert_box = m3.empty()
        sample_box = m4.empty()

        chart_placeholder = st.empty()
        alert_placeholder = st.empty()

        if start_btn:
            buffer = []

            for i, sample in enumerate(signal_segment):
                buffer.append(sample)

                # Keep rolling window
                if len(buffer) > WINDOW_SIZE:
                    buffer.pop(0)

                # Update chart every 18 samples (~20fps)
                if i % 18 == 0:
                    buf = np.array(buffer)

                    # Live BPM estimate
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(buf, distance=int(0.4 * fs))
                    if len(peaks) > 1:
                        rr = np.diff(peaks) / fs
                        live_bpm = int(60 / np.mean(rr))
                    else:
                        live_bpm = 0

                    # Status
                    if live_bpm < 50:
                        status = "⚠️ Bradycardia"
                        alert_placeholder.warning("ALERT: Low heart rate detected")
                    elif live_bpm > 120:
                        status = "🔴 Tachycardia"
                        alert_placeholder.error("ALERT: High heart rate detected")
                    else:
                        status = "✅ Normal"
                        alert_placeholder.empty()

                    # Update metrics
                    bpm_box.metric("Live BPM", f"{live_bpm}" if live_bpm > 0 else "—")
                    status_box.metric("Status", status)
                    alert_box.metric("Elapsed", f"{i/fs:.1f}s")
                    sample_box.metric("Sample", i)

                    # Update chart
                    chart_placeholder.line_chart(
                        pd.DataFrame({"ECG": buf}),
                        height=300
                    )

                    time.sleep(delay * 18)
            
            st.success("Monitoring session complete.")
    else:
        st.info("Upload both .hea and .dat files to start ICU monitoring.")
# ══════════════════════════════════════════════════════════════
# SYSTEM INFO PAGE
# ══════════════════════════════════════════════════════════════
elif page == "System Info":
    st.title("System Information")

    st.markdown("""
    | Component | Detail |
    |-----------|--------|
    | Model | Latent Neural ODE |
    | Encoder | GRU (hidden_dim=64) |
    | ODE Solver | Euler (fast inference) |
    | Classifier | Linear (64 → 2) |
    | Dataset | MIT-BIH (Kaggle preprocessed) |
    | Training Samples | 6,000 (balanced) |
    | Test Samples | 7,548 (balanced) |
    | Framework | PyTorch + torchdiffeq |
    | Dashboard | Streamlit |
    """)

    st.success("System Active")
    st.info("Model loaded and ready for inference")