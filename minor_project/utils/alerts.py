import numpy as np
from scipy.signal import find_peaks

def detect_spikes(signal, threshold=2.5):
    """Detect sudden amplitude spikes"""
    alerts = []
    z_scores = np.abs((signal - signal.mean()) / (signal.std() + 1e-5))
    spike_indices = np.where(z_scores > threshold)[0]
    if len(spike_indices) > 0:
        alerts.append({
            "type": "SPIKE",
            "severity": "HIGH",
            "message": f"{len(spike_indices)} spike(s) detected in signal",
            "indices": spike_indices.tolist()
        })
    return alerts

def detect_flatline(signal, window=30, std_threshold=0.02):
    """Detect flatline segments (near-zero variance)"""
    alerts = []
    for i in range(0, len(signal) - window, window):
        segment = signal[i:i+window]
        if segment.std() < std_threshold:
            alerts.append({
                "type": "FLATLINE",
                "severity": "CRITICAL",
                "message": f"Flatline detected at timesteps {i}–{i+window}",
                "indices": list(range(i, i+window))
            })
            break  # report first occurrence only
    return alerts

def detect_irregular_rhythm(signal, sampling_rate=250):
    """Detect irregular RR intervals (arrhythmia indicator)"""
    alerts = []
    peaks, _ = find_peaks(signal, distance=30)
    if len(peaks) < 3:
        return alerts
    
    rr_intervals = np.diff(peaks) / sampling_rate * 1000  # in ms
    mean_rr = rr_intervals.mean()
    std_rr = rr_intervals.std()
    cv = std_rr / (mean_rr + 1e-5)  # coefficient of variation

    if cv > 0.20:
        alerts.append({
            "type": "IRREGULAR RHYTHM",
            "severity": "HIGH",
            "message": f"Irregular RR intervals detected (CV={cv:.2f}). Possible arrhythmia.",
            "indices": peaks.tolist()
        })

    # Bradycardia / Tachycardia
    duration = len(signal) / sampling_rate
    heart_rate = int((len(peaks) / duration) * 60)
    if heart_rate < 50:
        alerts.append({
            "type": "BRADYCARDIA",
            "severity": "HIGH",
            "message": f"Low heart rate: {heart_rate} BPM (< 50)",
            "indices": []
        })
    elif heart_rate > 150:
        alerts.append({
            "type": "TACHYCARDIA",
            "severity": "HIGH",
            "message": f"High heart rate: {heart_rate} BPM (> 150)",
            "indices": []
        })

    return alerts

def detect_high_variance(signal, threshold=0.8):
    """Detect abnormally high signal variance"""
    alerts = []
    if signal.std() > threshold:
        alerts.append({
            "type": "HIGH VARIANCE",
            "severity": "MEDIUM",
            "message": f"Signal variance ({signal.std():.3f}) exceeds threshold ({threshold})",
            "indices": []
        })
    return alerts

def run_all_alerts(signal, sampling_rate=250):
    """Run all detectors and return combined alert list"""
    alerts = []
    alerts += detect_spikes(signal)
    alerts += detect_flatline(signal)
    alerts += detect_irregular_rhythm(signal, sampling_rate)
    alerts += detect_high_variance(signal)
    return alerts