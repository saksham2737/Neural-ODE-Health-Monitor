import wfdb
import numpy as np
import pandas as pd
import os

RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','203','205','207','208',
    '209','210','212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

# Normal beat types in MIT-BIH
NORMAL_BEATS = ['N', 'L', 'R', 'e', 'j']
DATA_PATH = 'data/mitbih'
OUT_PATH = 'data/mitbih_csv'
os.makedirs(OUT_PATH, exist_ok=True)

SEGMENT_LENGTH = 500
samples_saved = 0

for rec in RECORDS:
    try:
        record = wfdb.rdrecord(os.path.join(DATA_PATH, rec))
        ann = wfdb.rdann(os.path.join(DATA_PATH, rec), 'atr')
        signal = record.p_signal  # shape: (N_samples, n_leads)

        for i, (sample_idx, symbol) in enumerate(zip(ann.sample, ann.symbol)):
            start = sample_idx - SEGMENT_LENGTH // 2
            end = start + SEGMENT_LENGTH
            if start < 0 or end > len(signal):
                continue

            segment = signal[start:end, :2].astype(float)  # first 2 leads
            if segment.shape[1] < 2:
                continue

            label = 'normal' if symbol in NORMAL_BEATS else 'abnormal'
            df = pd.DataFrame(segment, columns=['lead1', 'lead2'])
            df['label'] = label
            fname = f"{label}_{rec}_{i}.csv"
            df.to_csv(os.path.join(OUT_PATH, fname), index=False)
            samples_saved += 1

    except Exception as e:
        print(f"Skipping {rec}: {e}")

print(f"Done. Saved {samples_saved} segments to {OUT_PATH}")