import mne, pandas as pd

raw = mne.io.read_raw_eeglab("C:/Users/iron/Desktop/EEG/lemon_preproc/sub-032301/sub-032301_EC.set", preload=True)
print("sfreq", raw.info["sfreq"])
print("highpass", raw.info["highpass"], "lowpass", raw.info["lowpass"])
print("n_ch", len(raw.ch_names))
print("duration s", round(raw.times[-1], 1))

paper19 = ["Fp1","Fp2","F7","F3","Fz","F4","F8","T7","C3","Cz","C4","T8","P7","P3","Pz","P4","P8","O1","O2"]

print("missing", [c for c in paper19 if c not in raw.ch_names])

p = pd.read_csv("Participants_MPILMBB_LEMON.csv")
print(p.head()); print(p.columns.tolist())

for c in p.columns:
    if p[c].nunique() < 15: print(c, p[c].value_counts().to_dict())

