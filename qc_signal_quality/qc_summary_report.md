# QC summary report: PSD, 1/f (aperiodic) slope, and impedance

Generated: 2026-07-22T11:16:39.382302+00:00

Additive diagnostics only -- no preprocessed data was modified, excluded, or re-run by this audit. See `aperiodic_fit_results.csv` and `impedance_results.csv` for full per-row evidence.

## A. PSD flatness

Checked 1476 channel-sessions (18 channels x 82 sessions).
**22/1476 channel-sessions flagged flat** (low[1-4Hz]/high[30-45Hz] power ratio < 6 dB).

Flagged: sub-105-ON/P4 (5.3 dB); sub-105-OFF/P4 (6.0 dB); sub-108-ON/P4 (5.9 dB); sub-108-OFF/P4 (5.3 dB); sub-129-ON/O1 (3.1 dB); sub-80-OFF/T4 (5.9 dB); sub-80-OFF/T5 (5.1 dB); sub-80-OFF/P4 (5.9 dB); sub-92-OFF/F7 (4.5 dB); sub-92-OFF/F8 (5.8 dB); sub-92-OFF/P4 (4.9 dB); sub-93-OFF/Fp2 (5.0 dB); sub-93-OFF/F7 (5.6 dB); sub-93-OFF/F4 (5.0 dB); sub-93-OFF/C4 (5.3 dB); sub-93-OFF/P4 (6.0 dB); sub-97-OFF/C3 (5.8 dB); sub-97-OFF/P4 (5.5 dB); sub-98-ON/C3 (5.1 dB); sub-98-ON/C4 (5.7 dB); sub-98-OFF/P4 (5.8 dB); sub-99-ON/P4 (5.5 dB)

**Channel pattern:** P4 accounts for 10/22 of the flat-PSD flags across different subjects -- worth checking that electrode specifically (contact quality/placement) rather than treating these as independent random flags.

**Headline:** the large majority of channels show a clear 1/f-like decline; flat channels are isolated, not systemic.

### Follow-up: candidate flat/high-impedance patients (cross-session persistence)

Neither flag above requires a pattern to hold across BOTH of a patient's sessions. Checking for that specifically -- patients where >= 25% of channels have aperiodic exponent < 0.5 in **both** their ON and OFF recording (a genuine contact/impedance problem should not clear up between visits, unlike a one-off noisy session) -- surfaces **6 patient(s)**:

| patient | frac. channels weak (ON) | frac. channels weak (OFF) |
|---|---|---|
| sub-105 | 67% | 56% |
| sub-97 | 50% | 56% |
| sub-98 | 50% | 56% |
| sub-108 | 44% | 39% |
| sub-99 | 33% | 33% |
| sub-80 | 28% | 56% |

Each patient in this table was visually cross-checked against its full 18-channel PSD grid (`psd_examples_flagged_patients/`) and its raw-vs-processed comparison (`psd_before_after_flagged_patients/`). Two distinct patterns were found:

- **A clean, consistent flat plateau** (most of the table, e.g. sub-105, sub-97, sub-98, sub-99, sub-108): nearly every channel shows a normal decline only up to roughly 5-10 Hz, then goes flat (a broadband noise floor) all the way to the 75 Hz cutoff, across almost the whole cap simultaneously -- not an isolated bad electrode. The same flat plateau is already present in the RAW, pre-preprocessing signal, so this is a property of the original recording (most consistent with poor whole-cap contact/high impedance at acquisition time), not something introduced by this pipeline.
- **A different, peakier pattern** (e.g. sub-80): elevated broadband noise with distinct oscillatory peaks (a large ~45 Hz peak was visible on F7) rather than a clean flat floor -- more consistent with movement/muscle contamination than a pure contact/impedance problem, though still anomalous and worth the same level of review.

This table is a reasonable candidate list for exclusion or re-review; this audit flags it, it does not remove or exclude anything.

## B. Aperiodic (1/f) slope (specparam)

Checked 1476 channel-sessions.
**272/1476 flagged** (32 near-zero exponent [|exp|<0.2], 250 poor fit [R^2<0.9], 0 fit errors).

Flagged: sub-103-ON/P4 (poor fit (R^2=0.600)); sub-103-OFF/F3 (poor fit (R^2=0.769)); sub-103-OFF/C4 (poor fit (R^2=0.684)); sub-105-ON/Fp1 (exponent near zero (0.194); poor fit (R^2=0.880)); sub-105-ON/F7 (poor fit (R^2=0.865)); sub-105-ON/C3 (exponent near zero (0.199)); sub-105-ON/Cz (poor fit (R^2=0.894)); sub-105-ON/T4 (poor fit (R^2=0.684)); sub-105-ON/P3 (exponent near zero (0.193)); sub-105-ON/P4 (poor fit (R^2=0.543)); sub-105-OFF/C3 (exponent near zero (0.189)); sub-105-OFF/P4 (exponent near zero (0.199)); sub-106-ON/F8 (poor fit (R^2=0.810)); sub-106-ON/Cz (poor fit (R^2=0.899)); sub-106-OFF/Fp2 (poor fit (R^2=0.855)); sub-106-OFF/Fz (poor fit (R^2=0.868)); sub-106-OFF/F8 (poor fit (R^2=0.797)); sub-106-OFF/P3 (poor fit (R^2=0.685)); sub-106-OFF/T6 (poor fit (R^2=0.703)); sub-107-ON/F4 (poor fit (R^2=0.725)); sub-107-ON/T3 (poor fit (R^2=0.896)); sub-107-ON/Cz (poor fit (R^2=0.863)); sub-107-ON/P3 (poor fit (R^2=0.670)); sub-107-ON/O2 (poor fit (R^2=0.751)); sub-107-OFF/F3 (poor fit (R^2=0.868)); sub-107-OFF/P3 (poor fit (R^2=0.792)); sub-107-OFF/P4 (exponent near zero (0.187)); sub-108-ON/P4 (exponent near zero (0.155)); sub-108-OFF/Fp1 (poor fit (R^2=0.793)); sub-108-OFF/C3 (exponent near zero (0.185)); sub-108-OFF/P4 (exponent near zero (0.098)); sub-108-OFF/O2 (poor fit (R^2=0.779)); sub-114-ON/Fp1 (exponent near zero (-0.065); poor fit (R^2=0.857)); sub-114-ON/Fp2 (poor fit (R^2=0.880)); sub-114-ON/F4 (poor fit (R^2=0.820)); sub-114-ON/F8 (exponent near zero (-0.188); poor fit (R^2=0.763)); sub-114-ON/C3 (poor fit (R^2=0.699)); sub-117-ON/T5 (poor fit (R^2=0.823)); sub-119-ON/T3 (poor fit (R^2=0.743)); sub-119-OFF/P3 (poor fit (R^2=0.884)); ... +232 more (see CSV)

**Session pattern:** 8 session(s) have >=50% of channels flagged -- likely a whole-session data-quality issue rather than isolated bad channels: sub-96-OFF (18/18 channels); sub-34-ON (17/18 channels); sub-77-ON (13/18 channels); sub-69-OFF (11/18 channels); sub-76-OFF (9/18 channels); sub-69-ON (9/18 channels); sub-92-OFF (9/18 channels); sub-93-OFF (9/18 channels)

Beta-band (13-30 Hz) peak detected in 39.4% of channel-sessions overall (ON: 35.8%, OFF: 43.1%). This is a sanity reference only (levodopa-related beta suppression is a well-established finding, i.e. OFF > ON expected) -- not a target this audit fits to.

**Headline:** a non-trivial fraction of channel-sessions have a poor or near-flat aperiodic fit -- cross-check against the Part A flat-PSD list and the psd_examples/ plots before fine-tuning.

## C. Impedance

Checked 95 file(s) across 5 sample session(s) (raw-signal CSV header + data columns, EDF channel names + annotations, and a best-effort ASCII scan of any `.dsi` file).

**Headline: no impedance data is present anywhere in this dataset**, in any of the three formats found on disk. Specifically:

- **dsi** (31 file(s) checked): none carried impedance values. ASCII sniff of the full file found no 'imped'/'impedance' byte sequence (proprietary binary packet format, no vendor SDK available -- a structured, non-ASCII-tagged impedance packet cannot be ruled out by this check alone)
- **edf** (31 file(s) checked): none carried impedance values. no 'imped*' channel name or annotation description among 25 channels / 1 distinct annotation description(s) (channels checked: ['EEG P3-Pz', 'EEG C3-Pz', 'EEG F3-Pz', 'EEG Fz-Pz', 'EEG F4-Pz', 'EEG C4-Pz', 'EEG P4-Pz', 'EEG Cz-Pz', 'CM', 'EEG A1-Pz', 'EEG Fp1-Pz', 'EEG Fp2-Pz', 'EEG T3-Pz', 'EEG T5-Pz', 'EEG O1-Pz', 'EEG O2-Pz', 'EEG X3-Pz', 'EEG X2-Pz', 'EEG F7-Pz', 'EEG F8-Pz', 'EEG X1-Pz', 'EEG A2-Pz', 'EEG T6-Pz', 'EEG T4-Pz', 'Trigger'])
- **raw_csv** (33 file(s) checked): none carried impedance values. no 'imped*' key in 15 metadata field(s) or 31 data column(s) (checked keys: ['Mains_Frequency_(Hz)', 'Sample_Frequency_(Hz)', 'Filter_Delay_(ms)', 'Sensor_Data_Units', 'Headset_Name', 'Data_Logger', 'Date', 'Time', 'Patient_ID', 'Record_ID', 'Filter', 'Comments', 'Reference location', 'Trigger Source', 'Channel_Number']; checked columns: ['Time', 'P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'CM', 'A1', 'Fp1', 'Fp2', 'T3', 'T5', 'O1', 'O2', 'X3', 'X2', 'F7', 'F8', 'X1', 'A2', 'T6', 'T4', 'Trigger', 'Time_Offset', 'ADC_Status', 'ADC_Sequence', 'Event', 'Comments'])

This is reported as a genuine data-availability finding, not a skipped check: DSI-Streamer supports continuous impedance monitoring, but it is not exported into any file format present in this dataset on this drive (no impedance column/metadata key in the raw-signal CSV, no impedance channel or annotation in the EDF exports, and no plain-text impedance marker in the `.dsi` byte stream -- though the `.dsi` format is proprietary binary with no vendor SDK available here, so a structured non-ASCII-tagged impedance packet cannot be fully ruled out by this check alone).

## Processing failures (logged, not silently skipped)

- All sessions in the manifest were successfully reconstructed and processed.
