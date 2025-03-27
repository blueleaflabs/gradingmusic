# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full audio analysis pipeline that:
  - Defines all relevant constants exactly as in your BBEdit file.
  - Preprocesses audio (load, resample, normalize, trim, bandpass, HPSS).
  - Computes spectral data, pitch, Praat metrics, RMS, LUFS, etc.
  - Builds the final JSON structure (time_matrices, summary, advanced features).
  - Inserts the result into PostgreSQL via QuantumMusicDB (save_to_db).
  - Provides a main function grade_single_file that orchestrates everything.

Updated to:
  - Use `librosa.feature.rhythm.tempo` instead of `librosa.beat.tempo`.
  - Zero-pad short chunks to avoid "n_fft too large" warnings.
"""

import os
import sys
import re
import math
import logging
import shutil
import concurrent.futures
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colormaps
from scipy.signal import find_peaks, iirnotch, filtfilt, butter
import pyloudnorm as pyln
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_histogram
import optuna
import parselmouth
from parselmouth.praat import call
import pyloudnorm as pyln  # ensure installed, otherwise LOUDNORM_AVAILABLE = False
from IPython.display import Audio, display


# -----------------------
# Database
import psycopg2
from psycopg2.extras import Json





# ============== CONSTANTS & IMPORTS from your BBEdit snippet =============

INPUT_DIR = "data/trainingdata"
OUTPUT_DIR = "data/analysisoutput"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DB_NAME = "quantummusic_csef"
DB_HOST = "localhost"
DB_USER = "postgres"  # placeholder
DB_PASSWORD = "postgres"  # placeholder

# Audio processing constants
STANDARD_SR = 44100  # Standard sampling rate
SILENCE_THRESHOLD_DB = 30  # dB threshold for silence trimming

# Band-pass filter constants
LOW_FREQ = 80.0
HIGH_FREQ = 3000.0

FIG_SIZE = (10, 4)

SAVE_TO_DB = True
N_MFCC = 40

# Frame-based approach for pitch detection
FRAME_SIZE = 2048
HOP_LENGTH = 512

# Praat-based chunk size
PRAAT_CHUNK_SIZE = 2048

# Deviation threshold in cents, for dev_flag
DEVIATION_THRESHOLD = 50.0

HPSS_MARGIN = (1.0, 3.0)
BANDPASS_FILTER_ORDER = 4

# Multi-chunk tempo analysis
TEMPO_CHUNK_SIZE_MEDIUM = 4096
TEMPO_CHUNK_SIZE_LARGE = 22050

#  constants for advanced vocal feature extraction
VOCAL_FEATURE_CHUNK_SIZE = 22050
VOCAL_FEATURE_CHUNK_HOP = 4096
# keeping these the same as large chunk tempo analysis 

#  constant for LUFS calculations (0.5s @ 44.1kHz)
LUFS_CHUNK_SIZE = 22050

# ---  CONSTANTS FOR ADVANCED VOCAL FEATURE EXTRACTION ---
# Formant analysis parameters
FORMANT_ANALYSIS_TIME = 0.1
FORMANT_TIME_STEP = 0.01
MAX_NUMBER_OF_FORMANTS = 5
MAXIMUM_FORMANT_FREQUENCY = 5500
NUM_FORMANTS_TO_EXTRACT = 3

# Pitch / jitter / shimmer parameters
MIN_F0 = 75
MAX_F0 = 500
JITTER_TIME_STEP = 0.0001
JITTER_MIN_PERIOD = 0.02
JITTER_MAX_PERIOD = 1.3
SHIMMER_MIN_AMPLITUDE = 0.0001
SHIMMER_MAX_AMPLITUDE = 0.02
SHIMMER_FACTOR = 1.6

# Vibrato analysis parameters
VIBRATO_MIN_HZ = 3
VIBRATO_MAX_HZ = 10
MEDIAN_FILTER_KERNEL_SIZE = 9


# ============== Database Class =============
class QuantumMusicDB:
    """
    Connection to the PostgreSQL database and basic CRUD operations.
    """
    def __init__(self, db_name=DB_NAME, host=DB_HOST, user=DB_USER, password=DB_PASSWORD):
        self.db_name = db_name
        self.host = host
        self.user = user
        self.password = password
        self.conn = None
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.db_name,
                host=self.host,
                user=self.user,
                password=self.password
            )
            #print(f"Connected to database {self.db_name} successfully.")
        except Exception as e:
            print(f"Error connecting to database: {e}")

    def create_tables(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS audio_analysis (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(255),
            upload_date TIMESTAMP DEFAULT NOW(),
            sample_rate INT,
            analysis_data JSONB
        );
        """
        with self.conn.cursor() as cur:
            cur.execute(create_table_query)
            self.conn.commit()
        #print("Tables ensured.")

    def insert_analysis(self, file_name, sample_rate, analysis_data):
        """
        Insert an analysis record into the DB.
        analysis_data is stored as a JSONB column using psycopg2.extras.Json.
        """
        insert_query = """
        INSERT INTO audio_analysis(file_name, sample_rate, analysis_data)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        with self.conn.cursor() as cur:
            cur.execute(insert_query, (file_name, sample_rate, Json(analysis_data)))
            _id = cur.fetchone()[0]
            self.conn.commit()
        return _id

    def fetch_analysis(self, record_id):
        """
        Fetch a specific analysis record by ID.
        """
        select_query = """
        SELECT id, file_name, sample_rate, analysis_data
        FROM audio_analysis
        WHERE id=%s;
        """
        with self.conn.cursor() as cur:
            cur.execute(select_query, (record_id,))
            row = cur.fetchone()
        return row

    def close(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")


# ============== Preprocessing =============
def preprocess_audio(
    file_path,
    target_sr=STANDARD_SR,
    silence_threshold_db=SILENCE_THRESHOLD_DB,
    low_freq=LOW_FREQ,
    high_freq=HIGH_FREQ,
    margin=HPSS_MARGIN
):
    """
    Load audio, resample, normalize, trim silence, apply bandpass filter,
    and remove percussive components (HPSS) in one function.
    """
    audio_data, sr = librosa.load(file_path, sr=None)

    # Resample if needed
    if sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Normalize
    peak = np.max(np.abs(audio_data))
    if peak > 1e-9:
        audio_data /= peak

    # Trim silence
    audio_data, _ = librosa.effects.trim(audio_data, top_db=silence_threshold_db)

    # Bandpass filter
    nyquist = 0.5 * sr
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(N=BANDPASS_FILTER_ORDER, Wn=[low, high], btype='band')
    audio_data = filtfilt(b, a, audio_data)

    # HPSS
    stft_data = librosa.stft(audio_data)
    harmonic_part, _ = librosa.decompose.hpss(stft_data, margin=margin)
    processed_audio = librosa.istft(harmonic_part)

    return processed_audio, sr


# ============== Utility to Avoid NaN in JSON =============
def _replace_nan_with_none(value):
    if isinstance(value, float) and np.isnan(value):
        return None
    elif isinstance(value, list):
        return [_replace_nan_with_none(v) for v in value]
    elif isinstance(value, tuple):
        return tuple(_replace_nan_with_none(v) for v in value)
    elif isinstance(value, dict):
        return {k: _replace_nan_with_none(v) for k, v in value.items()}
    else:
        return value



# ============== Base pitch detection =============
def detect_drone_pitch(audio_data, sr, min_freq=70.0, max_freq=300.0):
    """
    Attempt to find a continuous drone by computing the time-averaged magnitude spectrum
    and looking for the strongest peak in [min_freq..max_freq].
    
    If a peak is found above some dominance threshold, return that freq; else 0.0
    """
    # We'll do a single STFT for the entire file.
    # Zero-pad if audio is shorter than FRAME_SIZE to avoid 'n_fft too large' warnings.
    n_fft = FRAME_SIZE  # references your existing constant
    if len(audio_data) < n_fft:
        pad_len = n_fft - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad_len), mode='constant')

    # Compute STFT
    S_complex = librosa.stft(audio_data, n_fft=n_fft, hop_length=HOP_LENGTH)
    S_mag = np.abs(S_complex)

    # Average the magnitude across time -> a single "long-term" spectrum
    mean_spectrum = np.mean(S_mag, axis=1)  # shape => (n_fft/2+1,)

    # Frequency array for each bin
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Restrict analysis to [min_freq..max_freq]
    valid_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
    if len(valid_indices) == 0:
        return 0.0

    sub_spectrum = mean_spectrum[valid_indices]
    sub_freqs = freqs[valid_indices]

    # Find the index of the maximum amplitude within that sub-range
    max_idx = np.argmax(sub_spectrum)
    peak_freq = sub_freqs[max_idx]
    peak_val = sub_spectrum[max_idx]

    # We do a quick "dominance" check: compare the loudest bin to the second-loudest
    sorted_vals = np.sort(sub_spectrum)
    if len(sorted_vals) > 1:
        second_loudest = sorted_vals[-2]
    else:
        second_loudest = 1e-12

    ratio = peak_val / (second_loudest + 1e-12)
    # If ratio < ~1.5, we consider the peak not dominant enough to be a drone
    if ratio < 1.5:
        return 0.0

    return float(peak_freq)


def detect_base_pitch_with_pyin(audio_data, sr, fmin=MIN_F0, fmax=MAX_F0):
    """
    If no drone, we fallback to median pitch from pyin across entire audio.
    Returns 0.0 if no pitched frames are found.
    """
    all_pitches, voiced_flags, _ = librosa.pyin(
        y=audio_data,
        sr=sr,
        fmin=fmin,
        fmax=fmax
    )
    valid_pitches = []
    for p, vf in zip(all_pitches, voiced_flags):
        if vf and p is not None and not np.isnan(p):
            valid_pitches.append(p)
    if not valid_pitches:
        return 0.0
    return float(np.median(valid_pitches))



# ============== Tone to Noise Calculation =============
def compute_tone_to_noise(S_mag_column):
    """
    Use spectral flatness to approximate tone_to_noise.
    S_mag_column: magnitude for a single frame => shape (n_fft/2+1,).
    """
    if len(S_mag_column) < 1:
        return 0.0
    numerator = np.exp(np.mean(np.log(S_mag_column + 1e-12)))
    denominator = np.mean(S_mag_column + 1e-12)
    flatness = numerator / denominator
    return float(flatness)


# ============== Transition Score Calculation =============
def compute_transition_score(pitch_prev, pitch_curr):
    """
    Example: difference-based measure. The smaller the difference, the bigger the score.
    transition_score = 1.0 - min(|pitch_curr - pitch_prev| / 100.0, 1.0)
    """
    if pitch_prev < 1e-6 or pitch_curr < 1e-6:
        return 0.0
    diff = abs(pitch_curr - pitch_prev)
    raw_score = 1.0 - min(diff / 100.0, 1.0)
    return max(raw_score, 0.0)


# ============== Pitch -> Sruti Mapping =============
def map_pitch_to_sruti(pitch_hz, base_pitch=240.0):
    """
    Return (sruti_class, note_name, note_freq_hz, deviation_cents, dev_flag).
    """
    if pitch_hz <= 0.0 or np.isnan(pitch_hz):
        return ("sruti_unknown", "unknown_note", 0.0, 0.0, 0)
    sruti_class = "sruti_3"  # dummy
    note_name = "Sa"         # dummy
    note_freq_hz = base_pitch
    deviation_cents = (pitch_hz - base_pitch) * 10.0
    dev_flag = 1 if abs(deviation_cents) > DEVIATION_THRESHOLD else 0
    return (sruti_class, note_name, note_freq_hz, deviation_cents, dev_flag)


# ============== Praat-based metrics per short chunk =============
def compute_praat_metrics(chunk_data, sr):
    """
    Return a dict with:
      praat_hnr, hnr_category, jitter, shimmer,
      formants = {F1, F2, F3},
      vibrato_extent, vibrato_rate

    Now with a real vibrato calculation:
      - We extract pitch contour from parselmouth
      - Convert pitch to cents around the median
      - Band-limit the pitch contour to [3..10 Hz]
      - Estimate vibrato_extent from the amplitude of that band
      - Estimate vibrato_rate from the frequency peak in [3..10 Hz]
    """
    out = {
        "praat_hnr": 0.0,
        "hnr_category": "unknown",
        "jitter": 0.0,
        "shimmer": 0.0,
        "formants": {"F1": 0.0, "F2": 0.0, "F3": 0.0},
        "vibrato_extent": 0.0,
        "vibrato_rate": 0.0
    }
    # If chunk is too tiny, skip
    if len(chunk_data) < 10:
        return out

    try:
        import math
        import numpy as np
        import parselmouth
        from parselmouth.praat import call
        from scipy.signal import welch

        snd = parselmouth.Sound(chunk_data, sr)

        # 1) Pitch-based metrics (Jitter/Shimmer/HNR)
        pp = call(snd, "To PointProcess (periodic, cc)", MIN_F0, MAX_F0)
        jit_local = call(
            pp, "Get jitter (local)", 0, 0,
            JITTER_TIME_STEP, JITTER_MIN_PERIOD, JITTER_MAX_PERIOD
        )
        shim_local = call(
            [snd, pp], "Get shimmer (local)", 0, 0,
            SHIMMER_MIN_AMPLITUDE, SHIMMER_MAX_AMPLITUDE,
            JITTER_MAX_PERIOD, SHIMMER_FACTOR
        )
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 86.1329, 0.1, 1.0)
        hnr_val = call(harmonicity, "Get mean", 0, 0)
        if (hnr_val is None) or math.isnan(hnr_val):
            hnr_val = 0.0

        if hnr_val > 20.0:
            hnr_cat = "good"
        elif hnr_val > 15.0:
            hnr_cat = "acceptable"
        else:
            hnr_cat = "weak"

        # 2) Formants: pick midpoint of chunk
        formant_obj = snd.to_formant_burg(
            time_step=FORMANT_TIME_STEP,
            max_number_of_formants=MAX_NUMBER_OF_FORMANTS,
            maximum_formant=MAXIMUM_FORMANT_FREQUENCY
        )
        duration_s = snd.get_total_duration()
        analysis_time = duration_s * 0.5
        F1 = formant_obj.get_value_at_time(1, analysis_time) or 0.0
        F2 = formant_obj.get_value_at_time(2, analysis_time) or 0.0
        F3 = formant_obj.get_value_at_time(3, analysis_time) or 0.0

        # 3) Vibrato from pitch contour
        pitch_obj = snd.to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.6,
            pitch_floor=MIN_F0,
            pitch_ceiling=MAX_F0
        )
        pitch_values = pitch_obj.selected_array['frequency']
        times = pitch_obj.xs()  # array of times for each pitch sample
        # Filter out unvoiced or 0.0
        valid_indices = np.where(pitch_values > 0.0)[0]
        if len(valid_indices) < 4:
            vib_extent = 0.0
            vib_rate = 0.0
        else:
            pitched_times = times[valid_indices]
            pitched_freqs = pitch_values[valid_indices]

            # 3A) Convert to "cents around median pitch"
            median_pitch = np.median(pitched_freqs)
            if median_pitch <= 0.0:
                median_pitch = 1e-6
            pitch_cents = 1200.0 * np.log2(pitched_freqs / median_pitch)

            # 3B) Vibrato extent: standard deviation of pitch_cents
            vib_extent = float(np.std(pitch_cents))

            # 3C) Vibrato rate: we find the main frequency in [3..10 Hz] via Welch
            #     If chunk is short (< ~0.3s), we might not detect a strong vibrato peak.
            fs_pitch = 1.0 / (float(pitch_obj.get_time_step()))  # pitch sampling rate
            freqs_w, psd_w = welch(pitch_cents, fs=fs_pitch, nperseg=len(pitch_cents)//2 or 1)
            # band-limit to [3..10 Hz]
            vib_band = np.where((freqs_w >= VIBRATO_MIN_HZ) & (freqs_w <= VIBRATO_MAX_HZ))[0]
            if len(vib_band) < 1:
                vib_rate = 0.0
            else:
                freqs_sub = freqs_w[vib_band]
                psd_sub = psd_w[vib_band]
                peak_idx = np.argmax(psd_sub)
                vib_rate = float(freqs_sub[peak_idx])  # Hz

        out = {
            "praat_hnr": float(hnr_val),
            "hnr_category": hnr_cat,
            "jitter": float(jit_local) if jit_local is not None else 0.0,
            "shimmer": float(shim_local) if shim_local is not None else 0.0,
            "formants": {
                "F1": float(F1),
                "F2": float(F2),
                "F3": float(F3)
            },
            "vibrato_extent": vib_extent,
            "vibrato_rate": vib_rate
        }

    except Exception:
        pass

    return out

# ============== Build entire spectral_data =============
# TODO: With the changes we're making to time_matrix_small, this will not get called
# Delete it later
def build_spectral_data(audio_data, sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH):
    # If audio_data < n_fft, pad it to avoid warnings
    if len(audio_data) < n_fft:
        pad_len = n_fft - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad_len), 'constant')

    S_complex = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    S_mag = np.abs(S_complex)
    S_log = librosa.amplitude_to_db(S_mag, ref=np.max)

    power_spectrogram = S_mag**2
    mfcc_data = librosa.feature.mfcc(
        S=librosa.power_to_db(power_spectrogram),
        sr=sr,
        n_mfcc=N_MFCC
    )
    mfcc_delta = librosa.feature.delta(mfcc_data)

    spectral_data = {
        "spectrogram_magnitude": S_mag.tolist(),
        "spectrogram_log_db": S_log.tolist(),
        "mfcc": mfcc_data.tolist(),
        "delta_mfcc": mfcc_delta.tolist()
    }
    return spectral_data


# ============== Build time_matrix_small =============
def build_time_matrix_small(
    audio_data,
    sr,
    file_path,
    base_pitch,
    frame_size,
    hop_length,
    n_mfcc
):
    import os
    import numpy as np
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import pyloudnorm as pyln

    # If audio is too short, zero-pad to avoid n_fft warnings
    if len(audio_data) < frame_size:
        pad_len = frame_size - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad_len), mode='constant')

    # -------------------------------------------------
    # 1) Compute & Save the MEL-SPECTROGRAM as a PNG
    # -------------------------------------------------
    # Purely extracted for the AST model fine tuning
    # This is a second load but being done solely for the AST model
    mel_y, mel_sr = librosa.load(file_path, sr=16000)   # Force 16 kHz
    mel_spec = librosa.feature.melspectrogram(
        y=mel_y,
        sr=mel_sr,
        n_fft=400,
        hop_length=160,
        n_mels=128
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create a roughly 224×224 PNG in grayscale
    fig = plt.figure(figsize=(3.11, 3.11), dpi=72)  # ~224×224 px
    ax = fig.add_subplot(111)
    img = librosa.display.specshow(
        log_mel_spec,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        cmap='gray',
        ax=ax
    )
    plt.title('Mel Spectrogram')
    plt.axis('off')  # no axes or ticks
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # ensure output dir
    mel_filename = os.path.join(OUTPUT_DIR, f"{base_name}_mel.png")
    plt.savefig(mel_filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # -------------------------------------------------
    # 2) Now do your existing STFT-based analysis
    # -------------------------------------------------
    S_complex = librosa.stft(audio_data, n_fft=frame_size, hop_length=hop_length, center=False)
    S_mag = np.abs(S_complex)
    S_power = S_mag**2
    S_log = librosa.amplitude_to_db(S_mag, ref=np.max)

    # e.g., local ZCR, centroid, rolloff, bandwidth, RMS, LUFS, Pyin pitch, etc.
    zcr_data = librosa.feature.zero_crossing_rate(audio_data, frame_length=frame_size, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(S=S_mag)[0]
    rolloff  = librosa.feature.spectral_rolloff(S=S_mag)[0]
    bw       = librosa.feature.spectral_bandwidth(S=S_mag)[0]
    rms_vals = librosa.feature.rms(y=audio_data, frame_length=frame_size, hop_length=hop_length)[0]

    meter = pyln.Meter(sr)
    min_required = int(0.4 * sr)

    def compute_lufs_for_frame(i):
        start_s = i * hop_length
        end_s   = start_s + frame_size
        if end_s > len(audio_data):
            end_s = len(audio_data)
        chunk = audio_data[start_s:end_s]
        if len(chunk) >= min_required:
            return meter.integrated_loudness(chunk)
        else:
            return 0.0

    from librosa.feature import rhythm
    def compute_local_tempo_for_frame(i):
        start_s = i * hop_length
        end_s = start_s + frame_size
        if end_s > len(audio_data):
            end_s = len(audio_data)
        chunk = audio_data[start_s:end_s]
        if len(chunk) < 32:
            return 0.0
        tmp = rhythm.tempo(y=chunk, sr=sr, hop_length=hop_length, aggregate=None)
        return float(np.mean(tmp)) if tmp is not None and len(tmp) > 0 else 0.0

    # Pyin pitch
    pitches, voiced_flags, confidences = librosa.pyin(
        y=audio_data,
        fmin=MIN_F0,
        fmax=MAX_F0,
        sr=sr,
        frame_length=frame_size,
        hop_length=hop_length
    )
    times = librosa.times_like(pitches, sr=sr, hop_length=hop_length)

    # MFCC_data if you want to store or summarize MFCC as well
    MFCC_data = librosa.feature.mfcc(
        S=librosa.power_to_db(S_power),
        sr=sr,
        n_mfcc=N_MFCC
    )
    MFCC_data = MFCC_data.T  # shape (#frames, n_mfcc)

    time_matrix_small = []
    prev_pitch = 0.0
    n_frames = len(pitches)
    all_mfcc_frames = []

    for i in range(n_frames):
        t_s = float(times[i])
        pitch_val = pitches[i] if (pitches[i] is not None and not np.isnan(pitches[i])) else 0.0
        conf_val  = float(confidences[i]) if confidences[i] is not None else 0.0
        vf        = bool(voiced_flags[i]) if voiced_flags[i] is not None else False

        sruti_class, note_name, note_freq_hz, dev_cents, dev_flag = map_pitch_to_sruti(pitch_val, base_pitch)

        tone_val = 0.0
        if i < S_mag.shape[1]:
            tone_val = compute_tone_to_noise(S_mag[:, i])

        if i == 0:
            trans_score = 1.0
        else:
            trans_score = compute_transition_score(prev_pitch, pitch_val)
        prev_pitch = pitch_val

        local_bpm = compute_local_tempo_for_frame(i)

        start_samp = i * hop_length
        end_samp = start_samp + frame_size
        if end_samp > len(audio_data):
            end_samp = len(audio_data)
        chunk_data = audio_data[start_samp:end_samp]
        praat_dict = compute_praat_metrics(chunk_data, sr)

        zcr_val = float(zcr_data[i]) if i < len(zcr_data) else 0.0
        sc_val  = float(centroid[i])  if i < len(centroid)  else 0.0
        ro_val  = float(rolloff[i])   if i < len(rolloff)   else 0.0
        bw_val  = float(bw[i])        if i < len(bw)        else 0.0

        rms_db_val = -60.0
        if i < len(rms_vals):
            raw_rms = rms_vals[i]
            if raw_rms <= 0.0:
                rms_db_val = -60.0
            else:
                rms_db_val = float(20.0 * np.log10(raw_rms + 1e-12))

        lufs_val = compute_lufs_for_frame(i)

        # store frame-level MFCC if desired
        if i < MFCC_data.shape[0]:
            mfcc_vec = MFCC_data[i]
        else:
            mfcc_vec = np.zeros(n_mfcc)
        all_mfcc_frames.append(mfcc_vec)

        row = {
            "time_s": t_s,
            "pitch_hz": pitch_val,
            "note_freq_hz": note_freq_hz,
            "voiced_flag": vf,
            "confidence": conf_val,
            "sruti_class": sruti_class,
            "note_name": note_name,
            "deviation_cents": dev_cents,
            "dev_flag": dev_flag,
            "pitch_accuracy_category": "good" if dev_flag == 0 else "out_of_tune",

            "tone_to_noise": tone_val,
            "transition_score": trans_score,

            "praat_hnr": praat_dict["praat_hnr"],
            "hnr_category": praat_dict["hnr_category"],
            "jitter": praat_dict["jitter"],
            "shimmer": praat_dict["shimmer"],
            "formants": praat_dict["formants"],
            "vibrato_extent": praat_dict["vibrato_extent"],
            "vibrato_rate": praat_dict["vibrato_rate"],

            "zcr": zcr_val,
            "spec_centroid": sc_val,
            "spec_rolloff": ro_val,
            "spec_bandwidth": bw_val,

            "rms_db": rms_db_val,
            "lufs": float(lufs_val),
            "tempo_bpm": local_bpm
        }
        time_matrix_small.append(row)

    # Summarize MFCC if you like:
    all_mfcc_frames = np.array(all_mfcc_frames)
    if len(all_mfcc_frames) > 0:
        mfcc_mean = np.mean(all_mfcc_frames, axis=0).tolist()
        mfcc_std  = np.std(all_mfcc_frames, axis=0).tolist()
    else:
        mfcc_mean = [0.0]*n_mfcc
        mfcc_std  = [0.0]*n_mfcc

    mfcc_summary = {
        "mfcc_mean": mfcc_mean,
        "mfcc_std":  mfcc_std
    }

    return time_matrix_small, S_mag, S_power, S_log, mfcc_summary



# ============== tempo-based matrix for bigger chunks =============
def build_tempo_matrix(audio_data, sr, chunk_size, overlap=0.5):
    """
    Build a time matrix with chunk-based tempo, RMS, and LUFS,
    zero-padding if chunk < n_fft to avoid warnings.
    """
    from librosa.feature import rhythm
    hop = int(chunk_size * (1.0 - overlap))
    n_samps = len(audio_data)
    chunks = []
    idx = 0
    chunk_index = 0
    meter = pyln.Meter(sr)

    while idx < n_samps:
        start_samp = idx
        end_samp = idx + chunk_size
        if end_samp > n_samps:
            end_samp = n_samps
        chunk_data = audio_data[start_samp:end_samp]
        start_time_s = float(start_samp / sr)

        if len(chunk_data) < 32:
            tempo_bpm = 0.0
        else:
            tmp = rhythm.tempo(y=chunk_data, sr=sr, hop_length=HOP_LENGTH, aggregate=None)
            tempo_bpm = float(np.mean(tmp)) if tmp is not None and len(tmp) > 0 else 0.0

        # RMS
        if len(chunk_data) > 0:
            chunk_rms = np.sqrt(np.mean(chunk_data**2))
            chunk_rms_db = float(20.0 * np.log10(chunk_rms + 1e-12))
        else:
            chunk_rms_db = -60.0

        # LUFS
        if len(chunk_data) >= int(0.4 * sr):
            lufs_val = meter.integrated_loudness(chunk_data)
        else:
            lufs_val = 0.0

        row = {
            "chunk_index": chunk_index,
            "start_time_s": start_time_s,
            "tempo_bpm": tempo_bpm,
            "rms_db": chunk_rms_db,
            "lufs": float(lufs_val)
        }
        chunks.append(row)

        chunk_index += 1
        idx += hop
        if idx >= n_samps:
            break

    return chunks

# ============== Advanced note features =============
def build_advanced_note_features(audio_data, sr):
    # Dummy example
    return [
        {
            "start_time_s": 0.0,
            "end_time_s": 2.4,
            "pitch_mean": 233.8,
            "vibrato_extent": 0.42,
            "vibrato_rate": 5.6
        },
        {
            "start_time_s": 2.4,
            "end_time_s": 5.1,
            "pitch_mean": 236.1,
            "vibrato_extent": 0.50,
            "vibrato_rate": 5.3
        }
    ]

# ============== Build tempo and advanced features - Med and Large =============
def build_tempo_and_advanced_features(
    audio_data,
    sr,
    time_matrix_small,
    chunk_size=None,
    hop_size=None
):
    """
    Builds a chunk-based matrix for "large" segments (~0.5s each) by:
      1) Doing integrated LUFS on each chunk (0.5s, so >=0.4s threshold).
      2) Aggregating short-frame data from `time_matrix_small` that falls within
         [start_time_s, end_time_s] to compute average tempo, pitch, jitter, shimmer, formants, etc.
      3) Computing a real vibrato measure from the pitch contour in that chunk.
         vibrato_extent => std. dev in cents; vibrato_rate => peaks/sec in that pitch contour.

    Returns: a list of dicts, each describing one chunk:
        [
          {
            "chunk_index": ...,
            "start_time_s": ...,
            "end_time_s": ...,
            "lufs": ...,
            "avg_tempo_bpm": ...,
            "avg_pitch_hz": ...,
            "avg_jitter": ...,
            "avg_shimmer": ...,
            "avg_hnr": ...,
            "avg_formants": {"F1":..., "F2":..., "F3":...},
            "vibrato_extent": ...,
            "vibrato_rate": ...
          },
          ...
        ]
    """

    from psycopg2.extras import Json  # if needed, not strictly necessary here

    # If not specified, read from your global constants
    if chunk_size is None:
        chunk_size = TEMPO_CHUNK_SIZE_LARGE  # e.g., 22050
    if hop_size is None:
        hop_size = VOCAL_FEATURE_CHUNK_HOP   # e.g., 4096

    meter = pyln.Meter(sr)
    results = []

    n_samps = len(audio_data)
    chunk_index = 0
    idx = 0

    while idx < n_samps:
        start_samp = idx
        end_samp = min(idx + chunk_size, n_samps)
        start_time_s = float(start_samp / sr)
        end_time_s   = float(end_samp / sr)
        chunk_data   = audio_data[start_samp:end_samp]

        # 1) Compute integrated LUFS if chunk >= 0.4s
        chunk_duration_s = end_time_s - start_time_s
        if chunk_duration_s >= 0.4:
            lufs_val = meter.integrated_loudness(chunk_data)
        else:
            lufs_val = 0.0

        # 2) Find frames in time_matrix_small that fall in [start_time_s, end_time_s]
        sub_frames = [
            row for row in time_matrix_small
            if row["time_s"] >= start_time_s and row["time_s"] < end_time_s
        ]
        if not sub_frames:
            # no frames in this chunk => store zeros & continue
            results.append({
                "chunk_index": chunk_index,
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "lufs": float(lufs_val),
                "avg_tempo_bpm": 0.0,
                "avg_pitch_hz": 0.0,
                "avg_jitter": 0.0,
                "avg_shimmer": 0.0,
                "avg_hnr": 0.0,
                "avg_formants": {"F1":0.0, "F2":0.0, "F3":0.0},
                "vibrato_extent": 0.0,
                "vibrato_rate": 0.0
            })
            chunk_index += 1
            idx += hop_size
            continue

        # 3) Aggregations from sub_frames (tempo, pitch, jitter, shimmer, hnr, formants, etc.)
        tempo_vals   = [f["tempo_bpm"] for f in sub_frames if f["tempo_bpm"] > 0.0]
        pitch_vals   = [f["pitch_hz"]  for f in sub_frames if f["pitch_hz"]  > 0.0]
        jitter_vals  = [f["jitter"]    for f in sub_frames if f["jitter"]    > 0.0]
        shimmer_vals = [f["shimmer"]   for f in sub_frames if f["shimmer"]   > 0.0]
        hnr_vals     = [f["praat_hnr"] for f in sub_frames if f["praat_hnr"] > 0.0]

        # average formants
        f1_list = []
        f2_list = []
        f3_list = []
        for f in sub_frames:
            form_dict = f.get("formants", {})
            F1 = form_dict.get("F1", 0.0)
            F2 = form_dict.get("F2", 0.0)
            F3 = form_dict.get("F3", 0.0)
            if F1>0: f1_list.append(F1)
            if F2>0: f2_list.append(F2)
            if F3>0: f3_list.append(F3)

        avg_tempo   = float(np.mean(tempo_vals))   if tempo_vals   else 0.0
        avg_pitch   = float(np.mean(pitch_vals))   if pitch_vals   else 0.0
        avg_jitter  = float(np.mean(jitter_vals))  if jitter_vals  else 0.0
        avg_shimmer = float(np.mean(shimmer_vals)) if shimmer_vals else 0.0
        avg_hnr     = float(np.mean(hnr_vals))     if hnr_vals     else 0.0
        mean_f1     = float(np.mean(f1_list))      if f1_list      else 0.0
        mean_f2     = float(np.mean(f2_list))      if f2_list      else 0.0
        mean_f3     = float(np.mean(f3_list))      if f3_list      else 0.0

        # 4) Vibrato (real approach):
        #    - Convert pitch to "cents" relative to chunk's average pitch
        #    - vibrato_extent = std of that contour
        #    - vibrato_rate = #peak cycles per second
        if len(pitch_vals) < 2 or avg_pitch < 1.0:
            vib_extent = 0.0
            vib_rate   = 0.0
        else:
            # Convert pitch to cents around chunk's average pitch
            pitch_cents = []
            for p in pitch_vals:
                cents_diff = 1200.0 * math.log2(p / avg_pitch) if p>0.0 else 0.0
                pitch_cents.append(cents_diff)
            vib_extent = float(np.std(pitch_cents))

            # quick approach for vib_rate: find peaks in pitch_cents around zero
            # We'll center it at 0 by subtracting mean if you prefer
            # but since we used chunk's avg pitch, pitch_cents are ~0 mean anyway.
            # We'll find positive peaks & negative peaks => sum
            # then convert to "cycles / second" by chunk duration. 
            arr = np.array(pitch_cents)
            # find positive peaks
            pos_peaks, _ = find_peaks(arr, height=2.0, distance=2)  # tuned params
            # find negative peaks
            neg_peaks, _ = find_peaks(-arr, height=2.0, distance=2)
            # total # of peaks
            total_peaks = len(pos_peaks) + len(neg_peaks)

            # Each vibrato "cycle" has at least 1 positive + 1 negative peak,
            # so # cycles ~ total_peaks / 2
            chunk_dur_s = end_time_s - start_time_s
            if chunk_dur_s > 0:
                vib_rate = (total_peaks / 2.0) / chunk_dur_s
            else:
                vib_rate = 0.0

        row = {
            "chunk_index": chunk_index,
            "start_time_s": start_time_s,
            "end_time_s": end_time_s,
            "lufs": float(lufs_val),
            "avg_tempo_bpm": avg_tempo,
            "avg_pitch_hz":  avg_pitch,
            "avg_jitter":    avg_jitter,
            "avg_shimmer":   avg_shimmer,
            "avg_hnr":       avg_hnr,
            "avg_formants": {
                "F1": mean_f1,
                "F2": mean_f2,
                "F3": mean_f3
            },
            "vibrato_extent": vib_extent,
            "vibrato_rate":   vib_rate
        }
        results.append(row)

        chunk_index += 1
        idx += hop_size

    return results




# ============== Summaries from time_matrix_small =============
def build_summary(time_matrix_small,time_matrix_tempo_large):
    pitch_devs = [abs(row["deviation_cents"]) for row in time_matrix_small]
    if pitch_devs:
        mean_dev = float(np.mean(pitch_devs))
        std_dev = float(np.std(pitch_devs))
    else:
        mean_dev, std_dev = 0.0, 0.0

    tone_vals = [row["tone_to_noise"] for row in time_matrix_small]
    if tone_vals:
        tone_mean = float(np.mean(tone_vals))
        tone_std  = float(np.std(tone_vals))
    else:
        tone_mean, tone_std = 0.0, 0.0

    hnr_vals = [row["praat_hnr"] for row in time_matrix_small if row["praat_hnr"] > 0]
    if hnr_vals:
        hnr_mean = float(np.mean(hnr_vals))
        hnr_std  = float(np.std(hnr_vals))
    else:
        hnr_mean, hnr_std = 0.0, 0.0

    rms_vals = [row["rms_db"] for row in time_matrix_small]
    if rms_vals:
        avg_rms = float(np.mean(rms_vals))
        min_rms = float(np.min(rms_vals))
        max_rms = float(np.max(rms_vals))
        std_rms = float(np.std(rms_vals))
    else:
        avg_rms = min_rms = max_rms = std_rms = 0.0

    lufs_vals = [row["lufs"] for row in time_matrix_tempo_large]
    if lufs_vals:
        avg_lufs = float(np.mean(lufs_vals))
        min_lufs = float(np.min(lufs_vals))
        max_lufs = float(np.max(lufs_vals))
        std_lufs = float(np.std(lufs_vals))
    else:
        avg_lufs = min_lufs = max_lufs = std_lufs = 0.0

    summary = {
        "pitch_deviation": {
            "mean": mean_dev,
            "std": std_dev,
            "deviation_threshold": float(DEVIATION_THRESHOLD)
        },
        "tone_to_noise_ratio": {
            "mean": tone_mean,
            "std": tone_std
        },
        "praat": {
            "hnr_mean": hnr_mean,
            "hnr_std": hnr_std
        },
        "dynamics": {
            "rms_db": {
                "mean": float(avg_rms),
                "min": float(min_rms),
                "max": float(max_rms),
                "range": float(max_rms - min_rms),
                "std": float(std_rms),
                "dynamic_range": float(max_rms - min_rms)
            },
            "lufs": {
                "mean": float(avg_lufs),
                "min": float(min_lufs),
                "max": float(max_lufs),
                "range": float(max_lufs - min_lufs),
                "std": float(std_lufs),
                "dynamic_range": float(max_lufs - min_lufs)
            }
        }
    }
    return summary

# ============== Master analysis function =============
def analyze_audio_file(file_path):
    """
    1) Preprocess
    2) Build spectral_data
    3) time_matrix_small
    4) Medium/large tempo
    5) advanced note features
    6) summary
    7) final JSON
    """
    # Preprocess
    harmonic_audio, sr = preprocess_audio(file_path)


    # 2) Attempt to detect drone
    drone_pitch = detect_drone_pitch(harmonic_audio, sr)

    if drone_pitch > 1.0:
        # Found a strong drone
        final_base_pitch = drone_pitch
        #print(f"Drone found ~ {final_base_pitch:.2f} Hz, using as base pitch.")
    else:
        # fallback to pyin
        pyin_pitch = detect_base_pitch_with_pyin(harmonic_audio, sr, fmin=MIN_F0, fmax=MAX_F0)
        final_base_pitch = pyin_pitch
        #print(f"No dominant drone; fallback => base pitch ~ {final_base_pitch:.2f} Hz")

    # Spectral data (pad if short)
    #spectral_data = build_spectral_data(harmonic_audio, sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)

    # time_matrix_small (pad if short)
    time_matrix_small, S_mag, S_power, S_log, mfcc_summary = build_time_matrix_small(
    audio_data=harmonic_audio,
    sr=sr,
    file_path=file_path,
    base_pitch=final_base_pitch,
    frame_size=FRAME_SIZE,
    hop_length=HOP_LENGTH,
    n_mfcc=N_MFCC
    )

    # tempo-based bigger chunks
    #time_matrix_tempo_medium = build_tempo_matrix(harmonic_audio, sr, TEMPO_CHUNK_SIZE_MEDIUM, overlap=0.5)
    #time_matrix_tempo_large  = build_tempo_matrix(harmonic_audio, sr, TEMPO_CHUNK_SIZE_LARGE, overlap=0.5)

    # advanced note features
    #advanced_note_features = build_advanced_note_features(harmonic_audio, sr)

    time_matrix_tempo_large = build_tempo_and_advanced_features(
        audio_data=harmonic_audio,
        sr=sr,
        time_matrix_small=time_matrix_small,
        chunk_size=TEMPO_CHUNK_SIZE_LARGE,   # 22050
        hop_size=VOCAL_FEATURE_CHUNK_HOP     # 4096
    )


    # summary
    summary_data = build_summary(time_matrix_small, time_matrix_tempo_large)
    summary_data["base_pitch"] = float(final_base_pitch)
    summary_data["spectral_summary"] = mfcc_summary

    analysis_dict = {
        "file_name": os.path.basename(file_path),
        "sample_rate": sr,
        "summary": summary_data,
        "time_matrices": {
            "time_matrix_small": time_matrix_small,
            #"time_matrix_tempo_medium": time_matrix_tempo_medium,
            "time_matrix_tempo_large": time_matrix_tempo_large
        },
        #"spectral_data": spectral_data,
        #"advanced_note_features": advanced_note_features,
        "advanced_vocal_stats": {
            # placeholders or advanced stats
        }
    }

    return analysis_dict, harmonic_audio, sr


# ============== Final: grade_single_file with DB Insert =============
def grade_single_file(file_name):
    """
    1) Runs analyze_audio_file
    2) Replaces NaNs with None for JSON
    3) Saves to DB if SAVE_TO_DB
    4) Optionally playback or plot
    """
    input_path = os.path.join(INPUT_DIR, file_name)
    analysis_dict, harmonic_audio, sr = analyze_audio_file(input_path)

    # replace NaN => None
    sanitized_analysis_dict = _replace_nan_with_none(analysis_dict)

    if SAVE_TO_DB:
        db = QuantumMusicDB()
        db.create_tables()
        rec_id = db.insert_analysis(
            file_name=file_name,
            sample_rate=sr,
            analysis_data=sanitized_analysis_dict
        )
        #print(f"Inserted analysis record with ID: {rec_id}")
        db.close()

    # Optionally: playback harmonic audio if in IPython
    #try:
    #    display(Audio(data=harmonic_audio, rate=sr))
    #except:
    #    pass

    return sanitized_analysis_dict


def process_file(file, training_dir, output_dir, grade_single_file):
    """
    Helper function to process a single file: runs grade_single_file and then moves the file.
    """
    file_path = os.path.join(training_dir, file)
    print(f"Processing {file}...")
    try:
        # This call is assumed to do the DB insertion internally:
        result = grade_single_file(file)
        # Move the file to output_dir after processing
        shutil.move(file_path, os.path.join(output_dir, file))
        print(f"Processed & moved {file} -> {output_dir}")
        return result
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def process_all_files(
    grade_single_file,
    training_dir="data/trainingdata",
    output_dir="data/trainingdataoutput",
    num_workers=8
):
    """
    Processes all .wav files in `training_dir` using the provided
    `grade_single_file` function (which already inserts its results into the database),
    then moves the processed files to `output_dir` using parallel execution.
    """
    if not callable(grade_single_file):
        raise ValueError("grade_single_file must be a callable that accepts a filename.")

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List only .wav files
    files = [f for f in os.listdir(training_dir) if f.endswith(".wav")]
    if not files:
        print(f"No  .wav files found in {training_dir}")
        return

    # Use ProcessPoolExecutor to process files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit each file as a separate job
        futures = {
            executor.submit(process_file, file, training_dir, output_dir, grade_single_file): file
            for file in files
        }
        # Wait for each future to complete and handle exceptions if any
        for future in concurrent.futures.as_completed(futures):
            file = futures[future]
            try:
                _ = future.result()
            except Exception as exc:
                print(f"Error processing {file}: {exc}")

    print("All .wav files processed.")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "all"], default="single")
    parser.add_argument("--file", type=str, default="Bhairav3Rohan.wav")
    parser.add_argument("--training_dir", type=str, default="data/trainingdata")
    parser.add_argument("--output_dir", type=str, default="data/trainingdataoutput")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    if args.mode == "single":
        process_file(
            file=args.file,
            training_dir=args.training_dir,
            output_dir=args.output_dir,
            grade_single_file=grade_single_file
        )
    else:
        process_all_files(
            grade_single_file=grade_single_file,
            training_dir=args.training_dir,
            output_dir=args.output_dir,
            num_workers=args.num_workers
        )
    print("Analysis complete.")





