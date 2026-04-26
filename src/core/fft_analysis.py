"""FFT analysis and frequency-domain plotting functions extracted from engine.py."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import coherence, find_peaks

from ..visualization import plots

plt = plots.plt

from .statistics import as_float64_1d


def fft_analysis(series: pd.Series, *, fs: float = 1.0):
    """FFT analysis returning (freqs, amplitude, phase, peak_indices)."""
    arr = as_float64_1d(series.dropna().values)
    if arr.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    n = int(arr.size)
    fs = float(fs) if fs and np.isfinite(fs) and fs > 0 else 1.0
    dt = 1.0 / fs
    freqs = np.fft.fftfreq(n, d=dt)
    fft_vals = fft(arr)
    amplitude = np.abs(fft_vals)
    phase = np.angle(fft_vals)
    pos_mask = freqs >= 0
    freqs, amplitude, phase = freqs[pos_mask], amplitude[pos_mask], phase[pos_mask]
    height_thr = np.max(amplitude) * 0.2 if amplitude.size > 0 else 0
    peaks, _ = find_peaks(amplitude, height=height_thr)
    return freqs, amplitude, phase, peaks


def frequency_analysis(series: pd.Series, peak_height_ratio: float = 0.2, *, fs: float = 1.0):
    """Return (peak_freqs, peak_amplitudes, periods) or (None, None, None)."""
    freqs, amplitude, phase, peaks = fft_analysis(series, fs=fs)
    if freqs.size == 0 or peaks.size == 0:
        return None, None, None
    peak_freqs = freqs[peaks]
    peak_amps = amplitude[peaks]
    periods = 1 / peak_freqs
    return peak_freqs, peak_amps, periods


def plot_coherence_vs_frequency(
    series1: pd.Series,
    series2: pd.Series,
    title: str,
    *,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
) -> BytesIO:
    """Plot coherence as a function of frequency for two series."""
    s1 = as_float64_1d(series1.dropna().values)
    s2 = as_float64_1d(series2.dropna().values)
    n = int(min(s1.size, s2.size))
    if n <= 3:
        return BytesIO()
    s1, s2 = s1[:n], s2[:n]
    fs = float(fs) if fs and np.isfinite(fs) and fs > 0 else 1.0
    if nperseg is None:
        nperseg = int(max(8, min(64, n // 2)))
    nperseg = int(max(8, min(nperseg, n)))
    freqs, cxy = coherence(s1, s2, fs=fs, nperseg=nperseg, detrend="constant")
    if cxy.size:
        cxy = np.clip(np.asarray(cxy, dtype=np.float64), 0.0, 1.0)
        cxy[~np.isfinite(cxy)] = np.nan
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs, cxy, label="Когерентность")
    if cxy.size > 0 and np.isfinite(cxy).any():
        max_idx = int(np.nanargmax(cxy))
        max_freq, max_coh = freqs[max_idx], cxy[max_idx]
        ax.plot(max_freq, max_coh, "ro",
                label=f"Макс. связь: {max_coh:.3f} на {max_freq:.3f}Hz")
        ax.annotate(f"{max_freq:.3f} Hz", xy=(max_freq, max_coh),
                    xytext=(max_freq, max_coh + 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05))
    ax.set_title(title)
    ax.set_xlabel("Частота (Hz)")
    ax.set_ylabel("Когерентность")
    ax.set_ylim(0, 1)
    ax.legend()
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf
