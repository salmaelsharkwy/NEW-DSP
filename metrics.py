# metrics.py – evaluation helpers (SNR, PSNR, compression ratio)
import numpy as np


def snr_db(orig, recon):
    """SNR between original and reconstructed signal (dB)."""
    sig_pwr = np.mean(orig.astype(float) ** 2)
    err_pwr = np.mean((orig.astype(float) - recon.astype(float)) ** 2)
    return 10.0 * np.log10(sig_pwr / (err_pwr + 1e-12))


def psnr_db(orig_frames, recon_frames, peak=255.0):
    """Average PSNR over a list of frame pairs (dB)."""
    mse = np.mean(
        [(o.astype(float) - r.astype(float)) ** 2
         for o, r in zip(orig_frames, recon_frames)]
    )
    return 10.0 * np.log10(peak ** 2 / (mse + 1e-12))


def comp_ratio(orig_bytes, comp_bytes):
    """Compression ratio: original size / compressed size."""
    return round(orig_bytes / max(comp_bytes, 1), 2)
