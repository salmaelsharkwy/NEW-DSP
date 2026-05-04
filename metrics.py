# metrics.py – functions to measure compression quality
import numpy as np


def snr_db(original, reconstructed):
    # SNR = 10 * log10(signal power / noise power)
    signal_power = np.mean(original.astype(float) ** 2)
    noise_power  = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    return 10.0 * np.log10(signal_power / (noise_power + 1e-12))


def psnr_db(orig_frames, recon_frames, peak=255.0):
    # PSNR = 10 * log10(peak^2 / MSE)  – averaged over all frames
    total_mse = np.mean([
        np.mean((o.astype(float) - r.astype(float)) ** 2)
        for o, r in zip(orig_frames, recon_frames)
    ])
    return 10.0 * np.log10(peak ** 2 / (total_mse + 1e-12))


def comp_ratio(original_bytes, compressed_bytes):
    # how many times smaller is the compressed file
    return round(original_bytes / max(compressed_bytes, 1), 2)
