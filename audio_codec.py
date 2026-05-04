# audio_codec.py  –  MP3-like audio compressor
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
from collections import Counter
import heapq

FS            = 44100   # sample rate Hz
DUR           = 3       # length of generated test signal (seconds)
MAX_AUDIO_SEC = 45      # clip long files to this many seconds before encoding


def gen_wav(path):
    # build a clean multi-tone signal, then add noise and silence
    t     = np.linspace(0, DUR, FS * DUR, endpoint=False)
    clean = (0.60 * np.sin(2 * np.pi * 440  * t) +
             0.25 * np.sin(2 * np.pi * 880  * t) +
             0.10 * np.sin(2 * np.pi * 1760 * t))
    noisy = clean.copy()
    noisy[: FS]    += 0.15 * np.random.randn(FS)  # add noise to first second
    noisy[FS * 2:]  = 0.0                          # silence in last second
    if path is not None:
        wavfile.write(path, FS, (noisy * 32767).astype(np.int16))
    return clean, noisy


def _build_cb(data):
    # build a Huffman codebook from a list of symbols
    freq = Counter(data)
    heap = [[w, [sym, ""]] for sym, w in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        return {heap[0][1][0]: "0"}
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        for p in a[1:]:
            p[1] = "0" + p[1]
        for p in b[1:]:
            p[1] = "1" + p[1]
        heapq.heappush(heap, [a[0] + b[0]] + a[1:] + b[1:])
    return {sym: code for sym, code in heap[0][1:]}


def encode(wav_path, q_bits=6, cutoff_hz=None):
    # read WAV, apply STFT, mask weak frequencies, quantize, Huffman encode
    fs, raw = wavfile.read(wav_path)
    if raw.ndim > 1:
        raw = raw[:, 0]

    # trim to max length
    max_samples = int(fs * MAX_AUDIO_SEC)
    if len(raw) > max_samples:
        raw = raw[:max_samples]
    sig = raw.astype(np.float32) / 32768.0

    # apply STFT to convert to frequency domain
    _, _, Z = stft(sig, fs=fs, nperseg=1024, noverlap=768)
    mag   = np.abs(Z)
    phase = np.angle(Z)

    # perceptual masking – remove weak or high-frequency content
    if cutoff_hz is not None:
        # low-pass filter: zero all bins above cutoff
        cutoff_bin = int(cutoff_hz * 1024 / fs)
        mag[cutoff_bin:, :] = 0.0
    else:
        # zero the weakest 75% of frequency bins
        mag[mag < np.percentile(mag, 75)] = 0.0

    # uniform quantisation
    levels = 2 ** q_bits
    m_max  = mag.max() + 1e-9
    q      = np.round(mag / m_max * (levels - 1)).astype(np.int16)

    # Huffman encode the flattened quantized values
    flat = q.flatten().tolist()
    cb   = _build_cb(flat)
    bits = "".join(cb[v] for v in flat)
    pad  = (-len(bits)) % 8
    bits += "0" * pad
    enc  = bytes(int(bits[i: i + 8], 2) for i in range(0, len(bits), 8))

    meta = {
        "fs": fs, "shape": q.shape, "phase": phase, "cb": cb,
        "pad": pad, "m_max": m_max, "levels": levels, "orig_len": len(sig)
    }
    return enc, meta, len(raw) * 2   # compressed bytes, metadata, original size


def decode(enc, meta):
    # reverse the encoding: Huffman decode → dequantize → ISTFT
    bits = "".join(f"{b:08b}" for b in enc)
    if meta["pad"]:
        bits = bits[: -meta["pad"]]

    rev  = {v: k for k, v in meta["cb"].items()}
    flat = []
    buf  = ""
    for b in bits:
        buf += b
        if buf in rev:
            flat.append(rev[buf])
            buf = ""

    # make sure we have exactly the right number of values
    target = meta["shape"][0] * meta["shape"][1]
    if len(flat) < target:
        flat += [0] * (target - len(flat))

    q     = np.array(flat[:target], np.int16).reshape(meta["shape"])
    mag_r = q.astype(np.float32) / (meta["levels"] - 1) * meta["m_max"]
    Zr    = mag_r * np.exp(1j * meta["phase"])
    _, rec = istft(Zr, fs=meta["fs"], nperseg=1024, noverlap=768)
    return rec[: meta["orig_len"]].astype(np.float32)
