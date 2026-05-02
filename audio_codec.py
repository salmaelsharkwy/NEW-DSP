# audio_codec.py - MP3-like encoder/decoder
# Steps: WAV gen -> STFT -> perceptual masking -> quantize -> Huffman -> decode -> SNR
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
from collections import Counter
import heapq

FS            = 44100   # sample rate
DUR           = 3       # seconds for generated test signal
MAX_AUDIO_SEC = 10      # max seconds to encode (keeps Huffman fast for large files)


# 1. Test signal generation

def gen_wav(path):
    """Synthesise: clean tone, then add noise (1st sec) and silence (last sec)."""
    t = np.linspace(0, DUR, FS * DUR, endpoint=False)
    clean = (0.60 * np.sin(2 * np.pi * 440  * t) +
             0.25 * np.sin(2 * np.pi * 880  * t) +
             0.10 * np.sin(2 * np.pi * 1760 * t))
    noisy = clean.copy()
    noisy[: FS]     += 0.15 * np.random.randn(FS)   # noise in 1st second
    noisy[FS * 2:]   = 0.0                            # silence in last second
    wavfile.write(path, FS, (noisy * 32767).astype(np.int16))
    return clean, noisy   # both returned for plotting


# 2. Huffman codebook

def _build_cb(data):
    freq = Counter(data)
    heap = [[w, [s, ""]] for s, w in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        return {heap[0][1][0]: "0"}
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        for p in a[1:]: p[1] = "0" + p[1]
        for p in b[1:]: p[1] = "1" + p[1]
        heapq.heappush(heap, [a[0] + b[0]] + a[1:] + b[1:])
    return {s: c for s, c in heap[0][1:]}


# 3. Encoder

def encode(wav_path, q_bits=6):
    """STFT -> perceptual masking -> uniform quantise -> Huffman pack -> bytes."""
    fs, raw = wavfile.read(wav_path)
    if raw.ndim > 1:
        raw = raw[:, 0]
    # Truncate to MAX_AUDIO_SEC for performance (Huffman is slow on huge files)
    max_samp = int(fs * MAX_AUDIO_SEC)
    if len(raw) > max_samp:
        raw = raw[:max_samp]
    sig = raw.astype(np.float32) / 32768.0

    # STFT - 1024-sample window, 75% overlap
    _, _, Z = stft(sig, fs=fs, nperseg=1024, noverlap=768)
    mag, phase = np.abs(Z), np.angle(Z)

    # perceptual masking: zero coefficients below the 55th-percentile energy
    mag[mag < np.percentile(mag, 55)] = 0.0

    # uniform quantisation to q_bits levels
    levels = 2 ** q_bits
    m_max  = mag.max() + 1e-9
    q      = np.round(mag / m_max * (levels - 1)).astype(np.int16)

    # Huffman encode flattened quantised spectrum
    flat = q.flatten().tolist()
    cb   = _build_cb(flat)
    bits = "".join(cb[v] for v in flat)
    pad  = (-len(bits)) % 8
    bits += "0" * pad
    enc  = bytes(int(bits[i: i + 8], 2) for i in range(0, len(bits), 8))

    meta = dict(fs=fs, shape=q.shape, phase=phase, cb=cb,
                pad=pad, m_max=m_max, levels=levels, orig_len=len(sig))
    return enc, meta, len(raw) * 2   # (compressed bytes, metadata, original bytes)


# 4. Decoder

def decode(enc, meta):
    """Huffman unpack -> dequantise -> ISTFT -> time-domain signal."""
    bits = "".join(f"{b:08b}" for b in enc)
    if meta["pad"]:
        bits = bits[: -meta["pad"]]

    rev  = {v: k for k, v in meta["cb"].items()}
    flat, buf = [], ""
    for b in bits:
        buf += b
        if buf in rev:
            flat.append(rev[buf])
            buf = ""

    # safety: pad/trim to expected size
    target = meta["shape"][0] * meta["shape"][1]
    if len(flat) < target:
        flat += [0] * (target - len(flat))

    q     = np.array(flat[:target], dtype=np.int16).reshape(meta["shape"])
    mag_r = q.astype(np.float32) / (meta["levels"] - 1) * meta["m_max"]
    Zr    = mag_r * np.exp(1j * meta["phase"])
    _, rec = istft(Zr, fs=meta["fs"], nperseg=1024, noverlap=768)
    return rec[: meta["orig_len"]].astype(np.float32)
