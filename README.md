# DSP Media Codec Studio
### Digital Signal Processing Practical Exam -- Suez Canal University
### Faculty of Computers and Informatics -- Computer Science Department

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Exam Requirements Coverage](#exam-requirements-coverage)
3. [Audio Compression Pipeline](#audio-compression-pipeline)
4. [Video Compression Pipeline](#video-compression-pipeline)
5. [GUI Features](#gui-features)
6. [Project Structure](#project-structure)
7. [File Descriptions](#file-descriptions)
8. [Metrics and Formulas](#metrics-and-formulas)
9. [System Requirements](#system-requirements)
10. [Installation and Setup](#installation-and-setup)
11. [How to Use the GUI](#how-to-use-the-gui)
12. [Supported File Formats](#supported-file-formats)
13. [Expected Results](#expected-results)
14. [GitHub Upload Instructions](#github-upload-instructions)

---

## Project Overview

**DSP Media Codec Studio** is a complete audio and video compression system implemented from scratch in Python as the practical exam submission for the Digital Signal Processing course at Suez Canal University.

The project implements two full compression pipelines matching every exam requirement precisely:

- **MP3-like Audio Compression** -- STFT, perceptual masking, uniform quantization, Huffman entropy coding, ISTFT reconstruction, SNR evaluation
- **Video Compression (I-frame / P-frame)** -- DCT, JPEG Q-matrix, zig-zag RLE, full-search block matching, motion vectors, global Huffman, PSNR evaluation

Everything runs inside a single modern dark-themed desktop GUI built with CustomTkinter. No web server, no cloud -- pure Python, runs entirely offline.

---

## Exam Requirements Coverage

### Audio Compression (MP3 Encoder Implementation)

| Exam Requirement | Where Implemented | Status |
|-----------------|-------------------|--------|
| Create raw PCM WAV with noise and silence | `audio_codec.gen_wav()` -- 440+880+1760 Hz tone, noise sec 1, silence sec 3 | DONE |
| Plot audio signal WITHOUT noise | GUI Plot 1 -- Clean Signal (pure 440+880+1760 Hz tone) | DONE |
| Plot audio signal WITH noise and silence | GUI Plot 2 -- Noisy+Silence with red/gold region highlights | DONE |
| Apply STFT (Short-Time Fourier Transform) | `audio_codec.encode()` -- scipy.signal.stft, 1024-sample window, 75% overlap | DONE |
| Split audio into frequency bands | STFT naturally produces 513 frequency bins per frame | DONE |
| Quantization and Bit Allocation | Uniform scalar quantization to 2^q_bits levels (4-8 bits, default 7) | DONE |
| Huffman entropy encoding | `audio_codec._build_cb()` -- optimal variable-length codebook | DONE |
| Compare original vs decompressed | GUI Plot 3 -- 1-second zoom overlay with Error x5 gold shading | DONE |
| Signal-to-Noise Ratio (SNR) | `metrics.snr_db()` -- standard formula, displayed to 2 decimal places | DONE |

### Video Compression

| Exam Requirement | Where Implemented | Status |
|-----------------|-------------------|--------|
| Create video frame-by-frame | `video_codec.gen_video()` -- 30 synthetic frames, moving ball on gradient | DONE |
| Convert frames to YUV color space | `video_codec.rgb2yuv()` -- BT.601 matrix, 4:2:0 chroma subsampling | DONE |
| Choose I-frames and P-frames (every 10th = I-frame) | `video_codec.encode_video()` -- configurable GOP via GUI slider | DONE |
| Apply DCT on 8x8 blocks (I-frame) | `video_codec._dct8()` -- scipy.fft.dctn with ortho normalization | DONE |
| Quantize DCT coefficients | `video_codec._qmat()` -- scaled JPEG luminance Q-matrix | DONE |
| Zig-zag scan and RLE | `video_codec._ZZ` list + `_rle()` / `_rle_dec()` | DONE |
| Motion estimation -- block matching (P-frame) | `video_codec._motion_est()` -- 16x16 blocks, full search, SAD metric | DONE |
| Compute motion vectors | Stored per block as (dy, dx) tuples | DONE |
| Encode motion vectors and residuals | Residuals Huffman-encoded; motion vectors stored in bitstream | DONE |
| Huffman entropy coding on residuals | `video_codec._huff_build()` -- global codebook across all P-frames | DONE |
| Bitstream with headers and frame type indicators | pickle payload: fps, h, w, Q-matrix, GOP, codebook, frame type ('I'/'P') | DONE |
| Compare original and decoded video | GUI -- 3-panel: Original / Reconstructed / Difference x10 | DONE |
| Compression ratio | `metrics.comp_ratio()` -- orig_bytes / comp_bytes | DONE |
| PSNR (Peak Signal-to-Noise Ratio) | `metrics.psnr_db()` -- standard formula averaged over all frames | DONE |

---

## Audio Compression Pipeline

**Step 1 -- Signal Generation** (`gen_wav`): Synthesises a 3-second test WAV at 44100 Hz. The clean signal is a harmonic tone: 440 Hz (0.60 amplitude) + 880 Hz (0.25) + 1760 Hz (0.10). The noisy version adds Gaussian noise (sigma=0.15) to the first second and forces silence (zero amplitude) in the last second. Both signals are returned for separate plotting as required by the exam.

**Step 2 -- STFT Transform** (`encode`): Applies Short-Time Fourier Transform with a 1024-sample Hann window and 75% overlap (noverlap=768), producing a complex spectrogram of shape (513 frequency bins x N time frames). This splits the audio into 513 frequency bands simultaneously.

**Step 3 -- Perceptual Masking**: Zeroes all STFT magnitude coefficients below the 55th-percentile energy threshold, mimicking auditory masking -- quiet sounds near loud ones are inaudible. Discarding ~55% of coefficients saves bits with minimal quality loss.

**Step 4 -- Uniform Scalar Quantization**: Normalises masked magnitudes to [0, 1] and rounds to 2^q_bits discrete levels (default q_bits=7 = 128 levels). The integer indices are the values sent to the entropy coder.

**Step 5 -- Huffman Encoding**: Counts symbol frequencies, builds an optimal Huffman tree via a min-heap, and packs all quantised values into a compact bitstream. Padding bits are stored in metadata for exact byte-aligned recovery.

**Step 6 -- Decoding** (`decode`): Huffman-decodes the bitstream, dequantises indices back to float magnitudes, reconstructs the complex spectrogram using stored phase, and applies Inverse STFT to recover the time-domain audio signal.

**Step 7 -- Evaluation**: SNR measures reconstruction quality in dB. Compression ratio compares original byte count (samples x 2 bytes per int16) to compressed bitstream size.

**Performance note**: Audio longer than 10 seconds is automatically truncated to the first 10 seconds before encoding, keeping the pure-Python Huffman step fast while still demonstrating the full pipeline.

---

## Video Compression Pipeline

**Step 1 -- Frame Collection**: Reads frames from standard containers (MP4, AVI, MOV, MKV) or raw YUV files (.yuv, .y4m). Limited to 60 frames with step-based sampling. Frames resize to max 480x272 pixels, aligned to 16-pixel boundaries.

**Step 2 -- Colour Conversion**: RGB to YUV using BT.601 standard matrix with 4:2:0 chroma subsampling. Y (luma) is kept full resolution; U and V (chroma) are downsampled 2x in both dimensions, exploiting the eye's lower chroma sensitivity.

**Step 3 -- I-frame Encoding** (every GOP-th frame): Y channel split into 8x8 blocks. Each block: shift to [-128,127], 2D DCT (scipy.fft.dctn ortho norm), divide by scaled JPEG Q-matrix, round, read in zig-zag order (64 coefficients), run-length encode as (zero-count, value) pairs.

**Step 4 -- P-frame Encoding** (all other frames): For each 16x16 block in the current frame, full search over a +-2 pixel radius finds the best matching block in the reference (previous decoded) frame using Sum of Absolute Differences (SAD). Motion vector (dy, dx) and residual (current - predicted) are stored.

**Step 5 -- Global Huffman Coding**: After a first pass over all frames, all P-frame residual values are collected and a single global Huffman codebook is built. This one codebook entropy-encodes every P-frame residual in a second pass. The codebook is stored in the bitstream header.

**Step 6 -- Bitstream Formation**: All encoded data packed into a Python pickle structure containing: fps, frame dimensions (h, w), Q-matrix, GOP size, global Huffman codebook, and a list of encoded frames each tagged 'I' or 'P'.

**Step 7 -- Decoding**: Unpack bitstream, Huffman-decode P-frame residuals, reconstruct I-frames via IDCT, reconstruct P-frames via motion compensation (reference block + residual), upsample U/V chroma, convert YUV to RGB.

**Step 8 -- Evaluation**: PSNR averaged over all frame pairs. Compression ratio = raw bytes / bitstream bytes.

---

## GUI Features

| Feature | Detail |
|---------|--------|
| Audio Plot 1 (Generated WAV) | Clean signal -- no noise -- 440+880+1760 Hz pure tone |
| Audio Plot 1 (External WAV) | Full original waveform over time |
| Audio Plot 2 (Generated WAV) | Noisy+Silence signal with red (noise) and gold (silence) region highlights |
| Audio Plot 2 (External WAV) | STFT Spectrogram (magma colormap, frequency vs time) |
| Audio Plot 3 | 1-second zoom: Original vs Reconstructed + Error x5 gold shading |
| Video Panel 1 | Original frame at selected index |
| Video Panel 2 | Reconstructed (decoded) frame at same index |
| Video Panel 3 | Difference x10 -- bright areas show compression artifacts |
| Frame Navigator | Slider to browse any decoded frame |
| Live metrics | SNR / PSNR / CR / file sizes -- 2 decimal places throughout |
| Progress counter | Frame X/N, percentage, elapsed time, ETA during encoding |
| YUV settings | Width/Height/Format fields appear automatically for .yuv files |
| Non-blocking UI | Background threads -- GUI never freezes |
| Scrollable panels | All controls accessible at any window height |

---

## Project Structure

```
DSP PROJECT/
|
+-- main.py            # GUI application (entry point) -- CustomTkinter, 500 lines
+-- audio_codec.py     # MP3-like audio compression pipeline, 108 lines
+-- video_codec.py     # I-frame / P-frame video compression pipeline, 285 lines
+-- metrics.py         # SNR, PSNR, Compression Ratio, 23 lines
+-- requirements.txt   # Python package dependencies
+-- README.md          # This documentation file
```

---

## File Descriptions

### main.py (500 lines)
GUI entry point built with CustomTkinter dark mode. Two tabs: Audio Compression and Video Compression. Uses threading.Thread so the GUI never freezes. Progress polling via self.after(400, poll). Smart audio plot logic: generated WAV shows clean/noisy signals as exam requires; external WAV shows waveform and STFT spectrogram.

### audio_codec.py (108 lines)
- `gen_wav(path)` -- generates 3-second test WAV, returns (clean_signal, noisy_signal) for plotting
- `_build_cb(data)` -- builds Huffman codebook via min-heap
- `encode(wav_path, q_bits=7)` -- STFT -> masking -> quantise -> Huffman -> bytes; truncates to 10s
- `decode(enc, meta)` -- Huffman -> dequantise -> ISTFT -> float32 signal

### video_codec.py (285 lines)
- `gen_video(path)` -- 30-frame synthetic test video
- `encode_video(path, gop, quality, yuv_w, yuv_h, yuv_fmt, progress_cb)` -- full encode with progress
- `decode_video(bs)` -- full decode pipeline
- `read_yuv_frames(path, width, height, fmt)` -- reads .yuv (420/422/444) and .y4m

### metrics.py (23 lines)
- `snr_db(orig, recon)` -- SNR in dB
- `psnr_db(orig_frames, recon_frames)` -- average PSNR in dB across all frames
- `comp_ratio(orig_bytes, comp_bytes)` -- compression ratio

---

## Metrics and Formulas

**Signal-to-Noise Ratio (SNR)**
```
SNR (dB) = 10 * log10( mean(original^2) / mean((original - reconstructed)^2) )
```
Signal power is the mean squared amplitude of the original input. Noise power is the mean squared error between original and reconstructed. Epsilon 1e-12 prevents log(0) when signals are identical.

**Peak Signal-to-Noise Ratio (PSNR)**
```
MSE  = mean over all frames of: mean((original_frame - reconstructed_frame)^2)
PSNR = 10 * log10( 255^2 / MSE )
```
Peak pixel value is 255 (uint8 range). Higher PSNR = better reconstruction. Typical range for this codec: 30-42 dB.

**Compression Ratio (CR)**
```
CR = original_bytes / compressed_bytes
```
CR = 6 means the file is 6x smaller than raw. For audio: original = samples x 2 bytes (int16). For video: original = frames x height x width x 3 bytes (RGB).

---

## System Requirements

- Python 3.9 or higher (3.10+ recommended)
- OS: Windows 10/11, macOS, or Linux
- RAM: 4 GB minimum (8 GB recommended for longer videos)
- Disk: approximately 500 MB free for Python packages

---

## Installation and Setup

### Step 1 -- Install Python
Download from https://www.python.org/downloads/ (Python 3.10+).
On Windows: check "Add Python to PATH" during installation.

```bash
python --version
```

### Step 2 -- Install Git
Download from https://git-scm.com/downloads, accept all defaults.

```bash
git --version
```

### Step 3 -- Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### Step 4 -- Create Virtual Environment (Recommended)
```bash
python -m venv venv
```

Activate:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

### Step 5 -- Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 6 -- Run
```bash
python main.py
```

---

## How to Use the GUI

### Audio Tab -- Exam Demo

1. Click **"Generate Test WAV"** -- creates synthetic WAV. Plot 1 shows clean signal, Plot 2 shows noisy+silence immediately (exam requirement).
2. Set **Quantisation Bits** slider (7 bits default = 128 levels).
3. Click **"Run Compression"**.
4. Results: SNR in dB, Compression Ratio, file sizes.
5. Plot 1: Clean Signal without noise (exam: "Plot audio signal without noise").
6. Plot 2: Noisy+Silence with highlighted regions (exam: "Plot audio signal with noise and silence").
7. Plot 3: 1-second zoom, Original vs Reconstructed, Error x5 shading, SNR and CR in title.

To test with your own file: click **"Browse WAV"** -- Plot 2 switches to STFT Spectrogram.

### Video Tab -- Exam Demo

1. Click **"Generate Test Video"** or **"Browse Video"** (.mp4/.avi/.mov/.mkv/.yuv/.y4m).
2. For .yuv files: Width, Height, Format fields appear automatically.
3. Set **Quality** and **GOP Size**.
4. Click **"Run Compression"** -- frame counter with ETA shows progress.
5. Use **Frame Navigator** slider to browse frames.
6. Three panels: Original / Reconstructed / Difference x10.
7. Results: PSNR in dB, Compression Ratio.

---

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV (PCM) | .wav | Any sample rate, mono or stereo |
| MP4 (H.264) | .mp4 | Best compatibility |
| AVI | .avi | H.264 or MJPEG codec |
| MOV | .mov | H.264 only |
| MKV | .mkv | H.264 only |
| Raw YUV planar | .yuv | Enter Width + Height in GUI |
| YUV4MPEG2 | .y4m | Dimensions read from file header |

For H.265/HEVC videos: convert to H.264 first with VLC (Media > Convert/Save > H.264 profile).

---

## Expected Results

| Metric | Typical Value | Notes |
|--------|--------------|-------|
| Audio SNR | 30-40 dB | Higher bits = higher SNR |
| Audio Compression Ratio | 5-8x | Depends on audio content and silence ratio |
| Video PSNR | 30-42 dB | Higher quality setting = higher PSNR |
| Video Compression Ratio | 2-6x | Depends on motion and GOP size |

---

## GitHub Upload Instructions

### 1. Create Repository
Go to https://github.com, create a new Public repository (do NOT initialise with README).

### 2. Upload from Terminal

Open Command Prompt inside the project folder:
```bash
cd "D:\MyData\DSP PROJECT"
git init
git add .
git commit -m "Initial commit: DSP Media Codec Studio"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### 3. Share
Share the URL `https://github.com/YOUR_USERNAME/YOUR_REPO_NAME` with team members.
They follow Steps 1-6 in the Installation section to clone and run locally.
