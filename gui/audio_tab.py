# gui/audio_tab.py  –  Audio Compression tab
import os, threading
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
from scipy.io import wavfile
from scipy.signal import stft
import customtkinter as ctk
import audio_codec, metrics
from gui.helpers import label, divider


class AudioTab:
    def __init__(self, tab, app, default_wav):
        self.app = app; self.default_wav = default_wav; self.wav_path = ""

        left = ctk.CTkFrame(tab, width=230)
        left.pack(side="left", fill="y", padx=(4,6), pady=4)
        left.pack_propagate(False)
        right = ctk.CTkFrame(tab)
        right.pack(side="right", fill="both", expand=True, pady=4, padx=(0,4))

        label(left, "Audio Codec", 16, bold=True).pack(pady=(12,4))
        divider(left)
        label(left, "Input File").pack(anchor="w", padx=12)
        ctk.CTkButton(left, text="Browse WAV", command=self._browse).pack(fill="x", padx=12, pady=3)
        self.file_lbl = label(left, "No file selected", 10, color="#555")
        self.file_lbl.pack(anchor="w", padx=12)

        divider(left)
        label(left, "Quantisation Bits").pack(anchor="w", padx=12)
        self.q_sl = ctk.CTkSlider(left, from_=4, to=8, number_of_steps=4,
                                   command=lambda v: self.q_lbl.configure(
                                       text=f"{int(float(v))} bits  ({2**int(float(v))} levels)"))
        self.q_sl.set(5); self.q_sl.pack(fill="x", padx=12)
        self.q_lbl = label(left, "5 bits  (32 levels)", 10, color="#2563eb")
        self.q_lbl.pack(anchor="w", padx=12)

        divider(left)
        self.run_btn = ctk.CTkButton(left, text="Run Compression", command=self._run)
        self.run_btn.pack(fill="x", padx=12, pady=8)

        box = ctk.CTkFrame(left, corner_radius=6)
        box.pack(fill="x", padx=12, pady=4)
        label(box, "Results", bold=True).pack(pady=(6,2))
        self.snr_lbl = label(box, "SNR        :  --", color="#2563eb"); self.snr_lbl.pack(anchor="w", padx=10)
        self.cr_lbl  = label(box, "Comp Ratio :  --", color="#2563eb"); self.cr_lbl.pack(anchor="w", padx=10)
        self.sz_lbl  = label(box, "Size       :  --", 10, color="#555"); self.sz_lbl.pack(anchor="w", padx=10, pady=(0,6))
        self.status = label(left, "Select a file then click Run", 10, color="#555")
        self.status.pack(pady=4)

        self.fig, self.axes = plt.subplots(3, 1, figsize=(9, 6.5))
        self.fig.patch.set_facecolor("white")
        self.fig.subplots_adjust(hspace=0.6, top=0.94, bottom=0.06, left=0.09, right=0.97)
        for ax in self.axes:
            ax.set_facecolor("white"); ax.tick_params(labelsize=7)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.draw()

    def _browse(self):
        p = filedialog.askopenfilename(filetypes=[("WAV files","*.wav"),("All files","*.*")])
        if p:
            self.wav_path = p
            self.file_lbl.configure(text=os.path.basename(p)[:30])

    def _run(self):
        if not self.wav_path:
            self.status.configure(text="Please select a WAV file first"); return
        self.run_btn.configure(state="disabled", text="Processing...")
        self.status.configure(text="Encoding...")

        def worker():
            try:
                bits = int(float(self.q_sl.get()))
                enc, meta, orig_b = audio_codec.encode(self.wav_path, bits)
                rec = audio_codec.decode(enc, meta)
                fs, raw = wavfile.read(self.wav_path)
                if raw.ndim > 1: raw = raw[:, 0]
                orig = raw.astype(np.float32) / 32768.0
                n    = min(len(orig), len(rec))
                snr  = metrics.snr_db(orig[:n], rec[:n])
                cr   = metrics.comp_ratio(orig_b, len(enc))
                is_test = os.path.abspath(self.wav_path) == os.path.abspath(self.default_wav)
                clean, noisy = audio_codec.gen_wav(None) if is_test else (None, None)
                self.app.after(0, lambda: self._done(orig[:n], rec[:n], fs, snr, cr,
                                                      orig_b, len(enc), clean, noisy))
            except Exception as e:
                err = str(e)
                self.app.after(0, lambda: self._on_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_error(self, msg):
        self.status.configure(text=f"Error: {msg}")
        self.run_btn.configure(state="normal", text="Run Compression")

    def _done(self, orig, rec, fs, snr, cr, orig_b, comp_b, clean, noisy):
        self.snr_lbl.configure(text=f"SNR        :  {snr:.2f} dB")
        self.cr_lbl.configure( text=f"Comp Ratio :  {cr:.2f}x")
        self.sz_lbl.configure( text=f"Size  {orig_b//1024} KB → {comp_b//1024} KB")
        self.status.configure( text="Done")
        self.run_btn.configure(state="normal", text="Run Compression")
        ax0, ax1, ax2 = self.axes

        if clean is not None:
            t = np.linspace(0, audio_codec.DUR, len(clean))
            ax0.clear(); ax0.set_facecolor("white")
            ax0.plot(t, clean, lw=0.8, color="#2563eb")
            ax0.set_title("Clean Signal  (440 + 880 + 1760 Hz)", fontsize=9)
            ax0.set_ylabel("Amplitude", fontsize=7); ax0.tick_params(labelsize=7)

            ax1.clear(); ax1.set_facecolor("white")
            ax1.plot(t, noisy, lw=0.8, color="#dc2626")
            ax1.axvspan(0, 1, alpha=0.12, color="red",    label="Noise region")
            ax1.axvspan(2, 3, alpha=0.12, color="orange", label="Silence region")
            ax1.set_title("Noisy Signal  (noise sec 1, silence sec 3)", fontsize=9)
            ax1.set_ylabel("Amplitude", fontsize=7); ax1.tick_params(labelsize=7)
            ax1.legend(fontsize=7)
        else:
            t = np.linspace(0, len(orig)/fs, len(orig))
            ax0.clear(); ax0.set_facecolor("white")
            ax0.plot(t, orig, lw=0.4, color="#2563eb")
            ax0.set_title("Input Waveform", fontsize=9)
            ax0.set_ylabel("Amplitude", fontsize=7); ax0.tick_params(labelsize=7)

            ax1.clear(); ax1.set_facecolor("white")
            _, _, Z = stft(orig, fs=fs, nperseg=512, noverlap=384)
            mag_db  = 20 * np.log10(np.abs(Z) + 1e-9)
            ax1.imshow(mag_db, aspect="auto", origin="lower", cmap="viridis",
                       extent=[0, len(orig)/fs, 0, fs/2000],
                       vmin=float(np.percentile(mag_db, 5)), vmax=float(mag_db.max()))
            ax1.set_title("STFT Spectrogram", fontsize=9)
            ax1.set_xlabel("Time (s)", fontsize=7); ax1.set_ylabel("Freq (kHz)", fontsize=7)
            ax1.tick_params(labelsize=7)

        win = min(len(orig), fs)
        t   = np.linspace(0, 1.0, win)
        ax2.clear(); ax2.set_facecolor("white")
        ax2.plot(t, orig[:win], lw=0.8, color="#2563eb", label="Original")
        ax2.plot(t, rec[:win],  lw=0.8, color="#dc2626", alpha=0.85, label="Reconstructed")
        ax2.set_title(f"Original vs Reconstructed  –  SNR={snr:.2f} dB  CR={cr:.2f}x", fontsize=9)
        ax2.set_ylabel("Amplitude", fontsize=7); ax2.set_xlabel("Time (s)", fontsize=7)
        ax2.legend(fontsize=7); ax2.tick_params(labelsize=7)
        self.canvas.draw()
