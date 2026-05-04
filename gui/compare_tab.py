# gui/compare_tab.py  –  Audio Compare tab with playback
import os, time, threading
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
from scipy.io import wavfile
import customtkinter as ctk
import audio_codec, metrics
from gui.helpers import label, divider


class CompareTab:
    def __init__(self, tab, app):
        self.app = app; self.cmp_path = ""
        self._orig = self._orig_raw = self._proc = None
        self._fs = 44100; self._dur = 0.0
        self._playing = None; self._pos0 = 0; self._t0 = 0.0
        self._line0 = self._line1 = None

        left = ctk.CTkFrame(tab, width=230)
        left.pack(side="left", fill="y", padx=(4,6), pady=4)
        left.pack_propagate(False)
        right = ctk.CTkFrame(tab)
        right.pack(side="right", fill="both", expand=True, pady=4, padx=(0,4))

        label(left, "Audio Compare", 16, bold=True).pack(pady=(12,4))
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
        self.q_sl.set(4); self.q_sl.pack(fill="x", padx=12)
        self.q_lbl = label(left, "4 bits  (16 levels)", 10, color="#2563eb")
        self.q_lbl.pack(anchor="w", padx=12)

        divider(left)
        self.apply_btn = ctk.CTkButton(left, text="Apply Codec", command=self._apply)
        self.apply_btn.pack(fill="x", padx=12, pady=8)

        box = ctk.CTkFrame(left, corner_radius=6)
        box.pack(fill="x", padx=12, pady=4)
        label(box, "Results", bold=True).pack(pady=(6,2))
        self.snr_lbl = label(box, "SNR        :  --", color="#2563eb"); self.snr_lbl.pack(anchor="w", padx=10)
        self.cr_lbl  = label(box, "Comp Ratio :  --", color="#2563eb"); self.cr_lbl.pack(anchor="w", padx=10)
        self.sz_lbl  = label(box, "Size       :  --", 10, color="#555"); self.sz_lbl.pack(anchor="w", padx=10, pady=(0,6))
        self.status = label(left, "Load a WAV to start", 10, color="#555")
        self.status.pack(pady=6)

        # ── two separate plots ───────────────────────────────────
        self.fig, (self.ax0, self.ax1) = plt.subplots(2, 1, figsize=(9, 5.2))
        self.fig.patch.set_facecolor("white")
        self.fig.subplots_adjust(hspace=0.55, top=0.93, bottom=0.07, left=0.08, right=0.97)
        for ax in (self.ax0, self.ax1):
            ax.set_facecolor("white"); ax.tick_params(labelsize=7)

        # ── playback bar (bottom) ────────────────────────────────
        ctrl = ctk.CTkFrame(right)
        ctrl.pack(side="bottom", fill="x", padx=8, pady=(4,8))

        for which, col, txt in [("orig","#2563eb","ORIGINAL"), ("proc","#dc2626","PROCESSED")]:
            fr = ctk.CTkFrame(ctrl); fr.pack(side="left", fill="both", expand=True, padx=10, pady=8)
            label(fr, txt, 11, bold=True, color=col).pack()
            br = ctk.CTkFrame(fr); br.pack(pady=(4,0))
            ctk.CTkButton(br, text="▶  Play", width=85,
                          command=lambda w=which: self._play(w)).pack(side="left", padx=3)
            ctk.CTkButton(br, text="■  Stop", width=75,
                          command=self._stop).pack(side="left", padx=3)
            sl = ctk.CTkSlider(fr, from_=0, to=1, number_of_steps=200,
                               command=lambda v, w=which: self._seek(w, v))
            sl.set(0); sl.pack(fill="x", padx=6, pady=(6,0))
            tl = label(fr, "0:00 / 0:00", 10, color="#555"); tl.pack()
            if which == "orig":
                self.seek_o = sl; self.time_o = tl
                ctk.CTkFrame(ctrl, width=1, fg_color="#dddddd").pack(side="left", fill="y", pady=10)
            else:
                self.seek_p = sl; self.time_p = tl

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    @staticmethod
    def _fmt(s):
        s = max(0.0, float(s))
        return f"{int(s)//60}:{int(s)%60:02d}"

    def _browse(self):
        p = filedialog.askopenfilename(filetypes=[("WAV files","*.wav"),("All files","*.*")])
        if p:
            self.cmp_path = p
            self.file_lbl.configure(text=os.path.basename(p)[:30])
            self.status.configure(text="File loaded – click Apply Codec")

    def _apply(self):
        if not self.cmp_path:
            self.status.configure(text="No file selected!"); return
        self.apply_btn.configure(state="disabled", text="Processing...")
        self.status.configure(text="Encoding...")

        def worker():
            try:
                q = int(float(self.q_sl.get()))
                fs, raw_all = wavfile.read(self.cmp_path)
                raw_clip    = raw_all[:int(fs * audio_codec.MAX_AUDIO_SEC)]
                enc, meta, orig_b = audio_codec.encode(self.cmp_path, q, cutoff_hz=800)
                rec  = audio_codec.decode(enc, meta)
                mono = raw_clip[:,0] if raw_clip.ndim > 1 else raw_clip
                sig  = mono.astype(np.float32) / 32768.0
                n    = min(len(sig), len(rec))
                sig, rec, raw_play = sig[:n], rec[:n], raw_clip[:n]
                snr = metrics.snr_db(sig, rec)
                cr  = metrics.comp_ratio(orig_b, len(enc))
                self.app.after(0, lambda: self._done(sig, rec, raw_play, fs, snr, cr, orig_b, len(enc)))
            except Exception as e:
                err = str(e)
                self.app.after(0, lambda: self._on_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_error(self, msg):
        self.status.configure(text=f"Error: {msg}")
        self.apply_btn.configure(state="normal", text="Apply Codec")

    def _done(self, orig, rec, raw_play, fs, snr, cr, orig_b, comp_b):
        self._orig = orig; self._orig_raw = raw_play
        self._proc = rec;  self._fs = fs; self._dur = len(orig) / fs

        self.snr_lbl.configure(text=f"SNR        :  {snr:.2f} dB")
        self.cr_lbl.configure( text=f"Comp Ratio :  {cr:.2f}x")
        self.sz_lbl.configure( text=f"Size  {orig_b//1024} KB → {comp_b//1024} KB")
        self.status.configure( text="Ready – press Play to hear the difference!")
        self.apply_btn.configure(state="normal", text="Apply Codec")
        ts = self._fmt(self._dur)
        self.time_o.configure(text=f"0:00 / {ts}"); self.time_p.configure(text=f"0:00 / {ts}")
        self.seek_o.set(0); self.seek_p.set(0)

        # downsample full signal for plotting
        N    = len(orig)
        step = max(1, N // 5000)
        idx  = np.arange(0, N, step)
        t_l  = idx / float(fs)
        dur  = self._dur

        # normalize: both relative to original's peak
        # → processed shows its true relative amplitude (smaller = more content removed)
        o_pk = max(np.abs(orig).max(), 1e-9)
        o_l  = orig[idx] / o_pk          # original  → always ±1
        r_l  = rec[idx]  / o_pk          # processed → smaller if content was cut

        # ── ax0 : Original waveform ──────────────────────────────────────
        self.ax0.clear(); self.ax0.set_facecolor("white")
        self.ax0.fill_between(t_l, o_l,  alpha=0.15, color="#2563eb")
        self.ax0.plot(t_l, o_l, lw=0.7, color="#2563eb")
        self.ax0.axhline(0, color="#cccccc", lw=0.5)
        self.ax0.set_xlim(0, dur); self.ax0.set_ylim(-1.3, 1.3)
        self.ax0.set_title("Original Audio  (full quality — all frequencies)",
                           fontsize=9, color="#2563eb", fontweight="bold")
        self.ax0.set_ylabel("Amplitude", fontsize=7); self.ax0.tick_params(labelsize=7)
        self._line0 = self.ax0.axvline(0, color="green", lw=1.2, alpha=0.85, zorder=5)

        # ── ax1 : Original vs Processed overlaid + difference area ───────
        self.ax1.clear(); self.ax1.set_facecolor("white")
        # shade the area where they differ  (big red area = big quality loss)
        self.ax1.fill_between(t_l, o_l, r_l, color="#ef4444", alpha=0.45,
                              label="Quality loss (difference)", zorder=1)
        self.ax1.plot(t_l, r_l, lw=0.8, color="#f97316", alpha=0.95,
                     label="Processed  (low-pass + 4-bit)", zorder=2)
        self.ax1.plot(t_l, o_l, lw=0.8, color="#2563eb", alpha=0.80,
                     label="Original", zorder=3)
        self.ax1.axhline(0, color="#cccccc", lw=0.5)
        self.ax1.set_xlim(0, dur); self.ax1.set_ylim(-1.3, 1.3)
        self.ax1.set_title(
            f"Original  vs  Processed  —  SNR = {snr:.1f} dB  |  CR = {cr:.1f}x  "
            f"(low-pass ≤800 Hz, 4-bit quantisation)",
            fontsize=9, color="#dc2626", fontweight="bold")
        self.ax1.set_ylabel("Amplitude", fontsize=7)
        self.ax1.set_xlabel("Time (s)", fontsize=7)
        self.ax1.legend(fontsize=7, loc="upper right"); self.ax1.tick_params(labelsize=7)
        self._line1 = self.ax1.axvline(0, color="green", lw=1.2, alpha=0.85, zorder=5)

        self.canvas.draw_idle()

    # ── playback helpers ─────────────────────────────────────────────────
    def _norm_orig(self):
        data = self._orig_raw.astype(np.float32)
        pk   = np.abs(data).max()
        return (data / pk * 32700).astype(np.int16) if pk > 0 else data.astype(np.int16)

    def _norm_proc(self):
        data = self._proc.copy()
        pk   = np.abs(data).max()
        return np.clip(data / pk * 0.99, -1.0, 1.0).astype(np.float32) if pk > 0 else data.astype(np.float32)

    def _play(self, which):
        try: import sounddevice as sd
        except ImportError: self.status.configure(text="pip install sounddevice"); return
        if self._orig is None: self.status.configure(text="Apply Codec first!"); return
        sd.stop()
        seek = self.seek_o.get() if which == "orig" else self.seek_p.get()
        self._playing = which; self._t0 = time.time()
        if which == "orig":
            data = self._norm_orig(); self._pos0 = int(seek * len(data))
            sd.play(np.ascontiguousarray(data[self._pos0:]), samplerate=self._fs, latency="low")
        else:
            data = self._norm_proc(); self._pos0 = int(seek * len(data))
            sd.play(np.ascontiguousarray(data[self._pos0:]), samplerate=self._fs, latency="high")
        self.status.configure(text=f"Playing {'original' if which=='orig' else 'processed'}...")
        self._poll()

    def _stop(self):
        try: import sounddevice as sd; sd.stop()
        except ImportError: pass
        self._playing = None; self.status.configure(text="Stopped")

    def _seek(self, which, val):
        if self._orig is None: return
        cur_s = float(val) * self._dur
        ts, cs = self._fmt(self._dur), self._fmt(cur_s)
        if which == "orig": self.time_o.configure(text=f"{cs} / {ts}")
        else:               self.time_p.configure(text=f"{cs} / {ts}")
        if self._playing != which: return
        try:
            import sounddevice as sd; sd.stop(); self._t0 = time.time()
            if which == "orig":
                data = self._norm_orig(); self._pos0 = int(float(val)*len(data))
                sd.play(np.ascontiguousarray(data[self._pos0:]), samplerate=self._fs, latency="low")
            else:
                data = self._norm_proc(); self._pos0 = int(float(val)*len(data))
                sd.play(np.ascontiguousarray(data[self._pos0:]), samplerate=self._fs, latency="high")
        except ImportError: pass

    def _poll(self):
        if self._playing is None or self._orig is None: return
        elapsed = time.time() - self._t0
        cur_s   = min(self._pos0 / self._fs + elapsed, self._dur)
        seek_v  = cur_s / self._dur if self._dur > 0 else 0.0
        ts, cs  = self._fmt(self._dur), self._fmt(cur_s)
        # move the green playhead line on the plot (in ms)
        cur_ms  = min(cur_s * 1000, self.ax0.get_xlim()[1])
        if self._playing == "orig":
            self.seek_o.set(seek_v); self.time_o.configure(text=f"{cs} / {ts}")
            if self._line0: self._line0.set_xdata([cur_ms, cur_ms]); self.canvas.draw_idle()
        else:
            self.seek_p.set(seek_v); self.time_p.configure(text=f"{cs} / {ts}")
            if self._line1: self._line1.set_xdata([cur_ms, cur_ms]); self.canvas.draw_idle()
        if cur_s >= self._dur: self._playing = None; self.status.configure(text="Finished"); return
        self.app.after(150, self._poll)
