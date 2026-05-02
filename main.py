# main.py -- DSP Media Codec Studio GUI
# CustomTkinter dark theme | Audio tab + Video tab
import os
import threading
import numpy as np
import cv2
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
from scipy.io import wavfile

import audio_codec
import video_codec
import metrics

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG    = "#0d1117"
CARD  = "#161b22"
ACC   = "#58a6ff"
GREEN = "#3fb950"
RED   = "#f85149"
GOLD  = "#e3b341"
DIM   = "#30363d"


def lbl(parent, text, size=12, bold=False, color="#e6edf3"):
    w = "bold" if bold else "normal"
    return ctk.CTkLabel(parent, text=text,
                        font=ctk.CTkFont(size=size, weight=w),
                        text_color=color)


def sep(parent):
    ctk.CTkFrame(parent, height=1, fg_color=DIM).pack(
        fill="x", padx=8, pady=5)


def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#8b949e", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(DIM)


class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("DSP Media Codec Studio  -  Suez Canal University")
        self.geometry("1340x860")
        self.configure(fg_color=BG)
        self.resizable(True, True)

        self.clean_sig    = None
        self.noisy_sig    = None
        self.orig_frames  = []
        self.recon_frames = []

        self.tabs = ctk.CTkTabview(self, fg_color=CARD)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=12)
        self.tabs.add("Audio Compression")
        self.tabs.add("Video Compression")
        self.tabs.add("Audio Compare")

        self._build_audio(self.tabs.tab("Audio Compression"))
        self._build_video(self.tabs.tab("Video Compression"))
        self._build_compare(self.tabs.tab("Audio Compare"))

    # ================================================================
    # AUDIO TAB
    # ================================================================

    def _build_audio(self, tab):
        lo = ctk.CTkFrame(tab, width=255, fg_color=CARD, corner_radius=10)
        lo.pack(side="left", fill="y", padx=(0, 8), pady=6)
        lo.pack_propagate(False)

        left = ctk.CTkScrollableFrame(lo, fg_color="transparent",
                                       scrollbar_button_color=DIM,
                                       scrollbar_button_hover_color="#484f58")
        left.pack(fill="both", expand=True)

        lbl(left, "Audio Codec", 18, bold=True).pack(pady=(14, 6))
        sep(left)

        self.wav_path = ctk.StringVar(value="")
        lbl(left, "Input File").pack(padx=12, anchor="w")
        ctk.CTkButton(left, text="Browse WAV",
                      command=self._browse_wav).pack(
                          fill="x", padx=12, pady=3)
        ctk.CTkButton(left, text="Generate Test WAV",
                      fg_color="#238636", hover_color="#2ea043",
                      command=self._gen_wav).pack(
                          fill="x", padx=12, pady=3)
        self.wav_lbl = lbl(left, "", 11, color="gray")
        self.wav_lbl.pack(padx=12, anchor="w")

        sep(left)
        lbl(left, "Quantisation Bits").pack(padx=12, anchor="w", pady=(4, 0))
        self.q_audio = ctk.CTkSlider(left, from_=4, to=8, number_of_steps=4)
        self.q_audio.set(7)
        self.q_audio.pack(fill="x", padx=12)
        self.q_audio.configure(command=self._qa_changed)
        self.q_audio_lbl = lbl(left, "7 bits  (128 levels)", 11, color=ACC)
        self.q_audio_lbl.pack(padx=12, anchor="w")

        sep(left)
        self.btn_audio = ctk.CTkButton(
            left, text="Run Compression",
            fg_color="#1f6feb", hover_color=ACC,
            command=self._run_audio)
        self.btn_audio.pack(fill="x", padx=12, pady=8)

        mc = ctk.CTkFrame(left, fg_color=BG, corner_radius=8)
        mc.pack(fill="x", padx=12, pady=4)
        lbl(mc, "Results", bold=True).pack(pady=(8, 4))
        self.a_snr  = lbl(mc, "SNR        :  --", color=ACC)
        self.a_snr.pack(anchor="w", padx=12)
        self.a_cr   = lbl(mc, "Comp Ratio :  --", color=ACC)
        self.a_cr.pack(anchor="w", padx=12)
        self.a_size = lbl(mc, "Size       :  --", color="gray")
        self.a_size.pack(anchor="w", padx=12, pady=(0, 8))

        self.a_status = lbl(left, "Ready", 11, color="gray")
        self.a_status.pack(pady=6)

        pf = ctk.CTkFrame(tab, fg_color=CARD, corner_radius=10)
        pf.pack(side="right", fill="both", expand=True, pady=6)

        self.a_fig, self.a_axes = plt.subplots(3, 1, figsize=(9, 6.5))
        self.a_fig.patch.set_facecolor(CARD)
        self.a_fig.subplots_adjust(hspace=0.6, top=0.94, bottom=0.06,
                                   left=0.08, right=0.97)
        for ax, t in zip(self.a_axes, ["Clean Signal (440 Hz tone)",
                                        "Noisy + Silence Signal",
                                        "Original vs Reconstructed"]):
            style(ax)
            ax.set_title(t, color="#e6edf3", fontsize=9)

        self.a_canvas = FigureCanvasTkAgg(self.a_fig, master=pf)
        self.a_canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=4, pady=4)

    def _qa_changed(self, val):
        b = int(float(val))
        self.q_audio_lbl.configure(text=f"{b} bits  ({2**b} levels)")

    def _browse_wav(self):
        p = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if p:
            self.wav_path.set(p)
            self.wav_lbl.configure(text=os.path.basename(p)[:34],
                                   text_color=ACC)
            # clear generated signals so _audio_done uses external-WAV mode
            self.clean_sig = None
            self.noisy_sig = None

    def _gen_wav(self):
        p = os.path.join(os.path.expanduser("~"), "test_audio.wav")
        self.clean_sig, self.noisy_sig = audio_codec.gen_wav(p)
        self.wav_path.set(p)
        self.wav_lbl.configure(text="test_audio.wav  [generated]",
                               text_color=GREEN)
        self._plot_audio_input()

    def _plot_audio_input(self):
        if self.clean_sig is None:
            return
        t = np.linspace(0, audio_codec.DUR, len(self.clean_sig))
        ax0, ax1 = self.a_axes[0], self.a_axes[1]
        ax0.clear(); ax1.clear()
        style(ax0); style(ax1)
        ax0.plot(t, self.clean_sig, color=ACC, lw=0.7)
        ax0.set_ylabel("Amplitude", color="#8b949e", fontsize=7)
        ax0.set_title("Clean Signal (no noise)", color="#e6edf3", fontsize=9)
        ax1.plot(t, self.noisy_sig, color="#f78166", lw=0.7)
        ax1.axvspan(0.0, 1.0, alpha=0.12, color=RED)
        ax1.axvspan(2.0, 3.0, alpha=0.12, color=GOLD)
        ax1.set_ylabel("Amplitude", color="#8b949e", fontsize=7)
        ax1.set_title("Noisy + Silence Signal", color="#e6edf3", fontsize=9)
        self.a_canvas.draw()

    def _run_audio(self):
        path = self.wav_path.get()
        if not path:
            self.a_status.configure(text="No file selected!", text_color=RED)
            return
        self.btn_audio.configure(state="disabled", text="Processing...")
        self.a_status.configure(text="Encoding...", text_color=GOLD)

        def worker():
            try:
                q = int(float(self.q_audio.get()))
                enc, meta, orig_b = audio_codec.encode(path, q)
                rec = audio_codec.decode(enc, meta)
                fs, raw = wavfile.read(path)
                if raw.ndim > 1:
                    raw = raw[:, 0]
                orig  = raw.astype(np.float32) / 32768.0
                ln    = min(len(orig), len(rec))
                snr_v = metrics.snr_db(orig[:ln], rec[:ln])
                cr_v  = metrics.comp_ratio(orig_b, len(enc))
                self.after(0, lambda: self._audio_done(
                    orig, rec, fs, snr_v, cr_v, orig_b, len(enc)))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self.a_status.configure(
                    text="Error: " + err, text_color=RED))
                self.after(0, lambda: self.btn_audio.configure(
                    state="normal", text="Run Compression"))

        threading.Thread(target=worker, daemon=True).start()

    def _audio_done(self, orig, rec, fs, snr_v, cr_v, orig_b, comp_b):
        from scipy.signal import stft as _stft
        from matplotlib.patches import Patch

        # -- results panel --
        self.a_snr.configure( text=f"SNR        :  {snr_v:.2f} dB")
        self.a_cr.configure(  text=f"Comp Ratio :  {cr_v:.2f} x")
        self.a_size.configure(text=f"Size  {orig_b//1024} KB -> {comp_b//1024} KB")
        self.a_status.configure(text="Done!", text_color=GREEN)
        self.btn_audio.configure(state="normal", text="Run Compression")

        ln = min(len(orig), len(rec))
        t  = np.linspace(0, ln / fs, ln)
        ax0, ax1, ax2 = self.a_axes

        # ---- Plot 1 ------------------------------------------------
        # Generated WAV  -> clean tone (exam: "plot signal without noise")
        # External WAV   -> full original waveform
        ax0.clear(); style(ax0)
        ax0.set_ylabel("Amplitude", color="#8b949e", fontsize=7)
        ax0.set_xlabel("Time (s)",  color="#8b949e", fontsize=7)
        if self.clean_sig is not None:
            tc = np.linspace(0, audio_codec.DUR, len(self.clean_sig))
            ax0.plot(tc, self.clean_sig, color=ACC, lw=0.8)
            ax0.set_xlim(tc[0], tc[-1])
            ax0.set_title("Clean Signal  (440 + 880 + 1760 Hz, no noise)",
                          color="#e6edf3", fontsize=9)
        else:
            ax0.plot(t, orig[:ln], color=ACC, lw=0.4)
            ax0.set_xlim(t[0], t[-1])
            ax0.set_title("Original Signal  (Input Waveform)",
                          color="#e6edf3", fontsize=9)

        # ---- Plot 2 ------------------------------------------------
        # Generated WAV  -> noisy + silence (exam: "plot signal with noise and silence")
        # External WAV   -> STFT spectrogram
        ax1.clear()
        if self.noisy_sig is not None:
            style(ax1)
            tn = np.linspace(0, audio_codec.DUR, len(self.noisy_sig))
            ax1.plot(tn, self.noisy_sig, color="#f78166", lw=0.8)
            ax1.axvspan(0.0, 1.0, alpha=0.15, color=RED)
            ax1.axvspan(2.0, 3.0, alpha=0.15, color=GOLD)
            ax1.set_xlim(tn[0], tn[-1])
            ax1.set_ylabel("Amplitude", color="#8b949e", fontsize=7)
            ax1.set_xlabel("Time (s)",  color="#8b949e", fontsize=7)
            ax1.set_title(
                "Signal with Noise (sec 1)  &  Silence (sec 3)  [encoder input]",
                color="#e6edf3", fontsize=9)
            leg1 = ax1.legend(
                handles=[Patch(facecolor=RED,  alpha=0.5, label="Noise region"),
                         Patch(facecolor=GOLD, alpha=0.5, label="Silence region")],
                fontsize=7, facecolor=CARD, edgecolor=DIM)
            for txt in leg1.get_texts():
                txt.set_color("#e6edf3")
        else:
            ax1.set_facecolor(BG)
            for sp in ax1.spines.values():
                sp.set_edgecolor(DIM)
            _, _, Z = _stft(orig[:ln], fs=fs, nperseg=512, noverlap=384)
            mag_db  = 20.0 * np.log10(np.abs(Z) + 1e-9)
            vmin    = float(np.percentile(mag_db, 5))
            vmax    = float(mag_db.max())
            nyq_khz = fs / 2000.0
            ax1.imshow(mag_db, aspect="auto", origin="lower", cmap="magma",
                       extent=[0, ln / fs, 0, nyq_khz],
                       vmin=vmin, vmax=vmax)
            ax1.set_xlabel("Time (s)",   color="#8b949e", fontsize=7)
            ax1.set_ylabel("Freq (kHz)", color="#8b949e", fontsize=7)
            ax1.tick_params(colors="#8b949e", labelsize=7)
            ax1.set_title("STFT Spectrogram  (frequency content over time)",
                          color="#e6edf3", fontsize=9)

        # ---- Plot 3: original vs reconstructed (1-second zoom + error) ----
        ax2.clear(); style(ax2)
        win = min(ln, fs)       # first 1 second of samples
        tw  = t[:win]
        ax2.plot(tw, orig[:win], color=ACC,       lw=0.8, zorder=3,
                 label="Original")
        ax2.plot(tw, rec[:win],  color="#f78166", lw=0.8, alpha=0.85,
                 zorder=2, label="Reconstructed")
        err = np.abs(orig[:win].astype(np.float64) -
                     rec[:win].astype(np.float64)) * 5.0
        ax2.fill_between(tw, 0, err, color=GOLD, alpha=0.35, zorder=1,
                         label="Error x5")
        ax2.set_xlim(tw[0], tw[-1])
        ax2.set_title(
            f"Original vs Reconstructed (first 1 s)  |  "
            f"SNR={snr_v:.2f} dB  |  CR={cr_v:.2f}x",
            color="#e6edf3", fontsize=9)
        ax2.set_ylabel("Amplitude", color="#8b949e", fontsize=7)
        ax2.set_xlabel("Time (s)",  color="#8b949e", fontsize=7)
        leg = ax2.legend(fontsize=7, facecolor=CARD, edgecolor=DIM,
                         loc="upper right")
        for txt in leg.get_texts():
            txt.set_color("#e6edf3")

        self.a_fig.canvas.draw_idle()

    # ================================================================
    # VIDEO TAB
    # ================================================================

    def _build_video(self, tab):
        lo = ctk.CTkFrame(tab, width=260, fg_color=CARD, corner_radius=10)
        lo.pack(side="left", fill="y", padx=(0, 8), pady=6)
        lo.pack_propagate(False)

        # scrollable inner frame - nothing gets cut off
        left = ctk.CTkScrollableFrame(lo, fg_color="transparent",
                                       scrollbar_button_color=DIM,
                                       scrollbar_button_hover_color="#484f58")
        left.pack(fill="both", expand=True)
        self._vleft = left

        lbl(left, "Video Codec", 18, bold=True).pack(pady=(14, 6))
        sep(left)

        # file selection
        self.vid_path = ctk.StringVar(value="")
        lbl(left, "Input File").pack(padx=12, anchor="w")
        ctk.CTkButton(left, text="Browse Video",
                      command=self._browse_vid).pack(
                          fill="x", padx=12, pady=3)
        ctk.CTkButton(left, text="Generate Test Video",
                      fg_color="#238636", hover_color="#2ea043",
                      command=self._gen_vid).pack(
                          fill="x", padx=12, pady=3)
        self.vid_lbl = lbl(left, "", 11, color="gray")
        self.vid_lbl.pack(padx=12, anchor="w", pady=(0, 2))

        # YUV settings panel - NOT packed yet (shown only for .yuv/.y4m)
        self.yuv_outer = ctk.CTkFrame(left, fg_color=DIM, corner_radius=6)

        lbl(self.yuv_outer, "YUV Settings (.yuv files only)", 9,
            color="#8b949e").pack(anchor="w", padx=8, pady=(4, 1))
        r1 = ctk.CTkFrame(self.yuv_outer, fg_color="transparent")
        r1.pack(fill="x", padx=6, pady=1)
        lbl(r1, "Width", 10).pack(side="left", padx=(2, 4))
        self.yuv_w = ctk.CTkEntry(r1, width=75, height=26,
                                   placeholder_text="e.g. 352")
        self.yuv_w.pack(side="left")

        r2 = ctk.CTkFrame(self.yuv_outer, fg_color="transparent")
        r2.pack(fill="x", padx=6, pady=1)
        lbl(r2, "Height", 10).pack(side="left", padx=(2, 4))
        self.yuv_h = ctk.CTkEntry(r2, width=75, height=26,
                                   placeholder_text="e.g. 288")
        self.yuv_h.pack(side="left")

        r3 = ctk.CTkFrame(self.yuv_outer, fg_color="transparent")
        r3.pack(fill="x", padx=6, pady=(1, 6))
        lbl(r3, "Format", 10).pack(side="left", padx=(2, 4))
        self.yuv_fmt = ctk.CTkOptionMenu(r3, width=90, height=26,
                                          values=["420", "422", "444"])
        self.yuv_fmt.set("420")
        self.yuv_fmt.pack(side="left")

        # quality and GOP
        sep(left)
        lbl(left, "Quality  (30=low  90=high)").pack(
            padx=12, anchor="w", pady=(4, 0))
        self.q_vid = ctk.CTkSlider(left, from_=30, to=90, number_of_steps=12)
        self.q_vid.set(75)
        self.q_vid.pack(fill="x", padx=12)
        self.q_vid.configure(command=self._qv_changed)
        self.q_vid_lbl = lbl(left, "Quality: 75", 11, color=ACC)
        self.q_vid_lbl.pack(padx=12, anchor="w")

        lbl(left, "GOP  (I-frame interval)").pack(
            padx=12, anchor="w", pady=(6, 0))
        self.gop = ctk.CTkSlider(left, from_=5, to=20, number_of_steps=3)
        self.gop.set(10)
        self.gop.pack(fill="x", padx=12)
        self.gop.configure(command=self._gop_changed)
        self.gop_lbl = lbl(left, "GOP: 10", 11, color=ACC)
        self.gop_lbl.pack(padx=12, anchor="w")

        # run + navigator
        sep(left)
        self.btn_video = ctk.CTkButton(
            left, text="Run Compression",
            fg_color="#1f6feb", hover_color=ACC,
            command=self._run_video)
        self.btn_video.pack(fill="x", padx=12, pady=6)

        lbl(left, "Frame Navigator").pack(padx=12, anchor="w", pady=(2, 0))
        self.frame_sl = ctk.CTkSlider(
            left, from_=0, to=1, number_of_steps=1,
            command=self._show_frame)
        self.frame_sl.pack(fill="x", padx=12)
        self.frame_lbl = lbl(left, "Frame: --", 11, color="gray")
        self.frame_lbl.pack(padx=12, anchor="w")

        # results box
        sep(left)
        mc = ctk.CTkFrame(left, fg_color=BG, corner_radius=8)
        mc.pack(fill="x", padx=12, pady=4)
        lbl(mc, "Results", bold=True).pack(pady=(8, 4))
        self.v_psnr = lbl(mc, "PSNR       :  --", color=ACC)
        self.v_psnr.pack(anchor="w", padx=12)
        self.v_cr   = lbl(mc, "Comp Ratio :  --", color=ACC)
        self.v_cr.pack(anchor="w", padx=12)
        self.v_size = lbl(mc, "Size       :  --", color="gray")
        self.v_size.pack(anchor="w", padx=12, pady=(0, 8))

        self.v_status = lbl(left, "Ready", 11, color="gray")
        self.v_status.pack(pady=(4, 10))

        # plot canvas
        pf = ctk.CTkFrame(tab, fg_color=CARD, corner_radius=10)
        pf.pack(side="right", fill="both", expand=True, pady=6)

        self.v_fig, self.v_axes = plt.subplots(1, 3, figsize=(13, 4.8))
        self.v_fig.patch.set_facecolor(CARD)
        self.v_fig.subplots_adjust(left=0.02, right=0.98,
                                    top=0.90, bottom=0.04, wspace=0.06)
        for ax, t in zip(self.v_axes,
                         ["Original Frame",
                          "Reconstructed Frame",
                          "Difference x10"]):
            ax.set_facecolor(BG); ax.axis("off")
            ax.set_title(t, color="#e6edf3", fontsize=9)

        self.v_canvas = FigureCanvasTkAgg(self.v_fig, master=pf)
        self.v_canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=4, pady=4)

    # video callbacks

    def _qv_changed(self, val):
        self.q_vid_lbl.configure(text=f"Quality: {int(float(val))}")

    def _gop_changed(self, val):
        self.gop_lbl.configure(text=f"GOP: {int(float(val))}")

    def _browse_vid(self):
        p = filedialog.askopenfilename(
            filetypes=[("Video files",
                        "*.mp4 *.avi *.mov *.mkv *.yuv *.y4m"),
                       ("YUV raw",   "*.yuv *.y4m"),
                       ("All files", "*.*")])
        if p:
            self.vid_path.set(p)
            self.vid_lbl.configure(text=os.path.basename(p)[:36],
                                   text_color=ACC)
            ext = p.lower().rsplit('.', 1)[-1]
            if ext in ('yuv', 'y4m'):
                # insert right after vid_lbl in the scrollable frame
                self.yuv_outer.pack(in_=self._vleft,
                                    after=self.vid_lbl,
                                    fill="x", padx=12, pady=(2, 4))
            else:
                self.yuv_outer.pack_forget()

    def _gen_vid(self):
        p = os.path.join(os.path.expanduser("~"), "test_video.mp4")
        video_codec.gen_video(p)
        self.vid_path.set(p)
        self.vid_lbl.configure(text="test_video.mp4  [generated]",
                               text_color=GREEN)
        self.yuv_outer.pack_forget()

    def _run_video(self):
        import time
        path = self.vid_path.get()
        if not path:
            self.v_status.configure(text="No file selected!", text_color=RED)
            return
        self.btn_video.configure(state="disabled", text="Processing...")
        self.v_status.configure(text="Starting...", text_color=GOLD)

        # shared progress state
        self._enc_cur   = [0]
        self._enc_total = [0]
        self._enc_done  = [False]
        self._enc_start = [time.time()]

        def poll():
            if self._enc_done[0]:
                return
            cur   = self._enc_cur[0]
            total = self._enc_total[0]
            elapsed = time.time() - self._enc_start[0]
            if total > 0:
                pct  = int(100 * cur / total)
                eta  = (elapsed / cur * (total - cur)) if cur > 0 else 0
                self.v_status.configure(
                    text=f"Frame {cur}/{total}  {pct}%  "
                         f"{elapsed:.0f}s  ETA:{eta:.0f}s",
                    text_color=GOLD)
            self.after(400, poll)

        def worker():
            try:
                q   = int(float(self.q_vid.get()))
                gop = int(float(self.gop.get()))
                try:
                    yw = int(self.yuv_w.get())
                except Exception:
                    yw = 0
                try:
                    yh = int(self.yuv_h.get())
                except Exception:
                    yh = 0
                yf = self.yuv_fmt.get()

                def on_progress(cur, total):
                    self._enc_cur[0]   = cur
                    self._enc_total[0] = total

                # orig_rgb already RGB (converted inside encode_video)
                bs, orig_rgb, orig_b = video_codec.encode_video(
                    path, gop, q, yuv_w=yw, yuv_h=yh, yuv_fmt=yf,
                    progress_cb=on_progress)
                recon, _ = video_codec.decode_video(bs)
                n = min(len(orig_rgb), len(recon))
                psnr_v = metrics.psnr_db(orig_rgb[:n], recon[:n])
                cr_v   = metrics.comp_ratio(orig_b, len(bs))
                self._enc_done[0] = True
                self.after(0, lambda: self._video_done(
                    orig_rgb[:n], recon[:n],
                    psnr_v, cr_v, orig_b, len(bs)))
            except Exception as e:
                self._enc_done[0] = True
                err = str(e)
                self.after(0, lambda: self.v_status.configure(
                    text="Error: " + err, text_color=RED))
                self.after(0, lambda: self.btn_video.configure(
                    state="normal", text="Run Compression"))

        self.after(400, poll)
        threading.Thread(target=worker, daemon=True).start()

    def _video_done(self, orig_rgb, recon, psnr_v, cr_v, orig_b, comp_b):
        self.orig_frames  = orig_rgb
        self.recon_frames = recon
        self.v_psnr.configure(text=f"PSNR       :  {psnr_v:.2f} dB")
        self.v_cr.configure(  text=f"Comp Ratio :  {cr_v:.2f} x")
        self.v_size.configure(text=f"Size  {orig_b//1024} KB -> {comp_b//1024} KB")
        self.v_status.configure(text="Done!", text_color=GREEN)
        self.btn_video.configure(state="normal", text="Run Compression")
        n = len(recon)
        self.frame_sl.configure(to=max(n - 1, 1),
                                  number_of_steps=max(n - 1, 1))
        self._show_frame(0)

    def _show_frame(self, val):
        idx = int(float(val))
        if not self.orig_frames or idx >= len(self.orig_frames):
            return
        self.frame_lbl.configure(
            text=f"Frame: {idx} / {len(self.orig_frames) - 1}")
        ax0, ax1, ax2 = self.v_axes
        ax0.clear(); ax1.clear(); ax2.clear()
        for ax in self.v_axes:
            ax.set_facecolor(BG); ax.axis("off")

        orig  = self.orig_frames[idx]
        recon = self.recon_frames[idx]

        diff = np.clip(
            np.abs(orig.astype(np.int16) - recon.astype(np.int16)) * 10,
            0, 255).astype(np.uint8)

        ax0.imshow(orig)
        ax0.set_title(f"Original  (frame {idx})",
                      color="#e6edf3", fontsize=9)
        ax1.imshow(recon)
        ax1.set_title(f"Reconstructed  (frame {idx})",
                      color="#e6edf3", fontsize=9)
        ax2.imshow(diff)
        ax2.set_title("Difference x10  (bright = loss)",
                      color=GOLD, fontsize=9)
        self.v_canvas.draw()

    # ================================================================
    # AUDIO COMPARE TAB
    # ================================================================

    @staticmethod
    def _fmt_ms(s):
        s = max(0.0, float(s))
        return f"{int(s) // 60}:{int(s) % 60:02d}"

    def _build_compare(self, tab):
        # ---- left control panel ----
        lo = ctk.CTkFrame(tab, width=255, fg_color=CARD, corner_radius=10)
        lo.pack(side="left", fill="y", padx=(0, 8), pady=6)
        lo.pack_propagate(False)
        left = ctk.CTkScrollableFrame(lo, fg_color="transparent",
                                       scrollbar_button_color=DIM,
                                       scrollbar_button_hover_color="#484f58")
        left.pack(fill="both", expand=True)

        lbl(left, "Audio Compare", 18, bold=True).pack(pady=(14, 6))
        sep(left)

        self.cmp_path = ctk.StringVar(value="")
        lbl(left, "Input File").pack(padx=12, anchor="w")
        ctk.CTkButton(left, text="Browse WAV",
                      command=self._cmp_browse).pack(fill="x", padx=12, pady=3)
        self.cmp_lbl = lbl(left, "", 11, color="gray")
        self.cmp_lbl.pack(padx=12, anchor="w")

        sep(left)
        lbl(left, "Quantisation Bits").pack(padx=12, anchor="w", pady=(4, 0))
        self.cmp_q = ctk.CTkSlider(left, from_=4, to=8, number_of_steps=4)
        self.cmp_q.set(8)
        self.cmp_q.pack(fill="x", padx=12)
        self.cmp_q.configure(command=self._cmp_q_changed)
        self.cmp_q_lbl = lbl(left, "8 bits  (256 levels)", 11, color=ACC)
        self.cmp_q_lbl.pack(padx=12, anchor="w")

        sep(left)
        self.cmp_btn = ctk.CTkButton(
            left, text="Apply Codec",
            fg_color="#1f6feb", hover_color=ACC,
            command=self._cmp_apply)
        self.cmp_btn.pack(fill="x", padx=12, pady=8)

        mc = ctk.CTkFrame(left, fg_color=BG, corner_radius=8)
        mc.pack(fill="x", padx=12, pady=4)
        lbl(mc, "Results", bold=True).pack(pady=(8, 4))
        self.cmp_snr  = lbl(mc, "SNR        :  --", color=ACC)
        self.cmp_snr.pack(anchor="w", padx=12)
        self.cmp_cr   = lbl(mc, "Comp Ratio :  --", color=ACC)
        self.cmp_cr.pack(anchor="w", padx=12)
        self.cmp_size = lbl(mc, "Size       :  --", color="gray")
        self.cmp_size.pack(anchor="w", padx=12, pady=(0, 8))
        self.cmp_status = lbl(left, "Load a WAV to start", 11, color="gray")
        self.cmp_status.pack(pady=6)

        # ---- right panel: waveforms + playback ----
        right = ctk.CTkFrame(tab, fg_color=CARD, corner_radius=10)
        right.pack(side="right", fill="both", expand=True, pady=6)

        pf = ctk.CTkFrame(right, fg_color="transparent")
        pf.pack(fill="both", expand=True, padx=4, pady=(4, 0))

        self.cmp_fig, (self.cmp_ax0, self.cmp_ax1) = plt.subplots(
            2, 1, figsize=(9, 5.2))
        self.cmp_fig.patch.set_facecolor(CARD)
        self.cmp_fig.subplots_adjust(
            hspace=0.50, top=0.93, bottom=0.07, left=0.07, right=0.97)
        for ax, title in zip([self.cmp_ax0, self.cmp_ax1],
                              ["Original Audio", "Processed Audio (Codec Output)"]):
            style(ax)
            ax.set_title(title, color="#e6edf3", fontsize=9)
        self.cmp_canvas = FigureCanvasTkAgg(self.cmp_fig, master=pf)
        self.cmp_canvas.get_tk_widget().pack(fill="both", expand=True)

        # playback controls: two columns side by side
        ctrl = ctk.CTkFrame(right, fg_color=BG, corner_radius=8)
        ctrl.pack(fill="x", padx=8, pady=(4, 8))

        # --- Original column ---
        orig_col = ctk.CTkFrame(ctrl, fg_color="transparent")
        orig_col.pack(side="left", fill="both", expand=True, padx=10, pady=8)
        lbl(orig_col, "ORIGINAL", 11, bold=True, color=ACC).pack()
        ob = ctk.CTkFrame(orig_col, fg_color="transparent")
        ob.pack(pady=(4, 0))
        self.cmp_play_o = ctk.CTkButton(
            ob, text="▶  Play", width=85,
            fg_color="#238636", hover_color="#2ea043",
            command=lambda: self._cmp_play("orig"))
        self.cmp_play_o.pack(side="left", padx=3)
        ctk.CTkButton(ob, text="■  Stop", width=75,
                      fg_color=DIM, hover_color="#484f58",
                      command=self._cmp_stop).pack(side="left", padx=3)
        self.cmp_seek_o = ctk.CTkSlider(
            orig_col, from_=0, to=1, number_of_steps=200)
        self.cmp_seek_o.set(0)
        self.cmp_seek_o.pack(fill="x", padx=6, pady=(6, 0))
        self.cmp_seek_o.configure(
            command=lambda v: self._cmp_seek("orig", v))
        self.cmp_time_o = lbl(orig_col, "0:00 / 0:00", 10, color="gray")
        self.cmp_time_o.pack()

        # divider
        ctk.CTkFrame(ctrl, width=1, fg_color=DIM).pack(
            side="left", fill="y", pady=10)

        # --- Processed column ---
        proc_col = ctk.CTkFrame(ctrl, fg_color="transparent")
        proc_col.pack(side="left", fill="both", expand=True, padx=10, pady=8)
        lbl(proc_col, "PROCESSED", 11, bold=True, color=GOLD).pack()
        pb = ctk.CTkFrame(proc_col, fg_color="transparent")
        pb.pack(pady=(4, 0))
        self.cmp_play_p = ctk.CTkButton(
            pb, text="▶  Play", width=85,
            fg_color="#7d4e00", hover_color="#a36200",
            command=lambda: self._cmp_play("proc"))
        self.cmp_play_p.pack(side="left", padx=3)
        ctk.CTkButton(pb, text="■  Stop", width=75,
                      fg_color=DIM, hover_color="#484f58",
                      command=self._cmp_stop).pack(side="left", padx=3)
        self.cmp_seek_p = ctk.CTkSlider(
            proc_col, from_=0, to=1, number_of_steps=200)
        self.cmp_seek_p.set(0)
        self.cmp_seek_p.pack(fill="x", padx=6, pady=(6, 0))
        self.cmp_seek_p.configure(
            command=lambda v: self._cmp_seek("proc", v))
        self.cmp_time_p = lbl(proc_col, "0:00 / 0:00", 10, color="gray")
        self.cmp_time_p.pack()

        # playback state
        self._cmp_orig     = None   # float32 mono — for waveform display
        self._cmp_orig_raw = None   # int16 (possibly stereo) — perfect native playback
        self._cmp_proc     = None   # float32 decoded — codec artifacts
        self._cmp_fs       = 44100
        self._cmp_playing  = None   # "orig" | "proc" | None
        self._cmp_pos0     = 0      # start sample for current play
        self._cmp_t0       = 0.0
        self._cmp_dur      = 0.0
        self._cmp_line0    = None
        self._cmp_line1    = None
        self._cmp_zoom_s   = 0.0   # zoom window start (seconds)
        self._cmp_zoom_e   = 0.0   # zoom window end   (seconds)

    # ---- compare callbacks ----

    def _cmp_q_changed(self, val):
        b = int(float(val))
        self.cmp_q_lbl.configure(text=f"{b} bits  ({2**b} levels)")

    def _cmp_browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if p:
            self.cmp_path.set(p)
            self.cmp_lbl.configure(
                text=os.path.basename(p)[:34], text_color=ACC)
            self.cmp_status.configure(
                text="File loaded -- click Apply Codec", text_color=GOLD)

    def _cmp_apply(self):
        path = self.cmp_path.get()
        if not path:
            self.cmp_status.configure(
                text="No file selected!", text_color=RED)
            return
        self.cmp_btn.configure(state="disabled", text="Processing...")
        self.cmp_status.configure(text="Encoding...", text_color=GOLD)

        def worker():
            try:
                q = int(float(self.cmp_q.get()))
                # Read raw int16 FIRST (before encode truncates nothing)
                fs, raw_all = wavfile.read(path)
                max_samp    = int(fs * audio_codec.MAX_AUDIO_SEC)
                raw_clip    = raw_all[:max_samp]          # keep int16 + stereo
                # Encode with 1.5 kHz hard cutoff → strong audible degradation
                enc, meta, orig_b = audio_codec.encode(path, q, cutoff_hz=1500)
                rec = audio_codec.decode(enc, meta)
                # Float32 mono for display / SNR
                mono = raw_clip[:, 0] if raw_clip.ndim > 1 else raw_clip
                sig  = mono.astype(np.float32) / 32768.0
                ln   = min(len(sig), len(rec))
                sig, rec = sig[:ln], rec[:ln]
                # int16 trimmed to match — used for native playback
                raw_play = raw_clip[:ln]
                snr_v = metrics.snr_db(sig, rec)
                cr_v  = metrics.comp_ratio(orig_b, len(enc))
                self.after(0, lambda: self._cmp_done(
                    sig, rec, raw_play, fs, snr_v, cr_v, orig_b, len(enc)))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self.cmp_status.configure(
                    text="Error: " + err, text_color=RED))
                self.after(0, lambda: self.cmp_btn.configure(
                    state="normal", text="Apply Codec"))

        threading.Thread(target=worker, daemon=True).start()

    def _cmp_done(self, orig, rec, raw_play, fs, snr_v, cr_v, orig_b, comp_b):
        self._cmp_orig     = orig
        self._cmp_orig_raw = raw_play   # int16, native playback
        self._cmp_proc     = rec
        self._cmp_fs       = fs
        self._cmp_dur      = len(orig) / fs

        self.cmp_snr.configure( text=f"SNR        :  {snr_v:.2f} dB")
        self.cmp_cr.configure(  text=f"Comp Ratio :  {cr_v:.2f} x")
        self.cmp_size.configure(text=f"Size  {orig_b//1024} KB -> {comp_b//1024} KB")
        self.cmp_status.configure(text="Ready -- press Play!", text_color=GREEN)
        self.cmp_btn.configure(state="normal", text="Apply Codec")

        ts = self._fmt_ms(self._cmp_dur)
        self.cmp_time_o.configure(text=f"0:00 / {ts}")
        self.cmp_time_p.configure(text=f"0:00 / {ts}")
        self.cmp_seek_o.set(0)
        self.cmp_seek_p.set(0)

        dur = self._cmp_dur
        N   = len(orig)

        # Lines: 6000 pts — sharp enough, fast to render
        s_line = max(1, N // 6000)
        idx_l  = np.arange(0, N, s_line)
        t_l    = idx_l / float(fs)
        o_l    = orig[idx_l]
        r_l    = rec [idx_l]

        # Fill: 3000 pts — fewer vertices = much faster polygon
        s_fill = max(1, N // 3000)
        idx_f  = np.arange(0, N, s_fill)
        t_f    = idx_f / float(fs)
        o_f    = orig[idx_f]
        r_f    = rec [idx_f]

        # Auto Y-limit
        y_pk  = max(float(np.max(np.abs(o_l))), 0.02)
        y_lim = min(y_pk * 1.20, 1.0)

        # ── Plot 1: original waveform ──────────────────────────────────────
        self.cmp_ax0.clear(); style(self.cmp_ax0)
        self.cmp_ax0.plot(t_l, o_l, color=ACC, lw=0.5)
        self.cmp_ax0.set_xlim(0, dur)
        self.cmp_ax0.set_ylim(-y_lim, y_lim)
        self.cmp_ax0.set_ylabel("Amplitude", color="#8b949e", fontsize=7)
        self.cmp_ax0.set_title("Original Audio  (native quality)",
                               color="#e6edf3", fontsize=9)
        self._cmp_line0 = self.cmp_ax0.axvline(
            0, color=GREEN, lw=1.2, alpha=0.9, zorder=5)

        # ── Plot 2: processed behind, original on top, gap filled ──────────
        self.cmp_ax1.clear(); style(self.cmp_ax1)
        self.cmp_ax1.fill_between(t_f, o_f, r_f,
                                  color=RED, alpha=0.50,
                                  label="Difference", zorder=1)
        self.cmp_ax1.plot(t_l, r_l, color=GOLD,    lw=0.6, alpha=0.85,
                          label="Processed (codec)", zorder=2)
        self.cmp_ax1.plot(t_l, o_l, color="#ddeeff", lw=0.7, alpha=0.95,
                          label="Original", zorder=3)
        self.cmp_ax1.set_xlim(0, dur)
        self.cmp_ax1.set_ylim(-y_lim, y_lim)
        self.cmp_ax1.set_ylabel("Amplitude", color="#8b949e", fontsize=7)
        self.cmp_ax1.set_xlabel("Time (s)", color="#8b949e", fontsize=7)
        self.cmp_ax1.set_title(
            f"Original vs Processed  |  SNR={snr_v:.2f} dB  |  CR={cr_v:.2f}×",
            color="#e6edf3", fontsize=9)
        leg = self.cmp_ax1.legend(fontsize=7, facecolor=CARD, edgecolor=DIM,
                                  loc="upper right")
        for txt in leg.get_texts():
            txt.set_color("#e6edf3")
        self._cmp_line1 = self.cmp_ax1.axvline(
            0, color=GOLD, lw=1.2, alpha=0.9, zorder=5)

        self.cmp_canvas.draw_idle()

    def _cmp_play(self, which):
        try:
            import sounddevice as sd
        except ImportError:
            self.cmp_status.configure(
                text="Run: pip install sounddevice", text_color=RED)
            return
        import time
        if self._cmp_orig is None:
            self.cmp_status.configure(
                text="Apply Codec first!", text_color=RED)
            return
        sd.stop()
        seek_val = (self.cmp_seek_o.get() if which == "orig"
                    else self.cmp_seek_p.get())
        if which == "orig":
            # Native int16 playback — ZERO processing artifacts
            data              = self._cmp_orig_raw
            self._cmp_pos0    = int(seek_val * len(data))
            self._cmp_t0      = time.time()
            self._cmp_playing = which
            chunk = np.ascontiguousarray(data[self._cmp_pos0:])
            sd.play(chunk, samplerate=self._cmp_fs,
                    latency='low', blocking=False)
        else:
            # Float32 decoded — codec artifacts only, no cuts
            data              = self._cmp_proc
            self._cmp_pos0    = int(seek_val * len(data))
            self._cmp_t0      = time.time()
            self._cmp_playing = which
            chunk = np.ascontiguousarray(
                np.clip(data[self._cmp_pos0:], -1.0, 1.0), dtype=np.float32)
            sd.play(chunk, samplerate=self._cmp_fs,
                    latency='high', blocking=False)
        label = "original (native)" if which == "orig" else "processed (codec)"
        self.cmp_status.configure(
            text=f"Playing {label}...", text_color=GREEN)
        self._cmp_poll()

    def _cmp_stop(self):
        try:
            import sounddevice as sd
            sd.stop()
        except ImportError:
            pass
        self._cmp_playing = None
        self.cmp_status.configure(text="Stopped", text_color="gray")

    def _cmp_seek(self, which, val):
        import time
        if self._cmp_orig is None:
            return
        cur_s = float(val) * self._cmp_dur
        ts    = self._fmt_ms(self._cmp_dur)
        cs    = self._fmt_ms(cur_s)
        if which == "orig":
            self.cmp_time_o.configure(text=f"{cs} / {ts}")
        else:
            self.cmp_time_p.configure(text=f"{cs} / {ts}")
        # restart playback from new position if this track is playing
        if self._cmp_playing == which:
            try:
                import sounddevice as sd
                sd.stop()
                if which == "orig":
                    data           = self._cmp_orig_raw
                    self._cmp_pos0 = int(float(val) * len(data))
                    self._cmp_t0   = time.time()
                    chunk = np.ascontiguousarray(data[self._cmp_pos0:])
                    sd.play(chunk, samplerate=self._cmp_fs,
                            latency='low', blocking=False)
                else:
                    data           = self._cmp_proc
                    self._cmp_pos0 = int(float(val) * len(data))
                    self._cmp_t0   = time.time()
                    chunk = np.ascontiguousarray(
                        np.clip(data[self._cmp_pos0:], -1.0, 1.0),
                        dtype=np.float32)
                    sd.play(chunk, samplerate=self._cmp_fs,
                            latency='high', blocking=False)
            except ImportError:
                pass

    def _cmp_poll(self):
        import time
        if self._cmp_playing is None or self._cmp_orig is None:
            return
        elapsed = time.time() - self._cmp_t0
        cur_s   = min(self._cmp_pos0 / self._cmp_fs + elapsed,
                      self._cmp_dur)
        seek_v  = cur_s / self._cmp_dur if self._cmp_dur > 0 else 0.0
        ts = self._fmt_ms(self._cmp_dur)
        cs = self._fmt_ms(cur_s)
        if self._cmp_playing == "orig":
            self.cmp_seek_o.set(seek_v)
            self.cmp_time_o.configure(text=f"{cs} / {ts}")
        else:
            self.cmp_seek_p.set(seek_v)
            self.cmp_time_p.configure(text=f"{cs} / {ts}")
        # Each line tracks its own audio: line0→orig (plot1), line1→proc (plot2)
        redraw = False
        if self._cmp_playing == "orig" and self._cmp_line0 is not None:
            self._cmp_line0.set_xdata([cur_s, cur_s])
            redraw = True
        elif self._cmp_playing == "proc" and self._cmp_line1 is not None:
            self._cmp_line1.set_xdata([cur_s, cur_s])
            redraw = True
        if redraw:
            self.cmp_canvas.draw_idle()
        # stop polling when finished
        if cur_s >= self._cmp_dur:
            self._cmp_playing = None
            self.cmp_status.configure(
                text="Playback finished", text_color="gray")
            return
        self.after(150, self._cmp_poll)


if __name__ == "__main__":
    app = App()
    app.mainloop()
