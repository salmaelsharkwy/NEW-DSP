# gui/video_tab.py  –  Video Compression tab
import os, time, threading
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
import customtkinter as ctk
import video_codec, metrics
from gui.helpers import label, divider


class VideoTab:
    def __init__(self, tab, app):
        self.app = app; self.vid_path = ""
        self.orig_frames = []; self.recon_frames = []
        self._cur = [0]; self._total = [0]
        self._done_flag = [False]; self._t0 = [0.0]

        left = ctk.CTkFrame(tab, width=240)
        left.pack(side="left", fill="y", padx=(4,6), pady=4)
        left.pack_propagate(False)
        right = ctk.CTkFrame(tab)
        right.pack(side="right", fill="both", expand=True, pady=4, padx=(0,4))

        label(left, "Video Codec", 16, bold=True).pack(pady=(12,4))
        divider(left)
        label(left, "Input File").pack(anchor="w", padx=12)
        ctk.CTkButton(left, text="Browse Video", command=self._browse).pack(fill="x", padx=12, pady=3)
        self.file_lbl = label(left, "No file selected", 10, color="#555")
        self.file_lbl.pack(anchor="w", padx=12)

        divider(left)
        label(left, "Quality  (30=low  90=high)").pack(anchor="w", padx=12)
        self.q_sl = ctk.CTkSlider(left, from_=30, to=90, number_of_steps=12,
                                   command=lambda v: self.q_lbl.configure(text=f"Quality: {int(float(v))}"))
        self.q_sl.set(75); self.q_sl.pack(fill="x", padx=12)
        self.q_lbl = label(left, "Quality: 75", 10, color="#2563eb"); self.q_lbl.pack(anchor="w", padx=12)

        label(left, "GOP  (I-frame interval)").pack(anchor="w", padx=12, pady=(6,0))
        self.gop_sl = ctk.CTkSlider(left, from_=5, to=20, number_of_steps=3,
                                     command=lambda v: self.gop_lbl.configure(text=f"GOP: {int(float(v))}"))
        self.gop_sl.set(10); self.gop_sl.pack(fill="x", padx=12)
        self.gop_lbl = label(left, "GOP: 10", 10, color="#2563eb"); self.gop_lbl.pack(anchor="w", padx=12)

        divider(left)
        self.run_btn = ctk.CTkButton(left, text="Run Compression", command=self._run)
        self.run_btn.pack(fill="x", padx=12, pady=6)

        label(left, "Frame Navigator").pack(anchor="w", padx=12)
        self.frame_sl = ctk.CTkSlider(left, from_=0, to=1, number_of_steps=1,
                                       command=self._show_frame)
        self.frame_sl.pack(fill="x", padx=12)
        self.frame_lbl = label(left, "Frame: --", 10, color="#555"); self.frame_lbl.pack(anchor="w", padx=12)

        divider(left)
        box = ctk.CTkFrame(left, corner_radius=6)
        box.pack(fill="x", padx=12, pady=4)
        label(box, "Results", bold=True).pack(pady=(6,2))
        self.psnr_lbl = label(box, "PSNR       :  --", color="#2563eb"); self.psnr_lbl.pack(anchor="w", padx=10)
        self.cr_lbl   = label(box, "Comp Ratio :  --", color="#2563eb"); self.cr_lbl.pack(anchor="w", padx=10)
        self.sz_lbl   = label(box, "Size       :  --", 10, color="#555"); self.sz_lbl.pack(anchor="w", padx=10, pady=(0,6))
        self.status = label(left, "Ready", 10, color="#555"); self.status.pack(pady=4)

        self.fig, self.v_axes = plt.subplots(1, 3, figsize=(13, 4.8))
        self.fig.patch.set_facecolor("white")
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04, wspace=0.06)
        for ax, t in zip(self.v_axes, ["Original Frame","Reconstructed Frame","Difference x10"]):
            ax.set_facecolor("#f0f0f0"); ax.axis("off"); ax.set_title(t, fontsize=9)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv"),("All files","*.*")])
        if p:
            self.vid_path = p
            self.file_lbl.configure(text=os.path.basename(p)[:30])

    def _run(self):
        if not self.vid_path:
            self.status.configure(text="Select a video file first"); return
        self.run_btn.configure(state="disabled", text="Processing...")
        self.status.configure(text="Starting...")
        self._cur[0] = 0; self._total[0] = 0
        self._done_flag[0] = False; self._t0[0] = time.time()

        def worker():
            try:
                q   = int(float(self.q_sl.get()))
                gop = int(float(self.gop_sl.get()))
                bs, orig, orig_b = video_codec.encode_video(
                    self.vid_path, gop=gop, quality=q,
                    progress_cb=lambda c, t: (self._cur.__setitem__(0, c),
                                              self._total.__setitem__(0, t)))
                recon, _ = video_codec.decode_video(bs)
                n    = min(len(orig), len(recon))
                psnr = metrics.psnr_db(orig[:n], recon[:n])
                cr   = metrics.comp_ratio(orig_b, len(bs))
                self._done_flag[0] = True
                self.app.after(0, lambda: self._finish(orig[:n], recon[:n], psnr, cr, orig_b, len(bs)))
            except Exception as e:
                self._done_flag[0] = True
                err = str(e)
                self.app.after(0, lambda: self._on_error(err))

        self.app.after(400, self._poll)
        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        if self._done_flag[0]: return
        cur, total = self._cur[0], self._total[0]
        elapsed = time.time() - self._t0[0]
        if total > 0:
            pct = int(100 * cur / total)
            eta = (elapsed / cur * (total - cur)) if cur > 0 else 0
            self.status.configure(text=f"Frame {cur}/{total}  {pct}%  ETA: {eta:.0f}s")
        self.app.after(400, self._poll)

    def _on_error(self, msg):
        self.status.configure(text=f"Error: {msg}")
        self.run_btn.configure(state="normal", text="Run Compression")

    def _finish(self, orig, recon, psnr, cr, orig_b, comp_b):
        self.orig_frames = orig; self.recon_frames = recon
        self.psnr_lbl.configure(text=f"PSNR       :  {psnr:.2f} dB")
        self.cr_lbl.configure(  text=f"Comp Ratio :  {cr:.2f}x")
        self.sz_lbl.configure(  text=f"Size  {orig_b//1024} KB → {comp_b//1024} KB")
        self.status.configure(  text="Done")
        self.run_btn.configure(state="normal", text="Run Compression")
        n = len(recon)
        self.frame_sl.configure(to=max(n-1,1), number_of_steps=max(n-1,1))
        self._show_frame(0)

    def _show_frame(self, val):
        idx = int(float(val))
        if not self.orig_frames or idx >= len(self.orig_frames): return
        self.frame_lbl.configure(text=f"Frame: {idx} / {len(self.orig_frames)-1}")
        orig  = self.orig_frames[idx]
        recon = self.recon_frames[idx]
        diff  = np.clip(np.abs(orig.astype(np.int16) - recon.astype(np.int16)) * 10,
                        0, 255).astype(np.uint8)
        for ax in self.v_axes: ax.clear(); ax.axis("off")
        self.v_axes[0].imshow(orig);  self.v_axes[0].set_title(f"Original (frame {idx})", fontsize=9)
        self.v_axes[1].imshow(recon); self.v_axes[1].set_title(f"Reconstructed (frame {idx})", fontsize=9)
        self.v_axes[2].imshow(diff);  self.v_axes[2].set_title("Difference x10", fontsize=9)
        self.canvas.draw()
