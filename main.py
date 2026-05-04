# main.py
import os
import customtkinter as ctk
import audio_codec
from gui.audio_tab   import AudioTab
from gui.video_tab   import VideoTab
from gui.compare_tab import CompareTab

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Generate the test WAV once at startup (exam: clean signal + noise + silence)
_WAV = os.path.join(os.path.expanduser("~"), "test_audio.wav")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DSP Codec  —  Suez Canal University")
        self.geometry("1340x860")
        self.resizable(True, True)

        audio_codec.gen_wav(_WAV)   # generate silently; AudioTab reads it on demand

        tabs = ctk.CTkTabview(self)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)
        tabs.add("Audio Compression")
        tabs.add("Video Compression")
        tabs.add("Audio Compare")

        AudioTab(tabs.tab("Audio Compression"), self, _WAV)
        VideoTab(tabs.tab("Video Compression"), self)
        CompareTab(tabs.tab("Audio Compare"),   self)


if __name__ == "__main__":
    App().mainloop()
