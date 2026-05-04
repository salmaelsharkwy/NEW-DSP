# ui/helpers.py
import customtkinter as ctk


def label(parent, text, size=12, bold=False, color="#1a1a1a"):
    weight = "bold" if bold else "normal"
    return ctk.CTkLabel(parent, text=text,
                        font=ctk.CTkFont(size=size, weight=weight),
                        text_color=color)


def divider(parent):
    ctk.CTkFrame(parent, height=1, fg_color="#dddddd").pack(fill="x", pady=5)
