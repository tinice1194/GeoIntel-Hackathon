import csv
import random
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

PREVIEW_DIR = Path(r"G:/GIS_AI_PROJECT/output/train/preview")
OUTPUT_CSV = Path(r"G:/GIS_AI_PROJECT/output/roof_labels_1000.csv")
SAMPLE_SIZE = 1000
THUMB_SIZE = (700, 700)
CLASSES = ["rcc", "tin", "tiled", "granite_marble", "unknown", "skip"]

class RoofLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Roof Labeling Tool")
        self.root.geometry("1000x900")

        self.png_files = sorted(PREVIEW_DIR.glob("*.png"))
        if not self.png_files:
            messagebox.showerror("Error", f"No PNG files found in:\n{PREVIEW_DIR}")
            root.destroy()
            return

        self.sample_files = random.sample(self.png_files, min(SAMPLE_SIZE, len(self.png_files)))
        self.index = 0
        self.labels = []
        self.current_photo = None

        self.title_label = tk.Label(root, text="Roof Type Labeling Tool", font=("Arial", 20, "bold"))
        self.title_label.pack(pady=10)

        self.info_label = tk.Label(root, text="", font=("Arial", 12))
        self.info_label.pack(pady=5)

        self.image_label = tk.Label(root, bg="black")
        self.image_label.pack(padx=10, pady=10, expand=True)

        self.filename_label = tk.Label(root, text="", font=("Arial", 11))
        self.filename_label.pack(pady=5)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.buttons = {}
        btn_specs = [
            ("RCC (1)", "rcc"),
            ("TIN (2)", "tin"),
            ("TILED (3)", "tiled"),
            ("GRANITE/MARBLE (4)", "granite_marble"),
            ("UNKNOWN (5)", "unknown"),
            ("SKIP (6)", "skip"),
        ]

        for txt, val in btn_specs:
            b = tk.Button(btn_frame, text=txt, width=14, height=2,
                          command=lambda v=val: self.assign_label(v))
            b.pack(side=tk.LEFT, padx=6)
            self.buttons[val] = b

        nav_frame = tk.Frame(root)
        nav_frame.pack(pady=8)

        tk.Button(nav_frame, text="Backspace = Undo", width=18, command=self.undo_last).pack(side=tk.LEFT, padx=6)
        tk.Button(nav_frame, text="Save CSV", width=12, command=self.save_csv).pack(side=tk.LEFT, padx=6)
        tk.Button(nav_frame, text="Quit", width=12, command=self.on_quit).pack(side=tk.LEFT, padx=6)

        help_text = (
            "Keyboard shortcuts: 1=RCC, 2=TIN, 3=TILED, 4=GRANITE/MARBLE, 5=UNKNOWN, 6=SKIP, Backspace=Undo, S=Save, Q=Quit"
        )
        self.help_label = tk.Label(root, text=help_text, font=("Arial", 10), fg="gray30")
        self.help_label.pack(pady=4)

        root.bind("1", lambda e: self.assign_label("rcc"))
        root.bind("2", lambda e: self.assign_label("tin"))
        root.bind("3", lambda e: self.assign_label("tiled"))
        root.bind("4", lambda e: self.assign_label("granite_marble"))
        root.bind("5", lambda e: self.assign_label("unknown"))
        root.bind("6", lambda e: self.assign_label("skip"))
        root.bind("<BackSpace>", lambda e: self.undo_last())
        root.bind("s", lambda e: self.save_csv())
        root.bind("q", lambda e: self.on_quit())

        self.show_current_image()

    def show_current_image(self):
        if self.index >= len(self.sample_files):
            self.info_label.config(text=f"Completed {len(self.sample_files)} / {len(self.sample_files)}")
            self.filename_label.config(text="Done. Please save your CSV.")
            self.image_label.config(image="", text="All selected PNGs labeled", fg="white", font=("Arial", 18), width=60, height=20)
            return

        img_path = self.sample_files[self.index]
        self.info_label.config(text=f"Image {self.index + 1} of {len(self.sample_files)} | Labeled: {len(self.labels)}")
        self.filename_label.config(text=img_path.name)

        img = Image.open(img_path).convert("RGB")
        img.thumbnail(THUMB_SIZE)
        self.current_photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.current_photo)

    def assign_label(self, label):
        if self.index >= len(self.sample_files):
            return
        img_path = self.sample_files[self.index]
        self.labels.append({"png_file": img_path.name, "label": label})
        self.index += 1
        if len(self.labels) % 50 == 0:
            self.save_csv(silent=True)
        self.show_current_image()

    def undo_last(self):
        if not self.labels:
            return
        self.labels.pop()
        self.index = max(0, self.index - 1)
        self.show_current_image()

    def save_csv(self, silent=False):
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["png_file", "label"])
            writer.writeheader()
            writer.writerows(self.labels)
        if not silent:
            messagebox.showinfo("Saved", f"Saved labels to:\n{OUTPUT_CSV}")

    def on_quit(self):
        if self.labels:
            self.save_csv(silent=True)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RoofLabeler(root)
    root.mainloop()
