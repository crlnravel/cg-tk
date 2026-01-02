import os
import torch

GRID_RES = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"

CANVAS_SIZE = (1024, 1024)
SIDEBAR_WIDTH = 320

THEME = {
    "bg": (30, 30, 30),
    "sidebar": (25, 25, 25),
    "grid_light": (40, 40, 40),
    "grid_dark": (35, 35, 35),
    "text": (220, 220, 220),
    "accent": (66, 135, 245),
    "button": (50, 50, 55),
    "button_hover": (70, 70, 75),
    "modal_bg": (40, 40, 45, 250),
    "overlay_bg": (0, 0, 0, 200),
    "input_bg": (45, 45, 50),
    "input_active": (60, 60, 65),
}

os.makedirs("projects", exist_ok=True)
os.makedirs("templates", exist_ok=True)

if not os.path.exists("templates/sample_1"):
    os.makedirs("templates/sample_1")
    with open("templates/sample_1/prompt.txt", "w") as f:
        f.write("A futuristic sci-fi panel")

