import os
import sys
import math
import ctypes
import numpy as np
import cv2
import pygame
import threading
import glob
import time
from pygame.locals import *
from PIL import Image, ImageOps

# --- AI Imports (Lazy loaded in thread) ---
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from transformers import pipeline as depth_pipeline
from OpenGL.GL import *
from OpenGL.GL import shaders
import OpenGL.GL.shaders

# --- CONFIGURATION ---
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
    "input_active": (60, 60, 65)
}

# Ensure Directories
os.makedirs("projects", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Generate sample template
if not os.path.exists("templates/sample_1"):
    os.makedirs("templates/sample_1")
    with open("templates/sample_1/prompt.txt", "w") as f:
        f.write("A futuristic sci-fi panel")

# ==========================================
# 1. ADVANCED AI PIPELINE (THREADED)
# ==========================================
class AIPipeline:
    def __init__(self):
        print(f"--- Initializing AI on {DEVICE} ---")
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        if DEVICE == "cuda":
            self.pipe.enable_model_cpu_offload()
            
        self.depth_pipe = depth_pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device=DEVICE,
        )

    def generate(self, prompt, init_image_path, project_name, progress_callback=None):
        # Create Project Folder
        project_path = os.path.join("projects", project_name)
        os.makedirs(project_path, exist_ok=True)

        # 1. Prepare Sketch
        if progress_callback: progress_callback(0.1, "Processing Sketch...")
        sketch = Image.open(init_image_path).convert("RGB").resize((512, 512))
        sketch_gray = sketch.convert("L")
        sketch_array = np.array(sketch_gray)
        _, binary = cv2.threshold(sketch_array, 127, 255, cv2.THRESH_BINARY)
        sketch = Image.fromarray(binary).convert("RGB")
        sketch = ImageOps.invert(sketch)
        sketch.save(os.path.join(project_path, "sketch_processed.png"))

        # 2. Generate Albedo
        if progress_callback: progress_callback(0.2, "Generating Albedo...")
        
        pos_prompt = f"{prompt}, seamless texture, top down view, flat lighting, 8k, highly detailed, photorealistic material"
        neg_prompt = "perspective, 3d, shadows, people, objects, text, watermark, blurry, low quality, distortion"
        
        # Internal Diffusers Callback
        def pipe_callback(step, timestep, latents):
            # Map steps (approx 30) to progress 0.2 -> 0.7
            if progress_callback:
                prog = 0.2 + (step / 30) * 0.5
                progress_callback(prog, f"Diffusion Step {step}/30")

        albedo = self.pipe(
            pos_prompt,
            image=sketch,
            negative_prompt=neg_prompt,
            num_inference_steps=30,
            controlnet_conditioning_scale=0.7,
            guidance_scale=8.0,
            callback=pipe_callback,
            callback_steps=1
        ).images[0]
        
        p_albedo = os.path.join(project_path, "albedo.png")
        albedo.save(p_albedo)

        # 3. Generate Roughness
        if progress_callback: progress_callback(0.8, "Generating Roughness...")
        img_cv = cv2.cvtColor(np.array(albedo), cv2.COLOR_RGB2GRAY)
        roughness = cv2.normalize(img_cv, None, 0, 255, cv2.NORM_MINMAX)
        roughness = 255 - roughness
        p_rough = os.path.join(project_path, "roughness.png")
        Image.fromarray(roughness).save(p_rough)

        # 4. Generate Depth
        if progress_callback: progress_callback(0.9, "Calculating Depth...")
        depth_data = self.depth_pipe(albedo)["depth"].resize((512, 512))
        depth_np = np.array(depth_data)
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            depth_np = ((depth_np - d_min) / (d_max - d_min)) * 255.0
        p_depth = os.path.join(project_path, "depth.png")
        Image.fromarray(depth_np.astype(np.uint8)).save(p_depth)

        # 5. Generate Normal
        if progress_callback: progress_callback(0.95, "Calculating Normals...")
        d_np = depth_np.astype(np.float32)
        sobelx = cv2.Sobel(d_np, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(d_np, cv2.CV_64F, 0, 1, ksize=5)
        normal = np.dstack((-sobelx, -sobely, np.ones_like(d_np) * 500.0))
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = ((normal / norm) * 0.5 + 0.5) * 255
        p_normal = os.path.join(project_path, "normal.png")
        Image.fromarray(normal.astype(np.uint8)).save(p_normal)
        
        if progress_callback: progress_callback(1.0, "Done!")

        return (p_albedo, p_depth, p_normal, p_rough)


# ==========================================
# 2. UI COMPONENTS
# ==========================================
class Button:
    def __init__(self, rect, text, callback, color=THEME["button"], text_size=16):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.color = color
        self.hover = False
        self.text_size = text_size

    def draw(self, surface, font_dict, scroll_y=0):
        # Adjust position based on scroll
        draw_rect = self.rect.move(0, -scroll_y)
        
        # Simple clipping check
        screen_h = surface.get_height()
        if draw_rect.bottom < 0 or draw_rect.top > screen_h:
            return

        col = THEME["button_hover"] if self.hover else self.color
        if self.color == THEME["accent"]:
            col = (80, 150, 255) if self.hover else THEME["accent"]
        pygame.draw.rect(surface, col, draw_rect, border_radius=6)
        
        font = font_dict.get(self.text_size, font_dict[16])
        txt_surf = font.render(self.text, True, THEME["text"])
        txt_rect = txt_surf.get_rect(center=draw_rect.center)
        surface.blit(txt_surf, txt_rect)

    def check_click(self, pos, scroll_y=0):
        # Adjust click pos to match logic coordinate
        adj_rect = self.rect.move(0, -scroll_y)
        if adj_rect.collidepoint(pos):
            return self.callback()
        return None

    def check_hover(self, pos, scroll_y=0):
        adj_rect = self.rect.move(0, -scroll_y)
        self.hover = adj_rect.collidepoint(pos)

class Slider:
    def __init__(self, x, y, w, min_val, max_val, initial):
        self.rect = pygame.Rect(x, y, w, 20)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial
        self.dragging = False

    def draw(self, screen, font, scroll_y=0):
        draw_rect = self.rect.move(0, -scroll_y)
        
        # Draw Label
        lbl = font.render(f"Brush Size: {int(self.val)}px", True, (150, 150, 150))
        screen.blit(lbl, (draw_rect.x, draw_rect.y - 22))

        # Draw Track
        track_rect = pygame.Rect(draw_rect.x, draw_rect.y + 8, draw_rect.width, 4)
        pygame.draw.rect(screen, (60, 60, 60), track_rect, border_radius=2)

        # Draw Handle
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = draw_rect.x + ratio * draw_rect.width
        handle_rect = pygame.Rect(handle_x - 8, draw_rect.y, 16, 20)
        col = THEME["accent"] if self.dragging else (120, 120, 120)
        pygame.draw.rect(screen, col, handle_rect, border_radius=4)

    def handle_event(self, event, scroll_y=0):
        # Adjust rect for hit detection
        adj_rect = self.rect.move(0, -scroll_y)
        
        if event.type == MOUSEBUTTONDOWN:
            if adj_rect.inflate(0, 10).collidepoint(event.pos):
                self.dragging = True
                self.update_val(event.pos[0], adj_rect)
                return True
        elif event.type == MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == MOUSEMOTION and self.dragging:
            self.update_val(event.pos[0], adj_rect)
            return True
        return False

    def update_val(self, mouse_x, current_rect):
        rel = (mouse_x - current_rect.x) / current_rect.width
        rel = max(0.0, min(1.0, rel))
        self.val = self.min_val + rel * (self.max_val - self.min_val)


class PaintInterface:
    def __init__(self, screen_size):
        self.screen_w, self.screen_h = screen_size
        self.canvas_surf = pygame.Surface(CANVAS_SIZE)
        self.canvas_surf.fill((0, 0, 0))

        # Layout
        self.main_area_rect = pygame.Rect(
            SIDEBAR_WIDTH, 0, self.screen_w - SIDEBAR_WIDTH, self.screen_h
        )

        # Scale canvas calculation
        avail_w = self.main_area_rect.width - 40
        avail_h = self.main_area_rect.height - 40
        scale_w = avail_w / CANVAS_SIZE[0]
        scale_h = avail_h / CANVAS_SIZE[1]
        self.scale = min(scale_w, scale_h, 1.0)

        self.disp_w = int(CANVAS_SIZE[0] * self.scale)
        self.disp_h = int(CANVAS_SIZE[1] * self.scale)
        self.canvas_rect = pygame.Rect(0, 0, self.disp_w, self.disp_h)
        self.canvas_rect.center = self.main_area_rect.center

        # State
        self.brush_size = 20
        self.brush_color = (255, 255, 255)
        self.drawing = False
        self.last_pos = None
        self.scroll_y = 0
        self.max_scroll = 0
        
        # Inputs
        self.project_name = f"project_{int(time.time())}"
        self.prompt = "sci-fi metal texture"
        self.active_field = None
        self.show_hint = False
        self.show_load_modal = False

        # Fonts
        self.fonts = {
            14: pygame.font.SysFont("Segoe UI", 14),
            16: pygame.font.SysFont("Segoe UI", 16),
            18: pygame.font.SysFont("Segoe UI", 18, bold=True),
            24: pygame.font.SysFont("Segoe UI", 24, bold=True)
        }

        # UI Elements
        # Calculate Y positions relative to top of scrollable content area
        # Starting Y = 50 (below header)
        
        self.size_slider = Slider(20, 480, SIDEBAR_WIDTH - 40, 2, 100, 20)
        
        # Hint Button (Top Right Fixed)
        self.hint_btn = Button((SIDEBAR_WIDTH - 40, 10, 30, 30), "?", lambda: self.toggle_hint(), (60, 60, 65), 18)
        
        # Scan templates
        self.templates = sorted([d for d in os.listdir("templates") if os.path.isdir(os.path.join("templates", d))])
        
        self.tpl_buttons = []
        for i, tpl in enumerate(self.templates):
            row = i // 3
            col = i % 3
            size = (SIDEBAR_WIDTH - 50) // 3
            x = 20 + col * (size + 5)
            y = 350 + row * (40)
            btn = Button((x, y, size, 30), str(i+1), lambda t=tpl: self.load_template(t), text_size=14)
            self.tpl_buttons.append(btn)

        # Main Buttons
        btn_y = 540
        self.buttons = [
            Button((20, btn_y, SIDEBAR_WIDTH - 40, 40), "GENERATE", lambda: "GENERATE", THEME["accent"], 18),
            Button((20, btn_y + 50, SIDEBAR_WIDTH - 40, 40), "Clear Canvas", lambda: self.clear_canvas()),
            Button((20, btn_y + 100, SIDEBAR_WIDTH - 40, 40), "Load Project", lambda: self.toggle_load_modal()),
            # Exit button is relative to screen bottom usually, but let's make it part of scroll
            Button((20, btn_y + 160, SIDEBAR_WIDTH - 40, 40), "Exit", lambda: "EXIT", (180, 50, 50))
        ]
        
        # Calculate Max Scroll
        total_h = btn_y + 220 # Approximate bottom
        self.max_scroll = max(0, total_h - self.screen_h)

        self.project_list_buttons = []

    def clear_canvas(self):
        self.canvas_surf.fill((0, 0, 0))

    def toggle_hint(self):
        self.show_hint = not self.show_hint
        self.show_load_modal = False
        
    def toggle_load_modal(self):
        self.show_load_modal = not self.show_load_modal
        self.show_hint = False
        if self.show_load_modal:
            self.refresh_project_list()

    def refresh_project_list(self):
        projects = sorted([d for d in os.listdir("projects") if os.path.isdir(os.path.join("projects", d))])
        self.project_list_buttons = []
        cols = 3
        w = 180
        h = 40
        gap = 10
        start_x = (self.screen_w - (cols * w + (cols-1)*gap)) // 2
        start_y = 150
        for i, proj in enumerate(projects):
            r = i // cols
            c = i % cols
            rect = (start_x + c*(w+gap), start_y + r*(h+gap), w, h)
            cb = lambda p=proj: ("LOAD_PROJECT", p)
            self.project_list_buttons.append(Button(rect, proj[:20], cb, text_size=14))

    def load_template(self, folder_name):
        path = os.path.join("templates", folder_name)
        img_path = os.path.join(path, "sketch.png")
        if os.path.exists(img_path):
            try:
                img = pygame.image.load(img_path)
                img = pygame.transform.scale(img, CANVAS_SIZE)
                self.canvas_surf.blit(img, (0,0))
            except: pass
        txt_path = os.path.join(path, "prompt.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                self.prompt = f.read(60)

    def get_canvas_pos(self, screen_pos):
        if not self.canvas_rect.collidepoint(screen_pos):
            return None
        rel_x = screen_pos[0] - self.canvas_rect.left
        rel_y = screen_pos[1] - self.canvas_rect.top
        can_x = int(rel_x / self.scale)
        can_y = int(rel_y / self.scale)
        return (can_x, can_y)

    def handle_event(self, event):
        if self.show_hint:
            if event.type == MOUSEBUTTONDOWN: self.show_hint = False
            return None
            
        if self.show_load_modal:
            if event.type == MOUSEBUTTONDOWN:
                for btn in self.project_list_buttons:
                    res = btn.check_click(event.pos)
                    if res: return res
                self.show_load_modal = False
            if event.type == MOUSEMOTION:
                for btn in self.project_list_buttons: btn.check_hover(event.pos)
            return None

        # Sidebar Scroll
        if event.type == MOUSEWHEEL:
            if pygame.mouse.get_pos()[0] < SIDEBAR_WIDTH:
                self.scroll_y -= event.y * 20
                self.scroll_y = max(0, min(self.scroll_y, self.max_scroll))

        # Hint Button (Fixed)
        if event.type == MOUSEBUTTONDOWN:
            if self.hint_btn.check_click(event.pos): return None
        if event.type == MOUSEMOTION:
            self.hint_btn.check_hover(event.pos)

        # Scrollable Area Inputs
        # Adjust logic inputs by scroll
        if self.size_slider.handle_event(event, self.scroll_y):
            self.brush_size = int(self.size_slider.val)
            return None

        if event.type == MOUSEMOTION:
            for btn in self.buttons + self.tpl_buttons:
                btn.check_hover(event.pos, self.scroll_y)

            if self.drawing:
                c_pos = self.get_canvas_pos(event.pos)
                if c_pos and self.last_pos:
                    pygame.draw.line(self.canvas_surf, self.brush_color, self.last_pos, c_pos, self.brush_size)
                    pygame.draw.circle(self.canvas_surf, self.brush_color, self.last_pos, self.brush_size // 2)
                    pygame.draw.circle(self.canvas_surf, self.brush_color, c_pos, self.brush_size // 2)
                    self.last_pos = c_pos
                elif c_pos: self.last_pos = c_pos
                else: self.last_pos = None

        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                # Scrollable Buttons
                for btn in self.buttons + self.tpl_buttons:
                    res = btn.check_click(event.pos, self.scroll_y)
                    if res: return res

                # Adjust rects for scroll
                proj_rect = pygame.Rect(20, 80 - self.scroll_y, SIDEBAR_WIDTH - 40, 30)
                prompt_rect = pygame.Rect(20, 160 - self.scroll_y, SIDEBAR_WIDTH - 40, 100)

                if proj_rect.collidepoint(event.pos): self.active_field = 'project'
                elif prompt_rect.collidepoint(event.pos): self.active_field = 'prompt'
                else:
                    if not self.canvas_rect.collidepoint(event.pos): self.active_field = None

                # Canvas
                c_pos = self.get_canvas_pos(event.pos)
                if c_pos:
                    self.drawing = True
                    self.last_pos = c_pos
                    pygame.draw.circle(self.canvas_surf, self.brush_color, c_pos, self.brush_size // 2)

        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                self.drawing = False
                self.last_pos = None

        elif event.type == KEYDOWN:
            if self.active_field == 'project':
                if event.key == K_RETURN: self.active_field = None
                elif event.key == K_BACKSPACE: self.project_name = self.project_name[:-1]
                else: 
                    if len(self.project_name) < 20 and event.unicode.isprintable():
                        self.project_name += event.unicode
            elif self.active_field == 'prompt':
                if event.key == K_RETURN: self.active_field = None
                elif event.key == K_BACKSPACE: self.prompt = self.prompt[:-1]
                else:
                    if len(self.prompt) < 60 and event.unicode.isprintable():
                        self.prompt += event.unicode
            else:
                if event.key == K_LEFTBRACKET:
                    self.brush_size = max(2, self.brush_size - 2)
                    self.size_slider.val = self.brush_size
                elif event.key == K_RIGHTBRACKET:
                    self.brush_size += 2
                    self.size_slider.val = self.brush_size

        return None

    def draw_modal(self, screen, title, lines):
        overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        overlay.fill(THEME["overlay_bg"])
        screen.blit(overlay, (0, 0))
        mw, mh = 600, 400
        mx, my = (self.screen_w - mw) // 2, (self.screen_h - mh) // 2
        pygame.draw.rect(screen, (10, 10, 10), (mx+4, my+4, mw, mh), border_radius=12)
        pygame.draw.rect(screen, (50, 50, 55), (mx, my, mw, mh), border_radius=12)
        pygame.draw.rect(screen, THEME["accent"], (mx, my, mw, mh), width=2, border_radius=12)
        t_surf = self.fonts[24].render(title, True, (255, 255, 255))
        screen.blit(t_surf, (mx + 30, my + 30))
        y = my + 80
        for line in lines:
            l_surf = self.fonts[16].render(line, True, (220, 220, 220))
            screen.blit(l_surf, (mx + 30, y))
            y += 30
        hint = self.fonts[14].render("Click anywhere to close", True, THEME["accent"])
        hint_rect = hint.get_rect(center=(mx + mw // 2, my + mh - 30))
        screen.blit(hint, hint_rect)

    def draw_load_modal(self, screen):
        overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        overlay.fill(THEME["overlay_bg"])
        screen.blit(overlay, (0, 0))
        mw, mh = 700, 500
        mx, my = (self.screen_w - mw) // 2, (self.screen_h - mh) // 2
        pygame.draw.rect(screen, (50, 50, 55), (mx, my, mw, mh), border_radius=12)
        pygame.draw.rect(screen, THEME["accent"], (mx, my, mw, mh), width=2, border_radius=12)
        t = self.fonts[24].render("Load Project", True, (255, 255, 255))
        screen.blit(t, (mx + 30, my + 30))
        if not self.project_list_buttons:
             msg = self.fonts[16].render("No projects found in 'projects/' folder.", True, (200, 200, 200))
             screen.blit(msg, (mx + 30, my + 100))
        else:
            for btn in self.project_list_buttons: btn.draw(screen, self.fonts)

    def draw_sidebar_content(self, screen):
        # Create a clip rect for the sidebar content (everything below header)
        content_rect = pygame.Rect(0, 50, SIDEBAR_WIDTH, self.screen_h - 50)
        screen.set_clip(content_rect)
        
        # Offset all drawing by -self.scroll_y
        off = -self.scroll_y

        # Project Name Input
        lbl = self.fonts[14].render("Project Name:", True, (150, 150, 150))
        screen.blit(lbl, (20, 55 + off))
        proj_rect = pygame.Rect(20, 80 + off, SIDEBAR_WIDTH - 40, 30)
        bg = THEME["input_active"] if self.active_field == 'project' else THEME["input_bg"]
        pygame.draw.rect(screen, bg, proj_rect, border_radius=4)
        txt = self.fonts[16].render(self.project_name + ("|" if self.active_field == 'project' and time.time() % 1 > 0.5 else ""), True, (255, 255, 255))
        screen.blit(txt, (25, 85 + off))

        # Prompt Input
        lbl = self.fonts[14].render(f"Prompt ({len(self.prompt)}/60):", True, (150, 150, 150))
        screen.blit(lbl, (20, 135 + off))
        prompt_rect = pygame.Rect(20, 160 + off, SIDEBAR_WIDTH - 40, 100)
        bg = THEME["input_active"] if self.active_field == 'prompt' else THEME["input_bg"]
        pygame.draw.rect(screen, bg, prompt_rect, border_radius=4)
        words = self.prompt.split(' ')
        lines = []
        curr_line = ""
        for word in words:
            test_line = curr_line + word + " "
            if self.fonts[16].size(test_line)[0] < prompt_rect.width - 10: curr_line = test_line
            else:
                lines.append(curr_line)
                curr_line = word + " "
        lines.append(curr_line + ("|" if self.active_field == 'prompt' and time.time() % 1 > 0.5 else ""))
        for i, line in enumerate(lines):
            t = self.fonts[16].render(line, True, (255, 255, 255))
            screen.blit(t, (25, 165 + i * 20 + off))

        # Templates
        lbl = self.fonts[14].render("Templates:", True, (150, 150, 150))
        screen.blit(lbl, (20, 320 + off))
        if not self.tpl_buttons:
             lbl_none = self.fonts[14].render("No templates in 'templates/'", True, (100, 100, 100))
             screen.blit(lbl_none, (20, 350 + off))
        for btn in self.tpl_buttons: btn.draw(screen, self.fonts, self.scroll_y)

        # Controls
        self.size_slider.draw(screen, self.fonts[14], self.scroll_y)
        for btn in self.buttons: btn.draw(screen, self.fonts, self.scroll_y)

        screen.set_clip(None)

    def draw(self, screen):
        # 1. Main Grid
        clip = self.main_area_rect
        pygame.draw.rect(screen, THEME["bg"], clip)
        grid_sz = 40
        for y in range(0, clip.height, grid_sz):
            for x in range(0, clip.width, grid_sz):
                color = THEME["grid_light"] if (x // grid_sz + y // grid_sz) % 2 == 0 else THEME["grid_dark"]
                pygame.draw.rect(screen, color, (clip.x + x, clip.y + y, grid_sz, grid_sz))

        # 2. Canvas
        shadow = self.canvas_rect.inflate(4, 4)
        pygame.draw.rect(screen, (10, 10, 10), shadow)
        scaled_surf = pygame.transform.scale(self.canvas_surf, (self.disp_w, self.disp_h))
        screen.blit(scaled_surf, self.canvas_rect)

        # 3. Sidebar Background
        pygame.draw.rect(screen, THEME["sidebar"], (0, 0, SIDEBAR_WIDTH, self.screen_h))
        pygame.draw.line(screen, (50, 50, 50), (SIDEBAR_WIDTH, 0), (SIDEBAR_WIDTH, self.screen_h))

        # Sidebar Header (Fixed)
        title = self.fonts[18].render("AI Material Studio", True, (255, 255, 255))
        screen.blit(title, (20, 15))
        
        # Hint Button (Fixed)
        self.hint_btn.draw(screen, self.fonts)

        # Scrollable Content
        self.draw_sidebar_content(screen)
        
        # Scrollbar Indicator (Simple)
        if self.max_scroll > 0:
            scroll_h = self.screen_h - 50
            bar_h = max(20, (scroll_h / (scroll_h + self.max_scroll)) * scroll_h)
            bar_y = 50 + (self.scroll_y / self.max_scroll) * (scroll_h - bar_h)
            pygame.draw.rect(screen, (60, 60, 60), (SIDEBAR_WIDTH - 6, bar_y, 4, bar_h), border_radius=2)

        # Overlays
        if self.show_hint:
            self.draw_modal(screen, "How to Paint", [
                "1. Draw white shapes on black (Depth/Structure).",
                "2. Name your project and type a prompt.",
                "3. Use [ ] to change brush size.",
                "4. Click GENERATE to create PBR textures.",
                "5. Templates let you start quickly.",
                "6. Character limit for prompt is 60.",
            ])
            
        if self.show_load_modal:
            self.draw_load_modal(screen)

    def save_sketch(self):
        pygame.image.save(self.canvas_surf, "sketch_input.png")
        return "sketch_input.png"


# ==========================================
# 3. 3D RENDERER (UNCHANGED)
# ==========================================
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoords;
layout (location = 2) in vec3 normal;
out vec2 FragTexCoords;
out vec3 FragPos;
out vec3 FragNormal;
uniform mat4 model, view, projection;
uniform sampler2D depthMap;
uniform float displacementStrength;
void main() {
    FragTexCoords = texCoords;
    float depth = texture(depthMap, texCoords).r;
    vec3 displacedPos = position + (normal * depth * displacementStrength);
    FragPos = vec3(model * vec4(displacedPos, 1.0));
    mat3 normalMatrix = mat3(transpose(inverse(model)));
    FragNormal = normalMatrix * normal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
in vec2 FragTexCoords;
in vec3 FragPos;
in vec3 FragNormal;
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D roughnessMap;
uniform vec3 lightPos, viewPos;
vec3 getNormalFromMap() {
    vec3 tangentNormal = texture(normalMap, FragTexCoords).xyz * 2.0 - 1.0;
    vec3 Q1  = dFdx(FragPos);
    vec3 Q2  = dFdy(FragPos);
    vec2 st1 = dFdx(FragTexCoords);
    vec2 st2 = dFdy(FragTexCoords);
    vec3 N   = normalize(FragNormal);
    vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
    vec3 B  = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);
    return normalize(TBN * tangentNormal);
}
void main() {
    vec3 albedo = texture(albedoMap, FragTexCoords).rgb;
    float rough = texture(roughnessMap, FragTexCoords).r;
    vec3 norm = getNormalFromMap();
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float diff = max(dot(norm, lightDir), 0.0);
    float shininess = mix(2.0, 64.0, 1.0 - rough); 
    float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
    float rim = 1.0 - max(dot(viewDir, norm), 0.0);
    rim = pow(rim, 3.0) * 0.3;
    vec3 ambient = vec3(0.05) * albedo;
    vec3 diffuse = diff * albedo;
    vec3 specular = vec3(0.3) * spec * (1.0 - rough);
    FragColor = vec4(ambient + diffuse + specular + (vec3(1.0)*rim), 1.0);
}
"""

class Renderer3D:
    def __init__(self):
        self.create_plane()
        self.shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        self.texture_ids = [0, 0, 0, 0]
        self.cam = [0.0, 0.0, 1.3] 

    def create_plane(self):
        verts, inds = [], []
        scale = 0.5 
        for y in range(GRID_RES + 1):
            for x in range(GRID_RES + 1):
                u, v = x / GRID_RES, y / GRID_RES
                px, py, pz = (u - 0.5) * scale * 2, (v - 0.5) * scale * 2, 0.0
                verts.extend([px, py, pz, u, 1.0 - v, 0.0, 0.0, 1.0])
        for y in range(GRID_RES):
            for x in range(GRID_RES):
                tl = y * (GRID_RES + 1) + x
                tr = tl + 1
                bl = (y + 1) * (GRID_RES + 1) + x
                br = bl + 1
                inds.extend([tl, bl, tr, tr, bl, br])

        self.count = len(inds)
        v_data = np.array(verts, dtype=np.float32)
        i_data = np.array(inds, dtype=np.uint32)
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, v_data.nbytes, v_data, GL_STATIC_DRAW)
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, i_data.nbytes, i_data, GL_STATIC_DRAW)
        stride = 32
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))

    def load_textures(self, albedo_p, depth_p, normal_p, rough_p):
        paths = [albedo_p, normal_p, rough_p, depth_p]
        for i, path in enumerate(paths):
            img = Image.open(path).convert("RGB")
            data = np.array(img, dtype=np.uint8)
            if self.texture_ids[i] == 0: self.texture_ids[i] = glGenTextures(1)
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, self.texture_ids[i])
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def draw(self, aspect_ratio):
        glUseProgram(self.shader)
        dist = max(0.2, self.cam[2])
        proj = self.perspective(45, aspect_ratio, 0.1, 100.0)
        rx, ry = math.radians(self.cam[0]), math.radians(self.cam[1])
        cx = dist * math.sin(ry) * math.cos(rx)
        cy = dist * math.sin(rx)
        cz = dist * math.cos(ry) * math.cos(rx)
        view = self.lookat(np.array([cx, cy, cz]), np.array([0, 0, 0]), np.array([0, 1, 0]))
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, True, proj)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, True, view)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, True, np.identity(4, dtype=np.float32))
        glUniform1f(glGetUniformLocation(self.shader, "displacementStrength"), 0.3)
        glUniform3f(glGetUniformLocation(self.shader, "lightPos"), 2.0, 4.0, 5.0)
        glUniform3f(glGetUniformLocation(self.shader, "viewPos"), cx, cy, cz)
        tex_locs = ["albedoMap", "normalMap", "roughnessMap", "depthMap"]
        for i, name in enumerate(tex_locs):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, self.texture_ids[i])
            glUniform1i(glGetUniformLocation(self.shader, name), i)
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, self.count, GL_UNSIGNED_INT, None)

    def perspective(self, fov, aspect, near, far):
        f = 1.0 / math.tan(math.radians(fov) / 2)
        return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0], [0, 0, (far + near) / (near - far), -1], [0, 0, (2 * far * near) / (near - far), 0]], dtype=np.float32)

    def lookat(self, eye, target, up):
        f = target - eye
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        m = np.identity(4, dtype=np.float32)
        m[:3, :3] = np.array([s, u, -f])
        m[:3, 3] = -np.dot(m[:3, :3], eye)
        return m

def draw_modal_overlay_3d(surface, w, h, title, lines):
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill(THEME["modal_bg"])
    mw, mh = 500, 350
    mx, my = (w - mw) // 2, (h - mh) // 2
    pygame.draw.rect(overlay, (50, 50, 55), (mx, my, mw, mh), border_radius=12)
    pygame.draw.rect(overlay, THEME["accent"], (mx, my, mw, mh), width=2, border_radius=12)
    font_bold = pygame.font.SysFont("Segoe UI", 24, bold=True)
    font = pygame.font.SysFont("Segoe UI", 18)
    t = font_bold.render(title, True, (255, 255, 255))
    overlay.blit(t, (mx + 30, my + 30))
    y = my + 80
    for line in lines:
        l = font.render(line, True, (220, 220, 220))
        overlay.blit(l, (mx + 30, y))
        y += 30
    hint = font.render("Press H to Close/Open Help", True, THEME["accent"])
    overlay.blit(hint, (mx + 30, my + mh - 40))
    surface.blit(overlay, (0, 0))


# ==========================================
# MAIN & THREADING LOGIC
# ==========================================

# Globals for Loading
ai_pipeline = None
loading_done = False
loading_msg = "Starting..."

# Globals for Generation
generation_thread = None
generation_progress = 0.0
generation_status = ""
generated_files = []
generation_complete = False

def ai_loader_thread():
    global ai_pipeline, loading_done, loading_msg
    try:
        loading_msg = "Loading AI Models..."
        ai_pipeline = AIPipeline()
        loading_done = True
    except Exception as e:
        print(f"Error loading AI: {e}")
        loading_msg = "Error Loading AI!"

def generation_worker(prompt, sketch_path, project_name):
    global generated_files, generation_complete
    
    def cb(progress, status):
        global generation_progress, generation_status
        generation_progress = progress
        generation_status = status
        
    try:
        generated_files = ai_pipeline.generate(prompt, sketch_path, project_name, progress_callback=cb)
        generation_complete = True
    except Exception as e:
        print(f"Generation Error: {e}")
        # In a real app, handle error state
        generation_complete = True

def main():
    global ai_pipeline, loading_done, loading_msg
    global generation_thread, generation_progress, generation_status, generation_complete, generated_files
    
    pygame.init()
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    
    # Start AI Thread
    t = threading.Thread(target=ai_loader_thread, daemon=True)
    t.start()
    
    # 1. Loading Loop
    screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
    font_large = pygame.font.SysFont("Segoe UI", 40)
    font_small = pygame.font.SysFont("Segoe UI", 20)
    clock = pygame.time.Clock()
    spinner_angle = 0
    
    while not loading_done:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        screen.fill(THEME["bg"])
        spinner_rect = pygame.Rect(0, 0, 60, 60)
        spinner_rect.center = (W // 2, H // 2 - 50)
        pygame.draw.arc(screen, THEME["accent"], spinner_rect, spinner_angle, spinner_angle + 1.5, 5)
        spinner_angle += 0.1
        txt = font_large.render(loading_msg, True, (255, 255, 255))
        r = txt.get_rect(center=(W // 2, H // 2 + 20))
        screen.blit(txt, r)
        hint = font_small.render("Initial setup may take a minute depending on GPU...", True, (150, 150, 150))
        hr = hint.get_rect(center=(W // 2, H // 2 + 60))
        screen.blit(hint, hr)
        pygame.display.flip()
        
    # 2. Main Application
    painter = PaintInterface((W, H))
    renderer = None
    
    MODE = "PAINT"
    SHOW_3D_HELP = False
    
    running = True
    
    while running:
        clock.tick(60)
        events = pygame.event.get()
        
        # --- GENERATING MODE ---
        if MODE == "GENERATING":
            # Consume events to prevent freezing, but don't process interaction
            for event in events:
                if event.type == QUIT: running = False
            
            # Check for completion
            if generation_complete:
                MODE = "PRE_VIEW_INSTRUCTIONS"
                generation_thread.join()
            
            # Draw Loading Screen
            # Draw underlying interface dimmed
            painter.draw(screen)
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))
            
            # Progress Bar
            bar_w = 400
            bar_h = 20
            cx, cy = W // 2, H // 2
            
            # Text
            txt = font_large.render(generation_status, True, (255, 255, 255))
            tr = txt.get_rect(center=(cx, cy - 40))
            screen.blit(txt, tr)
            
            # Bar Back
            pygame.draw.rect(screen, (50, 50, 50), (cx - bar_w//2, cy, bar_w, bar_h), border_radius=10)
            # Bar Fill
            fill_w = int(bar_w * generation_progress)
            pygame.draw.rect(screen, THEME["accent"], (cx - bar_w//2, cy, fill_w, bar_h), border_radius=10)
            
            pygame.display.flip()
            continue

        # --- PRE-3D INSTRUCTION LOGIC ---
        if MODE == "PRE_VIEW_INSTRUCTIONS":
            for event in events:
                if event.type == MOUSEBUTTONDOWN or event.type == KEYDOWN:
                    MODE = "VIEW"
                    pygame.display.quit()
                    pygame.display.init()
                    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
                    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
                    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
                    screen = pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL | FULLSCREEN)
                    renderer = Renderer3D()
                    renderer.load_textures(*generated_files)
                    SHOW_3D_HELP = False
            
            painter.draw(screen) 
            painter.draw_modal(screen, "Ready to View", [
                "1. Navigation Controls:",
                "   W / S : Rotate Up / Down",
                "   A / D : Rotate Left / Right",
                "   R / E : Zoom In / Out",
                "",
                "2. Press 'H' to see these controls again.",
                "3. Press 'ESC' to return to drawing.",
                "",
                ">> Click anywhere to Start 3D View <<"
            ])
            pygame.display.flip()
            continue

        # --- EVENT LOOP ---
        for event in events:
            if event.type == QUIT:
                running = False

            if MODE == "PAINT":
                action = painter.handle_event(event)
                
                if isinstance(action, tuple) and action[0] == "LOAD_PROJECT":
                    p_name = action[1]
                    p_path = os.path.join("projects", p_name)
                    f_albedo = os.path.join(p_path, "albedo.png")
                    if os.path.exists(f_albedo):
                        generated_files = (
                            os.path.join(p_path, "albedo.png"),
                            os.path.join(p_path, "depth.png"),
                            os.path.join(p_path, "normal.png"),
                            os.path.join(p_path, "roughness.png")
                        )
                        MODE = "PRE_VIEW_INSTRUCTIONS"

                elif action == "EXIT":
                    running = False
                elif action == "GENERATE":
                    # Start Generation Thread
                    s_path = painter.save_sketch()
                    generation_progress = 0.0
                    generation_status = "Starting..."
                    generation_complete = False
                    
                    generation_thread = threading.Thread(
                        target=generation_worker,
                        args=(painter.prompt, s_path, painter.project_name),
                        daemon=True
                    )
                    generation_thread.start()
                    MODE = "GENERATING"

            elif MODE == "VIEW":
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        MODE = "PAINT"
                        pygame.display.quit()
                        pygame.display.init()
                        screen = pygame.display.set_mode((W, H), FULLSCREEN)
                        painter = PaintInterface((W, H))
                    elif event.key == K_h:
                        SHOW_3D_HELP = not SHOW_3D_HELP

        # --- DRAW LOOP ---
        if MODE == "PAINT":
            painter.draw(screen)
            pygame.display.flip()

        elif MODE == "VIEW" and renderer:
            keys = pygame.key.get_pressed()
            if keys[K_w]: renderer.cam[0] = max(-89, renderer.cam[0] - 2)
            if keys[K_s]: renderer.cam[0] = min(89, renderer.cam[0] + 2)
            if keys[K_a]: renderer.cam[1] -= 2
            if keys[K_d]: renderer.cam[1] += 2
            if keys[K_r]: renderer.cam[2] = max(0.1, renderer.cam[2] - 0.05)
            if keys[K_e]: renderer.cam[2] += 0.05

            glClearColor(0.1, 0.1, 0.1, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
            glViewport(0, 0, W, H)
            renderer.draw(W / H)

            try:
                overlay_surf = pygame.display.get_surface()
                if SHOW_3D_HELP and overlay_surf:
                    draw_modal_overlay_3d(overlay_surf, W, H, "3D Controls", [
                        "W / S : Rotate Up / Down",
                        "A / D : Rotate Left / Right",
                        "R / E : Zoom In / Out",
                        "ESC : Return to Paint"
                    ])
            except:
                pass

            pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
