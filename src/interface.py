import os
import time
import pygame
from pygame.locals import *

from src.config import CANVAS_SIZE, SIDEBAR_WIDTH, THEME
from src.ui_components import Button, Slider


class PaintInterface:
    def __init__(self, screen_size):
        self.screen_w, self.screen_h = screen_size
        self.canvas_surf = pygame.Surface(CANVAS_SIZE)
        self.canvas_surf.fill((0, 0, 0))

        self.main_area_rect = pygame.Rect(
            SIDEBAR_WIDTH, 0, self.screen_w - SIDEBAR_WIDTH, self.screen_h
        )

        avail_w = self.main_area_rect.width - 40
        avail_h = self.main_area_rect.height - 40
        scale_w = avail_w / CANVAS_SIZE[0]
        scale_h = avail_h / CANVAS_SIZE[1]
        self.scale = min(scale_w, scale_h, 1.0)

        self.disp_w = int(CANVAS_SIZE[0] * self.scale)
        self.disp_h = int(CANVAS_SIZE[1] * self.scale)
        self.canvas_rect = pygame.Rect(0, 0, self.disp_w, self.disp_h)
        self.canvas_rect.center = self.main_area_rect.center

        self.brush_size = 20
        self.brush_color = (255, 255, 255)
        self.drawing = False
        self.last_pos = None
        self.scroll_y = 0
        self.max_scroll = 0

        self.project_name = f"project_{int(time.time())}"
        self.prompt = "sci-fi metal texture"
        self.active_field = None
        self.show_hint = False
        self.show_load_modal = False

        self.fonts = {
            14: pygame.font.SysFont("Segoe UI", 14),
            16: pygame.font.SysFont("Segoe UI", 16),
            18: pygame.font.SysFont("Segoe UI", 18, bold=True),
            24: pygame.font.SysFont("Segoe UI", 24, bold=True),
        }

        self.size_slider = Slider(20, 480, SIDEBAR_WIDTH - 40, 2, 100, 20)

        self.hint_btn = Button(
            (SIDEBAR_WIDTH - 40, 10, 30, 30),
            "?",
            lambda: self.toggle_hint(),
            (60, 60, 65),
            18,
        )

        self.templates = sorted(
            [
                d
                for d in os.listdir("templates")
                if os.path.isdir(os.path.join("templates", d))
            ]
        )

        self.tpl_buttons = []
        for i, tpl in enumerate(self.templates):
            row = i // 3
            col = i % 3
            size = (SIDEBAR_WIDTH - 50) // 3
            x = 20 + col * (size + 5)
            y = 350 + row * (40)
            btn = Button(
                (x, y, size, 30),
                str(i + 1),
                lambda t=tpl: self.load_template(t),
                text_size=14,
            )
            self.tpl_buttons.append(btn)

        btn_y = 540
        self.buttons = [
            Button(
                (20, btn_y, SIDEBAR_WIDTH - 40, 40),
                "GENERATE",
                lambda: "GENERATE",
                THEME["accent"],
                18,
            ),
            Button(
                (20, btn_y + 50, SIDEBAR_WIDTH - 40, 40),
                "Clear Canvas",
                lambda: self.clear_canvas(),
            ),
            Button(
                (20, btn_y + 100, SIDEBAR_WIDTH - 40, 40),
                "Load Project",
                lambda: self.toggle_load_modal(),
            ),
            Button(
                (20, btn_y + 160, SIDEBAR_WIDTH - 40, 40),
                "Exit",
                lambda: "EXIT",
                (180, 50, 50),
            ),
        ]

        total_h = btn_y + 220
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
        projects = sorted(
            [
                d
                for d in os.listdir("projects")
                if os.path.isdir(os.path.join("projects", d))
            ]
        )
        self.project_list_buttons = []
        cols = 3
        w = 180
        h = 40
        gap = 10
        start_x = (self.screen_w - (cols * w + (cols - 1) * gap)) // 2
        start_y = 150
        for i, proj in enumerate(projects):
            r = i // cols
            c = i % cols
            rect = (start_x + c * (w + gap), start_y + r * (h + gap), w, h)
            cb = lambda p=proj: ("LOAD_PROJECT", p)
            self.project_list_buttons.append(Button(rect, proj[:20], cb, text_size=14))

    def load_template(self, folder_name):
        path = os.path.join("templates", folder_name)
        img_path = os.path.join(path, "sketch.png")
        if os.path.exists(img_path):
            try:
                img = pygame.image.load(img_path)
                img = pygame.transform.scale(img, CANVAS_SIZE)
                self.canvas_surf.blit(img, (0, 0))
            except:
                pass
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
            if event.type == MOUSEBUTTONDOWN:
                self.show_hint = False
            return None

        if self.show_load_modal:
            if event.type == MOUSEBUTTONDOWN:
                for btn in self.project_list_buttons:
                    res = btn.check_click(event.pos)
                    if res:
                        return res
                self.show_load_modal = False
            if event.type == MOUSEMOTION:
                for btn in self.project_list_buttons:
                    btn.check_hover(event.pos)
            return None

        if event.type == MOUSEWHEEL:
            if pygame.mouse.get_pos()[0] < SIDEBAR_WIDTH:
                self.scroll_y -= event.y * 20
                self.scroll_y = max(0, min(self.scroll_y, self.max_scroll))

        if event.type == MOUSEBUTTONDOWN:
            if self.hint_btn.check_click(event.pos):
                return None
        if event.type == MOUSEMOTION:
            self.hint_btn.check_hover(event.pos)

        if self.size_slider.handle_event(event, self.scroll_y):
            self.brush_size = int(self.size_slider.val)
            return None

        if event.type == MOUSEMOTION:
            for btn in self.buttons + self.tpl_buttons:
                btn.check_hover(event.pos, self.scroll_y)

            if self.drawing:
                c_pos = self.get_canvas_pos(event.pos)
                if c_pos and self.last_pos:
                    pygame.draw.line(
                        self.canvas_surf,
                        self.brush_color,
                        self.last_pos,
                        c_pos,
                        self.brush_size,
                    )
                    pygame.draw.circle(
                        self.canvas_surf,
                        self.brush_color,
                        self.last_pos,
                        self.brush_size // 2,
                    )
                    pygame.draw.circle(
                        self.canvas_surf, self.brush_color, c_pos, self.brush_size // 2
                    )
                    self.last_pos = c_pos
                elif c_pos:
                    self.last_pos = c_pos
                else:
                    self.last_pos = None

        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                for btn in self.buttons + self.tpl_buttons:
                    res = btn.check_click(event.pos, self.scroll_y)
                    if res:
                        return res

                proj_rect = pygame.Rect(20, 80 - self.scroll_y, SIDEBAR_WIDTH - 40, 30)
                prompt_rect = pygame.Rect(
                    20, 160 - self.scroll_y, SIDEBAR_WIDTH - 40, 100
                )

                if proj_rect.collidepoint(event.pos):
                    self.active_field = "project"
                elif prompt_rect.collidepoint(event.pos):
                    self.active_field = "prompt"
                else:
                    if not self.canvas_rect.collidepoint(event.pos):
                        self.active_field = None

                c_pos = self.get_canvas_pos(event.pos)
                if c_pos:
                    self.drawing = True
                    self.last_pos = c_pos
                    pygame.draw.circle(
                        self.canvas_surf, self.brush_color, c_pos, self.brush_size // 2
                    )

        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                self.drawing = False
                self.last_pos = None

        elif event.type == KEYDOWN:
            if self.active_field == "project":
                if event.key == K_RETURN:
                    self.active_field = None
                elif event.key == K_BACKSPACE:
                    self.project_name = self.project_name[:-1]
                else:
                    if len(self.project_name) < 20 and event.unicode.isprintable():
                        self.project_name += event.unicode
            elif self.active_field == "prompt":
                if event.key == K_RETURN:
                    self.active_field = None
                elif event.key == K_BACKSPACE:
                    self.prompt = self.prompt[:-1]
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
        pygame.draw.rect(
            screen, (10, 10, 10), (mx + 4, my + 4, mw, mh), border_radius=12
        )
        pygame.draw.rect(screen, (50, 50, 55), (mx, my, mw, mh), border_radius=12)
        pygame.draw.rect(
            screen, THEME["accent"], (mx, my, mw, mh), width=2, border_radius=12
        )
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
        pygame.draw.rect(
            screen, THEME["accent"], (mx, my, mw, mh), width=2, border_radius=12
        )
        t = self.fonts[24].render("Load Project", True, (255, 255, 255))
        screen.blit(t, (mx + 30, my + 30))
        if not self.project_list_buttons:
            msg = self.fonts[16].render(
                "No projects found in 'projects/' folder.", True, (200, 200, 200)
            )
            screen.blit(msg, (mx + 30, my + 100))
        else:
            for btn in self.project_list_buttons:
                btn.draw(screen, self.fonts)

    def draw_sidebar_content(self, screen):
        content_rect = pygame.Rect(0, 50, SIDEBAR_WIDTH, self.screen_h - 50)
        screen.set_clip(content_rect)

        off = -self.scroll_y

        lbl = self.fonts[14].render("Project Name:", True, (150, 150, 150))
        screen.blit(lbl, (20, 55 + off))
        proj_rect = pygame.Rect(20, 80 + off, SIDEBAR_WIDTH - 40, 30)
        bg = (
            THEME["input_active"]
            if self.active_field == "project"
            else THEME["input_bg"]
        )
        pygame.draw.rect(screen, bg, proj_rect, border_radius=4)
        txt = self.fonts[16].render(
            self.project_name
            + ("|" if self.active_field == "project" and time.time() % 1 > 0.5 else ""),
            True,
            (255, 255, 255),
        )
        screen.blit(txt, (25, 85 + off))

        lbl = self.fonts[14].render(
            f"Prompt ({len(self.prompt)}/60):", True, (150, 150, 150)
        )
        screen.blit(lbl, (20, 135 + off))
        prompt_rect = pygame.Rect(20, 160 + off, SIDEBAR_WIDTH - 40, 100)
        bg = (
            THEME["input_active"]
            if self.active_field == "prompt"
            else THEME["input_bg"]
        )
        pygame.draw.rect(screen, bg, prompt_rect, border_radius=4)
        words = self.prompt.split(" ")
        lines = []
        curr_line = ""
        for word in words:
            test_line = curr_line + word + " "
            if self.fonts[16].size(test_line)[0] < prompt_rect.width - 10:
                curr_line = test_line
            else:
                lines.append(curr_line)
                curr_line = word + " "
        lines.append(
            curr_line
            + ("|" if self.active_field == "prompt" and time.time() % 1 > 0.5 else "")
        )
        for i, line in enumerate(lines):
            t = self.fonts[16].render(line, True, (255, 255, 255))
            screen.blit(t, (25, 165 + i * 20 + off))

        lbl = self.fonts[14].render("Templates:", True, (150, 150, 150))
        screen.blit(lbl, (20, 320 + off))
        if not self.tpl_buttons:
            lbl_none = self.fonts[14].render(
                "No templates in 'templates/'", True, (100, 100, 100)
            )
            screen.blit(lbl_none, (20, 350 + off))
        for btn in self.tpl_buttons:
            btn.draw(screen, self.fonts, self.scroll_y)

        self.size_slider.draw(screen, self.fonts[14], self.scroll_y)
        for btn in self.buttons:
            btn.draw(screen, self.fonts, self.scroll_y)

        screen.set_clip(None)

    def draw(self, screen):
        clip = self.main_area_rect
        pygame.draw.rect(screen, THEME["bg"], clip)
        grid_sz = 40
        for y in range(0, clip.height, grid_sz):
            for x in range(0, clip.width, grid_sz):
                color = (
                    THEME["grid_light"]
                    if (x // grid_sz + y // grid_sz) % 2 == 0
                    else THEME["grid_dark"]
                )
                pygame.draw.rect(
                    screen, color, (clip.x + x, clip.y + y, grid_sz, grid_sz)
                )

        shadow = self.canvas_rect.inflate(4, 4)
        pygame.draw.rect(screen, (10, 10, 10), shadow)
        scaled_surf = pygame.transform.scale(
            self.canvas_surf, (self.disp_w, self.disp_h)
        )
        screen.blit(scaled_surf, self.canvas_rect)

        pygame.draw.rect(screen, THEME["sidebar"], (0, 0, SIDEBAR_WIDTH, self.screen_h))
        pygame.draw.line(
            screen, (50, 50, 50), (SIDEBAR_WIDTH, 0), (SIDEBAR_WIDTH, self.screen_h)
        )

        title = self.fonts[18].render("AI Material Studio", True, (255, 255, 255))
        screen.blit(title, (20, 15))

        self.hint_btn.draw(screen, self.fonts)

        self.draw_sidebar_content(screen)

        if self.max_scroll > 0:
            scroll_h = self.screen_h - 50
            bar_h = max(20, (scroll_h / (scroll_h + self.max_scroll)) * scroll_h)
            bar_y = 50 + (self.scroll_y / self.max_scroll) * (scroll_h - bar_h)
            pygame.draw.rect(
                screen,
                (60, 60, 60),
                (SIDEBAR_WIDTH - 6, bar_y, 4, bar_h),
                border_radius=2,
            )

        if self.show_hint:
            self.draw_modal(
                screen,
                "How to Paint",
                [
                    "1. Draw white shapes on black (Depth/Structure).",
                    "2. Name your project and type a prompt.",
                    "3. Use [ ] to change brush size.",
                    "4. Click GENERATE to create PBR textures.",
                    "5. Templates let you start quickly.",
                    "6. Character limit for prompt is 60.",
                ],
            )

        if self.show_load_modal:
            self.draw_load_modal(screen)

    def save_sketch(self, project_path):
        os.makedirs(project_path, exist_ok=True)
        sketch_path = os.path.join(project_path, "sketch_input.png")
        pygame.image.save(self.canvas_surf, sketch_path)
        return sketch_path

