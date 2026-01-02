import pygame
from pygame.locals import *

from src.config import THEME


class Button:
    def __init__(self, rect, text, callback, color=THEME["button"], text_size=16):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.color = color
        self.hover = False
        self.text_size = text_size

    def draw(self, surface, font_dict, scroll_y=0):
        draw_rect = self.rect.move(0, -scroll_y)
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
        lbl = font.render(f"Brush Size: {int(self.val)}px", True, (150, 150, 150))
        screen.blit(lbl, (draw_rect.x, draw_rect.y - 22))

        track_rect = pygame.Rect(draw_rect.x, draw_rect.y + 8, draw_rect.width, 4)
        pygame.draw.rect(screen, (60, 60, 60), track_rect, border_radius=2)

        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = draw_rect.x + ratio * draw_rect.width
        handle_rect = pygame.Rect(handle_x - 8, draw_rect.y, 16, 20)
        col = THEME["accent"] if self.dragging else (120, 120, 120)
        pygame.draw.rect(screen, col, handle_rect, border_radius=4)

    def handle_event(self, event, scroll_y=0):
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

