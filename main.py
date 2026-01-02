import os
import sys
import threading
import pygame
from pygame.locals import *
from OpenGL.GL import *

from src.config import THEME
from src.timer import Timer
from src.pipeline import AIPipeline
from src.interface import PaintInterface
from src.renderer import Renderer3D, draw_modal_overlay_3d, draw_3d_controls_overlay

pipeline = None
loading_done = False
loading_msg = "Starting..."

generation_thread = None
generation_progress = 0.0
generation_status = ""
generated_files = []
generation_complete = False


def ai_loader_thread():
    global pipeline, loading_done, loading_msg
    try:
        loading_msg = "Loading AI Models..."
        with Timer("AI Pipeline Initialization"):
            pipeline = AIPipeline()
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
        with Timer(f"Complete Generation Workflow for '{project_name}'"):
            generated_files = pipeline.generate(
                prompt, sketch_path, project_name, progress_callback=cb
            )
        generation_complete = True
    except Exception as e:
        print(f"Generation Error: {e}")
        generation_complete = True


def main():
    global pipeline, loading_done, loading_msg
    global generation_thread, generation_progress, generation_status
    global generation_complete, generated_files

    pygame.init()
    info = pygame.display.Info()
    W = int(info.current_w * 0.8) if info.current_w > 0 else 1920
    H = int(info.current_h * 0.8) if info.current_h > 0 else 1080
    W = max(W, 1280)
    H = max(H, 720)

    t = threading.Thread(target=ai_loader_thread, daemon=True)
    t.start()

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Sketch-to-Material Studio - Loading...")
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
        pygame.draw.arc(
            screen, THEME["accent"], spinner_rect, spinner_angle, spinner_angle + 1.5, 5
        )
        spinner_angle += 0.1
        txt = font_large.render(loading_msg, True, (255, 255, 255))
        r = txt.get_rect(center=(W // 2, H // 2 + 20))
        screen.blit(txt, r)
        hint = font_small.render(
            "Initial setup may take a minute depending on GPU...", True, (150, 150, 150)
        )
        hr = hint.get_rect(center=(W // 2, H // 2 + 60))
        screen.blit(hint, hr)
        pygame.display.flip()

    painter = PaintInterface((W, H))
    renderer = None

    MODE = "PAINT"
    SHOW_3D_HELP = False

    running = True

    while running:
        clock.tick(60)
        events = pygame.event.get()

        if MODE == "GENERATING":
            for event in events:
                if event.type == QUIT:
                    running = False

            if generation_complete:
                MODE = "PRE_VIEW_INSTRUCTIONS"
                generation_thread.join()

            painter.draw(screen)
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))

            bar_w = 400
            bar_h = 20
            cx, cy = W // 2, H // 2

            txt = font_large.render(generation_status, True, (255, 255, 255))
            tr = txt.get_rect(center=(cx, cy - 40))
            screen.blit(txt, tr)

            pygame.draw.rect(
                screen,
                (50, 50, 50),
                (cx - bar_w // 2, cy, bar_w, bar_h),
                border_radius=10,
            )
            fill_w = int(bar_w * generation_progress)
            pygame.draw.rect(
                screen,
                THEME["accent"],
                (cx - bar_w // 2, cy, fill_w, bar_h),
                border_radius=10,
            )

            pygame.display.flip()
            continue

        if MODE == "PRE_VIEW_INSTRUCTIONS":
            for event in events:
                if event.type == MOUSEBUTTONDOWN or event.type == KEYDOWN:
                    MODE = "VIEW"
                    pygame.display.quit()
                    pygame.display.init()
                    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
                    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
                    pygame.display.gl_set_attribute(
                        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
                    )
                    screen = pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
                    pygame.display.set_caption("Sketch-to-Material Studio - 3D View")

                    try:
                        glClearColor(0.1, 0.1, 0.1, 1.0)
                        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                        glEnable(GL_DEPTH_TEST)
                        pygame.display.flip()
                    except Exception as e:
                        print(f"Warning: OpenGL initialization issue: {e}")

                    try:
                        renderer = Renderer3D()
                        renderer.load_textures(*generated_files)
                        SHOW_3D_HELP = False
                    except Exception as e:
                        print(f"Error creating renderer: {e}")
                        print("Falling back to paint mode...")
                        MODE = "PAINT"
                        pygame.display.quit()
                        pygame.display.init()
                        screen = pygame.display.set_mode((W, H))
                        pygame.display.set_caption("Sketch-to-Material Studio")
                        painter = PaintInterface((W, H))

            painter.draw(screen)
            painter.draw_modal(
                screen,
                "Ready to View",
                [
                    "1. Navigation Controls:",
                    "   W / S : Rotate Up / Down",
                    "   A / D : Rotate Left / Right",
                    "   R / E : Zoom In / Out",
                    "   Mouse Wheel : Zoom In / Out",
                    "",
                    "2. Press 'H' or click ? button to see",
                    "   controls again.",
                    "3. Press 'ESC' to return to drawing.",
                ],
            )
            pygame.display.flip()
            continue

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
                            os.path.join(p_path, "roughness.png"),
                        )
                        MODE = "PRE_VIEW_INSTRUCTIONS"

                elif action == "EXIT":
                    running = False
                elif action == "GENERATE":
                    project_path = os.path.join("projects", painter.project_name)
                    s_path = painter.save_sketch(project_path)
                    generation_progress = 0.0
                    generation_status = "Starting..."
                    generation_complete = False

                    generation_thread = threading.Thread(
                        target=generation_worker,
                        args=(painter.prompt, s_path, painter.project_name),
                        daemon=True,
                    )
                    generation_thread.start()
                    MODE = "GENERATING"

            elif MODE == "VIEW":
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        MODE = "PAINT"
                        pygame.display.quit()
                        pygame.display.init()
                        screen = pygame.display.set_mode((W, H))
                        pygame.display.set_caption("Sketch-to-Material Studio")
                        painter = PaintInterface((W, H))
                    elif event.key == K_h:
                        SHOW_3D_HELP = not SHOW_3D_HELP
                elif event.type == MOUSEWHEEL and renderer:
                    zoom_speed = 0.1
                    if event.y > 0:
                        renderer.cam[2] = max(0.1, renderer.cam[2] - zoom_speed)
                    elif event.y < 0:
                        renderer.cam[2] += zoom_speed
                elif event.type == MOUSEBUTTONDOWN and renderer:
                    info_btn_rect = pygame.Rect(10, 10, 32, 32)
                    if info_btn_rect.collidepoint(event.pos):
                        SHOW_3D_HELP = not SHOW_3D_HELP

        if MODE == "PAINT":
            painter.draw(screen)
            pygame.display.flip()

        elif MODE == "VIEW" and renderer:
            keys = pygame.key.get_pressed()
            if keys[K_w]:
                renderer.cam[0] = max(-89, renderer.cam[0] - 2)
            if keys[K_s]:
                renderer.cam[0] = min(89, renderer.cam[0] + 2)
            if keys[K_a]:
                renderer.cam[1] -= 2
            if keys[K_d]:
                renderer.cam[1] += 2
            if keys[K_r]:
                renderer.cam[2] = max(0.1, renderer.cam[2] - 0.05)
            if keys[K_e]:
                renderer.cam[2] += 0.05

            glClearColor(0.1, 0.1, 0.1, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
            glViewport(0, 0, W, H)
            renderer.draw(W / H)

            try:
                overlay_surf = pygame.display.get_surface()
                mouse_pos = pygame.mouse.get_pos()
                info_btn_rect = pygame.Rect(10, 10, 32, 32)
                info_btn_hover = info_btn_rect.collidepoint(mouse_pos)
                draw_3d_controls_overlay(overlay_surf, W, H, info_btn_hover)
                if SHOW_3D_HELP and overlay_surf:
                    draw_modal_overlay_3d(
                        overlay_surf,
                        W,
                        H,
                        "3D Controls",
                        [
                            "W / S : Rotate Up / Down",
                            "A / D : Rotate Left / Right",
                            "R / E : Zoom In / Out",
                            "Mouse Wheel : Zoom In / Out",
                            "H : Toggle Help",
                            "ESC : Return to Paint",
                        ],
                    )
            except:
                pass

            pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

