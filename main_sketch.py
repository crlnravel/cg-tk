import os
import sys
import math
import ctypes
import numpy as np
import cv2
import pygame
from pygame.locals import *
from PIL import Image

# --- AI Imports ---
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import pipeline as depth_pipeline

# --- OpenGL Imports ---
from OpenGL.GL import *
from OpenGL.GL import shaders
import OpenGL.GL.shaders

# CONFIG
WINDOW_SIZE = (1024, 768) # Slightly larger for painting
GRID_RES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available(): DEVICE = "mps"

# ==========================================
# 1. AI PIPELINE (IMG2IMG)
# ==========================================
class AIPipeline:
    def __init__(self):
        print("Loading Img2Img Pipeline...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)
        self.pipe.safety_checker = None 
        
        print("Loading Depth Estimator...")
        self.depth_pipe = depth_pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=DEVICE)

    def generate(self, prompt, init_image_path):
        # 1. Load Sketch
        init_image = Image.open(init_image_path).convert("RGB").resize((512, 512))
        
        # 2. Img2Img Generation
        # Strength 0.75 means: "Follow my colors, but add a lot of texture detail"
        pos = f"{prompt}, top down view, flat lighting, albedo texture, seamless, high resolution"
        neg = "shadows, highlights, perspective, tilted, depth of field, 3d render, lowres"
        
        print(f"Dreaming up '{prompt}' based on sketch...")
        image = self.pipe(prompt=pos, image=init_image, negative_prompt=neg, strength=0.75, guidance_scale=8.0).images[0]
        image.save("assets_albedo.png")

        # 3. Depth & Normal (Same as before)
        print("Calculating Geometry...")
        depth = self.depth_pipe(image)["depth"].resize((512, 512))
        depth.save("assets_depth.png")

        d_np = np.array(depth)
        sobelx = cv2.Sobel(d_np, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(d_np, cv2.CV_64F, 0, 1, ksize=3)
        ones = np.ones(d_np.shape)
        normal = np.dstack((-sobelx, -sobely, ones * 200.0)) 
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = ((normal / norm) * 0.5 + 0.5) * 255
        Image.fromarray(normal.astype(np.uint8)).save("assets_normal.png")
        
        return "assets_albedo.png", "assets_depth.png", "assets_normal.png"

# ==========================================
# 2. 2D PAINT CANVAS (Classical Raster) - DRAG & DROP VERSION
# ==========================================
class PaintCanvas:
    def __init__(self, surface):
        self.display = surface
        self.canvas = pygame.Surface(WINDOW_SIZE)
        self.canvas.fill((128, 128, 128)) # Grey background
        
        # Color Palette
        self.palette = [
            (0, 0, 0),       # 1: Black
            (255, 255, 255), # 2: White
            (255, 0, 0),     # 3: Red
            (0, 255, 0),     # 4: Green
            (0, 0, 255)      # 5: Blue
        ]
        
        self.brush_color = self.palette[0]
        self.brush_size = 20
        self.drawing = False
        self.last_pos = None

    def handle_event(self, event):
        # --- DRAG AND DROP HANDLER (Replaces Tkinter) ---
        if event.type == pygame.DROPFILE:
            try:
                filepath = event.file  # Get path from the drop event
                print(f"Loading dropped file: {filepath}")
                img = pygame.image.load(filepath)
                img = pygame.transform.scale(img, WINDOW_SIZE)
                self.canvas.blit(img, (0,0))
            except Exception as e:
                print(f"Error loading file: {e}")

        # BRUSH CONTROLS
        elif event.type == MOUSEBUTTONDOWN:
            self.drawing = True
            self.last_pos = event.pos
        elif event.type == MOUSEBUTTONUP:
            self.drawing = False
            self.last_pos = None
        elif event.type == MOUSEMOTION and self.drawing:
            if self.last_pos:
                pygame.draw.line(self.canvas, self.brush_color, self.last_pos, event.pos, self.brush_size)
                pygame.draw.circle(self.canvas, self.brush_color, event.pos, self.brush_size // 2)
            self.last_pos = event.pos
        
        # COLOR PALETTE SHORTCUTS
        elif event.type == KEYDOWN:
            if event.key == K_1: self.brush_color = self.palette[0]
            if event.key == K_2: self.brush_color = self.palette[1]
            if event.key == K_3: self.brush_color = self.palette[2]
            if event.key == K_4: self.brush_color = self.palette[3]
            if event.key == K_5: self.brush_color = self.palette[4]
            if event.key == K_c: self.canvas.fill((128, 128, 128))  # Clear

    def save_sketch(self):
        pygame.image.save(self.canvas, "sketch_input.png")
        return "sketch_input.png"

    def draw(self):
        # 1. Draw Sketch
        self.display.blit(self.canvas, (0, 0))
        
        # 2. UI Overlay
        pygame.draw.rect(self.display, (50, 50, 50), (0, 0, WINDOW_SIZE[0], 60))
        font = pygame.font.SysFont("Arial", 16)
        
        # Update Instructions text
        text_surf = font.render("Draw with Mouse | Drag & Drop Image to Upload | 'C': Clear | ENTER: Generate 3D", True, (200, 200, 200))
        self.display.blit(text_surf, (250, 20))
        
        # Draw Palette
        start_x = 10
        start_y = 10
        box_size = 40
        for i, color in enumerate(self.palette):
            rect = pygame.Rect(start_x + (i * 50), start_y, box_size, box_size)
            pygame.draw.rect(self.display, color, rect)
            if self.brush_color == color:
                pygame.draw.rect(self.display, (255, 255, 0), rect, 3)
            else:
                pygame.draw.rect(self.display, (100, 100, 100), rect, 1)
            num_surf = font.render(str(i+1), True, (128, 128, 128) if color == (255,255,255) else (255, 255, 255))
            self.display.blit(num_surf, (rect.x + 15, rect.y + 10))

        # Mouse Cursor
        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos[1] > 60:
            pygame.draw.circle(self.display, self.brush_color, mouse_pos, self.brush_size // 2, 1)
            
# ==========================================
# 3. 3D RENDERER (Classical OpenGL)
# ==========================================
# (Shaders are same as previous, shortened for brevity)
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
    FragNormal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""
FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
in vec2 FragTexCoords;
in vec3 FragPos;
in vec3 FragNormal;
uniform sampler2D albedoMap, normalMap;
uniform vec3 lightPos, viewPos;
void main() {
    FragColor = vec4(1.0, 0.0, 1.0, 1.0); 
    vec3 color = texture(albedoMap, FragTexCoords).rgb;
    vec3 normRGB = texture(normalMap, FragTexCoords).rgb;
    vec3 finalNormal = normalize(FragNormal + normalize(normRGB * 2.0 - 1.0) * 0.5);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(finalNormal, lightDir), 0.0);
    vec3 viewDir = normalize(viewPos - FragPos);
    float spec = pow(max(dot(finalNormal, normalize(lightDir + viewDir)), 0.0), 32.0);
    FragColor = vec4((vec3(0.2) + diff + spec*0.5) * color, 1.0);
}
"""

# ==========================================
# 3. 3D RENDERER (Classical OpenGL) - MAC FIX
# ==========================================
class Renderer3D:
    def __init__(self):
        # --- MAC FIX: Create VAO *BEFORE* Compiling Shaders ---
        # macOS requires a VAO to be bound for shader validation to pass.
        self.create_plane()
        
        self.shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        
        self.texture_ids = [0, 0, 0]
        self.cam = [90, 0, 5.0]

    def create_plane(self):
        # 1. Generate Grid Data (Python CPU side)
        verts, inds = [], []
        scale = 0.2  # Scale down the mesh
        for z in range(GRID_RES + 1):
            for x in range(GRID_RES + 1):
                # Pos (x, y, z), UV (u, v), Normal (nx, ny, nz)
                # Flipped V coordinate to fix vertical flip: 1.0 - (z/GRID_RES)
                verts.extend([((x/GRID_RES)*2-1)*scale, 0, ((z/GRID_RES)*2-1)*scale, x/GRID_RES, 1.0 - (z/GRID_RES), 0, 1, 0])
        for z in range(GRID_RES):
            for x in range(GRID_RES):
                tl, tr = z*(GRID_RES+1)+x, z*(GRID_RES+1)+x+1
                bl, br = (z+1)*(GRID_RES+1)+x, (z+1)*(GRID_RES+1)+x+1
                inds.extend([tl, bl, tr, tr, bl, br])
        
        self.count = len(inds)
        v_data = np.array(verts, dtype=np.float32)
        i_data = np.array(inds, dtype=np.uint32)
        
        # 2. OpenGL Buffers (GPU side)
        
        # A. Create and Bind VAO first (Critical for Mac)
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # B. Create and Bind VBO (Vertex Data)
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, v_data.nbytes, v_data, GL_STATIC_DRAW)
        
        # C. Create and Bind EBO (Index Data)
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, i_data.nbytes, i_data, GL_STATIC_DRAW)
        
        # D. Set Attribute Pointers (Layout)
        stride = 32 # 8 floats * 4 bytes
        # Pos (Location 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # UV (Location 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        # Normal (Location 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))

    def load_textures(self, albedo, depth, normal):
        files = [albedo, depth, normal]
        for i, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            data = np.array(list(img.getdata()), np.uint8)
            if self.texture_ids[i] == 0: self.texture_ids[i] = glGenTextures(1)
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, self.texture_ids[i])
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
            glGenerateMipmap(GL_TEXTURE_2D)

    def draw(self):
        glUseProgram(self.shader)
        
        w, h = pygame.display.get_surface().get_size()
        glViewport(0, 0, w, h)
        
        # Ensure VAO is bound before drawing
        glBindVertexArray(self.VAO)
        
        glDisable(GL_CULL_FACE) 

        # Projection & View Logic
        proj = self.perspective(45, WINDOW_SIZE[0]/WINDOW_SIZE[1], 0.1, 100.0)
        
        rx, ry = math.radians(self.cam[0]), math.radians(self.cam[1])
        cx = self.cam[2] * math.sin(ry) * math.cos(rx)
        cy = self.cam[2] * math.sin(rx)
        cz = self.cam[2] * math.cos(ry) * math.cos(rx)
        
        # LookAt
        f = -np.array([cx, cy, cz]); f /= np.linalg.norm(f)
        s = np.cross(f, np.array([0,1,0])); s /= np.linalg.norm(s)
        u = np.cross(s, f)
        view = np.identity(4, dtype=np.float32)
        view[:3,:3] = np.array([s,u,-f])
        view[:3,3] = -np.dot(np.array([s,u,-f]), np.array([cx, cy, cz]))

        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, False, proj)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, False, view)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, False, np.identity(4, dtype=np.float32))
        
        glUniform1f(glGetUniformLocation(self.shader, "displacementStrength"), 0.3)
        glUniform3f(glGetUniformLocation(self.shader, "lightPos"), 2, 4, 2)
        glUniform3f(glGetUniformLocation(self.shader, "viewPos"), cx, cy, cz)
        
        for i, name in enumerate(["albedoMap", "depthMap", "normalMap"]):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, self.texture_ids[i])
            glUniform1i(glGetUniformLocation(self.shader, name), i)
            
        glDrawElements(GL_TRIANGLES, self.count, GL_UNSIGNED_INT, None)

    def perspective(self, fov, aspect, near, far):
        f = 1.0 / math.tan(math.radians(fov) / 2)
        mat = np.zeros((4, 4), dtype=np.float32)
        mat[0, 0] = f / aspect
        mat[1, 1] = f
        mat[2, 2] = -(far + near) / (far - near)  # Fixed sign for proper depth
        mat[2, 3] = -1.0
        mat[3, 2] = -(2 * far * near) / (far - near)  # Fixed sign for proper depth
        mat[3, 3] = 0.0 # Explicitly set w to 0
        return mat

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    # 1. INIT AI
    ai = AIPipeline()
    
    # 2. INIT PYGAME
    pygame.init()
    # We start in standard mode (Paint), then switch to OPENGL mode (3D)
    screen = pygame.display.set_mode(WINDOW_SIZE) 
    pygame.display.set_caption("Sketch-to-Material Studio")
    
    painter = PaintCanvas(screen)
    renderer = None # Created only when switching to 3D mode
    
    MODE = "PAINT" # or "VIEW"
    running = True
    clock = pygame.time.Clock()

    while running:
        clock.tick(60)
        events = pygame.event.get()
        
        for event in events:
            if event.type == QUIT: running = False
            
            # --- PAINT MODE INPUTS ---
            if MODE == "PAINT":
                painter.handle_event(event)
                
                # GENERATE TRIGGER
                if event.type == KEYDOWN and event.key == K_RETURN:
                    # 1. Save Sketch
                    sketch_path = painter.save_sketch()
                    
                    # 2. Get Prompt
                    prompt = input("Enter prompt for your sketch (console): ")
                    if not prompt: prompt = "Detailed texture"
                    
                    # 3. Run AI (Blocking)
                    files = ai.generate(prompt, sketch_path)
                    
                    # 4. Switch to 3D Mode
                    MODE = "VIEW"
                    
                    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
                    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
                    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
                    
                    screen = pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
                    renderer = Renderer3D()
                    renderer.load_textures(*files)

            # --- VIEW MODE INPUTS ---
            elif MODE == "VIEW":
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    # Switch back to Paint Mode
                    MODE = "PAINT"
                    screen = pygame.display.set_mode(WINDOW_SIZE) # Reset to 2D
                    
                if event.type == MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                    renderer.cam[1] += event.rel[0]*0.5
                    renderer.cam[0] = max(-89, min(89, renderer.cam[0] + event.rel[1]*0.5))

        # --- DRAW LOOP ---
        if MODE == "PAINT":
            painter.draw()
            pygame.display.flip()
            
        elif MODE == "VIEW":
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
            renderer.draw()
            pygame.display.flip()

    pygame.quit()
