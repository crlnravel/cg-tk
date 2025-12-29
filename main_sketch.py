import os
import sys

import math

import ctypes

import numpy as np

import cv2

import pygame

from pygame.locals import *

from PIL import Image, ImageOps


# --- AI Imports ---

import torch

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from transformers import pipeline as depth_pipeline


# --- OpenGL Imports ---

from OpenGL.GL import *

from OpenGL.GL import shaders

import OpenGL.GL.shaders


# CONFIG

WINDOW_SIZE = (1024, 768)

GRID_RES = 200  # Increased resolution for smoother 3D mesh

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.backends.mps.is_available():
    DEVICE = "mps"


os.makedirs("assets", exist_ok=True)


# ==========================================

# 1. ADVANCED AI PIPELINE (FIXED)

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

    def generate(self, prompt, init_image_path):
        # 1. Prepare Sketch

        sketch = Image.open(init_image_path).convert("RGB").resize((512, 512))

        sketch_gray = sketch.convert("L")

        sketch_array = np.array(sketch_gray)

        # Stronger threshold for clean lines

        _, binary = cv2.threshold(sketch_array, 127, 255, cv2.THRESH_BINARY)

        sketch = Image.fromarray(binary).convert("RGB")

        sketch = ImageOps.invert(sketch)

        # 2. Generate Albedo

        pos_prompt = f"{prompt}, seamless texture, top down view, flat lighting, 8k, highly detailed, photorealistic material"

        neg_prompt = "perspective, 3d, shadows, people, objects, text, watermark, blurry, low quality, distortion"

        print(f"Generating '{prompt}'...")

        albedo = self.pipe(
            pos_prompt,
            image=sketch,
            negative_prompt=neg_prompt,
            num_inference_steps=30,
            controlnet_conditioning_scale=0.7,
            guidance_scale=8.0,
        ).images[0]

        albedo.save("assets/albedo.png")

        # 3. Generate Roughness (Inverted Intensity)

        img_cv = cv2.cvtColor(np.array(albedo), cv2.COLOR_RGB2GRAY)

        roughness = cv2.normalize(img_cv, None, 0, 255, cv2.NORM_MINMAX)

        roughness = 255 - roughness

        Image.fromarray(roughness).save("assets/roughness.png")

        # 4. Generate Depth Map (FIXED NORMALIZATION)

        print("Calculating Geometry...")

        depth_data = self.depth_pipe(albedo)["depth"].resize((512, 512))

        depth_np = np.array(depth_data)

        # Normalize depth to full 0-255 range for maximum 3D detail

        d_min, d_max = depth_np.min(), depth_np.max()

        if d_max > d_min:
            depth_np = ((depth_np - d_min) / (d_max - d_min)) * 255.0

        depth_img = Image.fromarray(depth_np.astype(np.uint8))

        depth_img.save("assets/depth.png")

        # 5. Generate Normal Map (IMPROVED MATH)

        # Use stronger gradients for more visible bump mapping

        d_np = depth_np.astype(np.float32)

        sobelx = cv2.Sobel(d_np, cv2.CV_64F, 1, 0, ksize=5)

        sobely = cv2.Sobel(d_np, cv2.CV_64F, 0, 1, ksize=5)

        # The Z component determines flatness. Lower Z = deeper bumps.

        normal = np.dstack((-sobelx, -sobely, np.ones_like(d_np) * 500.0))

        norm = np.linalg.norm(normal, axis=2, keepdims=True)

        normal = ((normal / norm) * 0.5 + 0.5) * 255

        Image.fromarray(normal.astype(np.uint8)).save("assets/normal.png")

        return (
            "assets/albedo.png",
            "assets/depth.png",
            "assets/normal.png",
            "assets/roughness.png",
        )


# ==========================================

# 2. ENHANCED PAINT CANVAS (Unchanged)

# ==========================================


class PaintCanvas:
    def __init__(self, surface):
        self.display = surface

        self.canvas = pygame.Surface(WINDOW_SIZE)

        self.canvas.fill((0, 0, 0))

        self.color_palette = [
            (255, 255, 255),
            (200, 200, 200),
            (128, 128, 128),
            (64, 64, 64),
            (0, 0, 0),
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
        ]

        self.current_color_index = 0

        self.brush_color = self.color_palette[0]

        self.brush_size = 20

        self.brush_mode = "normal"

        self.drawing = False

        self.last_pos = None

        self.prompt_input = ""

        self.entering_prompt = False

    def handle_event(self, event):
        if self.entering_prompt:
            if event.type == KEYDOWN:
                if event.key == K_RETURN:
                    self.entering_prompt = False

                    return "GENERATE"

                elif event.key == K_BACKSPACE:
                    self.prompt_input = self.prompt_input[:-1]

                elif event.key == K_ESCAPE:
                    self.entering_prompt = False

                    self.prompt_input = ""

                elif event.unicode.isprintable():
                    self.prompt_input += event.unicode

            return None

        if event.type == MOUSEBUTTONDOWN:
            if event.pos[1] > 50:
                self.drawing = True

                self.last_pos = event.pos

                self._draw_at(event.pos)

        elif event.type == MOUSEBUTTONUP:
            self.drawing = False

            self.last_pos = None

        elif event.type == MOUSEMOTION and self.drawing:
            if self.last_pos:
                pygame.draw.line(
                    self.canvas,
                    self.brush_color,
                    self.last_pos,
                    event.pos,
                    self.brush_size,
                )

            self._draw_at(event.pos)

            self.last_pos = event.pos

        elif event.type == KEYDOWN:
            if event.key == K_c:
                self.canvas.fill((0, 0, 0))

            elif event.key == K_RETURN:
                self.entering_prompt = True

                self.prompt_input = ""

            elif event.key == K_l:
                # Load previous assets

                return "LOAD_PREVIOUS"

            elif event.key == K_SPACE:
                self.current_color_index = (self.current_color_index + 1) % len(
                    self.color_palette
                )

                self.brush_color = self.color_palette[self.current_color_index]

            elif event.key == K_LEFTBRACKET:
                self.brush_size = max(5, self.brush_size - 5)

            elif event.key == K_RIGHTBRACKET:
                self.brush_size += 5

    def _draw_at(self, pos):
        pygame.draw.circle(self.canvas, self.brush_color, pos, self.brush_size // 2)

    def save_sketch(self):
        pygame.image.save(self.canvas, "sketch_input.png")

        return "sketch_input.png"

    def draw(self):
        self.display.blit(self.canvas, (0, 0))

        pygame.draw.rect(self.display, (40, 40, 40), (0, 0, WINDOW_SIZE[0], 50))

        font = pygame.font.SysFont("Arial", 16)

        if self.entering_prompt:
            txt = font.render(f"Prompt: {self.prompt_input}_", True, (255, 255, 0))

            self.display.blit(txt, (10, 15))

        else:
            txt = font.render(
                "Draw White on Black. SPACE:Color, []:Size, ENTER:Generate, L:Load Previous",
                True,
                (200, 200, 200),
            )

            self.display.blit(txt, (10, 15))

        # Cursor

        m_pos = pygame.mouse.get_pos()

        if m_pos[1] > 50:
            pygame.draw.circle(
                self.display, self.brush_color, m_pos, self.brush_size // 2, 2
            )


# ==========================================

# 3. 3D RENDERER (FIXED SHADERS)

# ==========================================


# Vertex Shader: Cleaner Displacement

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

    

    // Sample depth

    float depth = texture(depthMap, texCoords).r;

    

    // Displace along the normal

    vec3 displacedPos = position + (normal * depth * displacementStrength);

    

    FragPos = vec3(model * vec4(displacedPos, 1.0));

    

    // Pass normal to fragment shader

    mat3 normalMatrix = mat3(transpose(inverse(model)));

    FragNormal = normalMatrix * normal;

    

    gl_Position = projection * view * vec4(FragPos, 1.0);

}

"""


# Fragment Shader: Proper Tangent Space Normal Mapping

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


// Calculate TBN Matrix automatically using derivatives

// This avoids sending pre-calculated tangents from Python

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

    

    // Get proper normal from normal map

    vec3 norm = getNormalFromMap();

    

    // Lighting

    vec3 lightDir = normalize(lightPos - FragPos);

    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 halfwayDir = normalize(lightDir + viewDir);

    

    // Diffuse

    float diff = max(dot(norm, lightDir), 0.0);

    

    // Specular (Blinn-Phong)

    float shininess = mix(2.0, 64.0, 1.0 - rough); 

    float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);

    

    // Rim Light (Fresnel-ish effect for 3D pop)

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
        self.cam = [0, 0, 1]  # Camera sits at Z=3.5 looking at the origin

    def create_plane(self):
        verts, inds = [], []
        scale = 0.3  # Scale factor

        # We iterate 'y' (rows) and 'x' (cols)
        # To make it vertical, we map loop variables to X and Y coords, keeping Z at 0.
        for y in range(GRID_RES + 1):
            for x in range(GRID_RES + 1):
                u = x / GRID_RES
                v = y / GRID_RES

                # --- GEOMETRY CHANGES ---
                # 1. X Position: standard -0.5 to 0.5
                px = (u - 0.5) * scale

                # 2. Y Position: We flip v so 0 (top of image) is positive Y (top of 3D space)
                py = (v - 0.5) * scale

                # 3. Z Position: Flat at 0
                pz = 0.0

                # 4. Normals: Point Positive Z (Towards camera)
                # This ensures displacement moves "out" towards the viewer
                nx, ny, nz = 0.0, 0.0, 1.0

                # Append: x, y, z, u, v, nx, ny, nz
                verts.extend([px, py, pz, u, 1.0 - v, nx, ny, nz])

        # Indices generation remains the same logic (connecting the grid squares)
        for y in range(GRID_RES):
            for x in range(GRID_RES):
                tl = y * (GRID_RES + 1) + x
                tr = y * (GRID_RES + 1) + x + 1
                bl = (y + 1) * (GRID_RES + 1) + x
                br = (y + 1) * (GRID_RES + 1) + x + 1
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

        # Stride is 8 floats * 4 bytes = 32
        stride = 32
        # Pos (0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # UV (1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        # Normal (2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))

    def load_textures(self, albedo_p, depth_p, normal_p, rough_p):
        paths = [albedo_p, normal_p, rough_p, depth_p]

        for i, path in enumerate(paths):
            img = Image.open(path).convert("RGB")
            # Flip image vertically for OpenGL texture coordinates if needed,
            # though we handled 1.0-v in vertices, doing it here is sometimes safer for normal maps.
            # img = ImageOps.flip(img)

            data = np.array(img, dtype=np.uint8)

            if self.texture_ids[i] == 0:
                self.texture_ids[i] = glGenTextures(1)

            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, self.texture_ids[i])

            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGB,
                img.width,
                img.height,
                0,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                data,
            )
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(
                GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def draw(self):
        glUseProgram(self.shader)
        w, h = pygame.display.get_surface().get_size()
        glViewport(0, 0, w, h)

        dist = self.cam[2]
        near = dist * 0.2  # Near plane at 1% of distance, minimum 0.1
        far = dist * 50000  # Far plane at 100x distanc

        proj = self.perspective(45, w / h, near, far)

        # Orbit Camera Logic
        rx, ry = math.radians(self.cam[0]), math.radians(self.cam[1])

        cx = dist * math.sin(ry) * math.cos(rx)
        cy = dist * math.sin(rx)
        cz = dist * math.cos(ry) * math.cos(rx)

        view = self.lookat(
            np.array([cx, cy, cz]), np.array([0, 0, 0]), np.array([0, 1, 0])
        )

        # --- THE FIX IS HERE ---
        # We change the 3rd argument from False to True.
        # This tells OpenGL: "My matrix is sideways, please flip it."

        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"), 1, True, proj
        )

        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, True, view)
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "model"),
            1,
            True,  # Also flip the model matrix (identity is same flipped, but good habit)
            np.identity(4, dtype=np.float32),
        )

        glUniform1f(glGetUniformLocation(self.shader, "displacementStrength"), 0.3)
        glUniform3f(glGetUniformLocation(self.shader, "lightPos"), 2.0, 4.0, 5.0)
        glUniform3f(glGetUniformLocation(self.shader, "viewPos"), cx, cy, cz)

        tex_locs = ["albedoMap", "normalMap", "roughnessMap", "depthMap"]
        for i, name in enumerate(tex_locs):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, self.texture_ids[i])
            glUniform1i(
                glGetUniformLocation(self.shader, "albedoMap" if i == 0 else name), i
            )

        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, self.count, GL_UNSIGNED_INT, None)

    def perspective(self, fov, aspect, near, far):
        f = 1.0 / math.tan(math.radians(fov) / 2)
        return np.array(
            [
                [f / aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) / (near - far), -1],
                [0, 0, (2 * far * near) / (near - far), 0],
            ],
            dtype=np.float32,
        )

    def lookat(self, eye, target, up):
        f = normalize(target - eye)
        s = normalize(np.cross(f, up))
        u = np.cross(s, f)
        m = np.identity(4, dtype=np.float32)
        m[:3, :3] = np.array([s, u, -f])
        m[:3, 3] = -np.dot(m[:3, :3], eye)
        return m


def normalize(v):
    norm = np.linalg.norm(v)

    return v / norm if norm > 0 else v


# ==========================================

# UTILITY: Check if previous assets exist

# ==========================================


def check_previous_assets():
    """Check if all required asset files exist"""

    required_files = [
        "assets/albedo.png",
        "assets/depth.png",
        "assets/normal.png",
        "assets/roughness.png",
    ]

    return all(os.path.exists(f) for f in required_files)


def load_previous_assets():
    """Return paths to previous asset files"""

    return (
        "assets/albedo.png",
        "assets/depth.png",
        "assets/normal.png",
        "assets/roughness.png",
    )


# ==========================================

# MAIN

# ==========================================

if __name__ == "__main__":
    pygame.init()

    # Initial setup for 2D

    screen = pygame.display.set_mode(WINDOW_SIZE)

    pygame.display.set_caption("AI Material Studio v2 (Fixed)")

    ai = AIPipeline()

    painter = PaintCanvas(screen)

    renderer = None

    MODE = "PAINT"

    running = True

    clock = pygame.time.Clock()

    while running:
        clock.tick(60)

        events = pygame.event.get()

        for event in events:
            if event.type == QUIT:
                running = False

            if MODE == "PAINT":
                res = painter.handle_event(event)

                if res == "GENERATE":
                    s_path = painter.save_sketch()

                    prompt = (
                        painter.prompt_input
                        if painter.prompt_input
                        else "scifi metal wall"
                    )

                    files = ai.generate(prompt, s_path)

                    # SWITCH TO OPENGL MODE

                    MODE = "VIEW"

                    pygame.display.quit()

                    pygame.display.init()

                    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)

                    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)

                    pygame.display.gl_set_attribute(
                        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
                    )

                    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

                    screen = pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)

                    renderer = Renderer3D()

                    renderer.load_textures(*files)

                elif res == "LOAD_PREVIOUS":
                    # Check if previous assets exist

                    if check_previous_assets():
                        print("Loading previous assets...")

                        files = load_previous_assets()

                        # SWITCH TO OPENGL MODE

                        MODE = "VIEW"

                        pygame.display.quit()

                        pygame.display.init()

                        pygame.display.gl_set_attribute(
                            pygame.GL_CONTEXT_MAJOR_VERSION, 3
                        )

                        pygame.display.gl_set_attribute(
                            pygame.GL_CONTEXT_MINOR_VERSION, 3
                        )

                        pygame.display.gl_set_attribute(
                            pygame.GL_CONTEXT_PROFILE_MASK,
                            pygame.GL_CONTEXT_PROFILE_CORE,
                        )

                        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

                        screen = pygame.display.set_mode(
                            WINDOW_SIZE, DOUBLEBUF | OPENGL
                        )

                        renderer = Renderer3D()

                        renderer.load_textures(*files)

                    else:
                        print("No previous assets found. Generate something first!")

            elif MODE == "VIEW":
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    MODE = "PAINT"

                    pygame.display.quit()

                    pygame.display.init()

                    screen = pygame.display.set_mode(WINDOW_SIZE)

                    painter = PaintCanvas(screen)  # Re-init painter surface

                if event.type == MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                    renderer.cam[1] += event.rel[0] * 0.5

                    renderer.cam[0] = max(
                        -89, min(89, renderer.cam[0] + event.rel[1] * 0.5)
                    )

                # Zoom with wheel

                if event.type == MOUSEWHEEL:
                    renderer.cam[2] = max(0.5, renderer.cam[2] - event.y * 0.01)

        if MODE == "PAINT":
            painter.draw()

            pygame.display.flip()

        elif MODE == "VIEW":
            glClearColor(0.1, 0.1, 0.1, 1.0)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glEnable(GL_DEPTH_TEST)

            renderer.draw()

            pygame.display.flip()

    pygame.quit()

