import math
import ctypes
import numpy as np
from PIL import Image
from OpenGL.GL import *
import OpenGL.GL.shaders
import pygame

from src.config import GRID_RES, THEME

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
        try:
            test = glGetString(GL_VERSION)
            if test is None:
                raise RuntimeError("OpenGL context not active")
        except Exception as e:
            raise RuntimeError(
                f"OpenGL context error: {e}. Make sure OpenGL context is initialized before creating Renderer3D."
            )

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

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

    def draw(self, aspect_ratio):
        glUseProgram(self.shader)
        dist = max(0.2, self.cam[2])
        near = dist * 0.2
        far = dist * 50000
        proj = self.perspective(45, aspect_ratio, near, far)
        rx, ry = math.radians(self.cam[0]), math.radians(self.cam[1])
        cx = dist * math.sin(ry) * math.cos(rx)
        cy = dist * math.sin(rx)
        cz = dist * math.cos(ry) * math.cos(rx)
        view = self.lookat(
            np.array([cx, cy, cz]), np.array([0, 0, 0]), np.array([0, 1, 0])
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"), 1, True, proj
        )
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, True, view)
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "model"),
            1,
            True,
            np.identity(4, dtype=np.float32),
        )
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
    pygame.draw.rect(
        overlay, THEME["accent"], (mx, my, mw, mh), width=2, border_radius=12
    )
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


def draw_3d_controls_overlay(surface, w, h):
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)

    font = pygame.font.SysFont("Segoe UI", 14)
    small_font = pygame.font.SysFont("Segoe UI", 12)

    panel_w, panel_h = 220, 180
    px, py = 10, 10
    pygame.draw.rect(overlay, (20, 20, 25, 200), (px, py, panel_w, panel_h), border_radius=8)
    pygame.draw.rect(overlay, THEME["accent"], (px, py, panel_w, panel_h), width=1, border_radius=8)

    title = font.render("3D Controls", True, (255, 255, 255))
    overlay.blit(title, (px + 10, py + 8))

    controls = [
        "W/S - Rotate Up/Down",
        "A/D - Rotate Left/Right",
        "R/E - Zoom In/Out",
        "Q - Reset Camera",
        "H - Toggle Help",
        "ESC - Back to Paint"
    ]

    y = py + 30
    for ctrl in controls:
        text = small_font.render(ctrl, True, (200, 200, 200))
        overlay.blit(text, (px + 10, y))
        y += 22

    surface.blit(overlay, (0, 0))

