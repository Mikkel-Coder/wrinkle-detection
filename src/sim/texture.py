from pathlib import Path
import taichi as ti
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from src.sim.cloth import ClothSimulation


@ti.data_oriented
class UVTexture:
    def __init__(self, cloth: ClothSimulation):
        self.cloth = cloth
        self.n = cloth.n

        self.textures_path = Path("textures")
        self.texture_fields = []
        self.texture_fields_len = None
        self.texture_field = ti.Vector.field(3, dtype=float, shape=(512, 512))

        self.uv_coords = ti.Vector.field(2, dtype=float, shape=(self.n, self.n))
        self.initialize_uv()
        self.colors = ti.Vector.field(3, dtype=float, shape=self.n * self.n)

        self.load_textures()

    def load_textures(self):
        texture_files = sorted(
            self.textures_path.iterdir(),
            key=lambda f: int(f.stem)
        )

        for file in texture_files:
            texture_img = ti.tools.imread(str(file))
            texture_img = texture_img / 255.0  # Normalize to [0,1]
            texture_field = ti.Vector.field(
                3, dtype=float, shape=texture_img.shape[:2]
            )
            texture_field.from_numpy(texture_img)
            self.texture_fields.append(texture_field)

        self.texture_fields_len = len(self.texture_fields)

    @ti.kernel
    def initialize_uv(self):
        for i, j in ti.ndrange(self.n, self.n):
            self.uv_coords[i, j] = ti.Vector([i / (self.n - 1), j / (self.n - 1)], dt=float)


    def apply_mask(self):
        kernel_size = 3
        mask_np = self.cloth.winkle_mask.to_numpy().astype(np.uint8)
        mask_np *= 255
        mask_np = cv.resize(mask_np, (self.n, self.n), interpolation=cv.INTER_NEAREST)
        # Black out edges (top, bottom, left, right)
        mask_np[:kernel_size, :] = 0
        mask_np[-kernel_size:, :] = 0
        mask_np[:, :kernel_size] = 0
        mask_np[:, -kernel_size:] = 0

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened_mask = cv.morphologyEx(mask_np, cv.MORPH_OPEN, kernel)
        closed_mask = cv.morphologyEx(opened_mask, cv.MORPH_CLOSE, kernel)
        mask_flat = closed_mask.flatten().astype(np.float32)
        rgb_mask = np.stack([mask_flat] * 3, axis=1)
        
        self.colors.from_numpy(rgb_mask)

    def apply_texture(self, texture_index=None):
        # Overwrite the static GPU texture buffer with a random texture
        if texture_index is None:
            idx = np.random.randint(0, self.texture_fields_len)
        else:
            idx = texture_index
        self.texture_field.copy_from(self.texture_fields[idx])
        angle = np.random.uniform(0, 2 * np.pi)
        offset_u = np.random.uniform(0, 1)
        offset_v = np.random.uniform(0, 1)
        
        # Call kernel with those parameters
        self.apply_random_texture(angle, offset_u, offset_v)
    
    @ti.kernel
    def apply_random_texture(self, angle: float, offset_u: float, offset_v: float):
        cos_angle = ti.cos(angle)
        sin_angle = ti.sin(angle)

        for i, j in ti.ndrange(self.n, self.n):
            u, v = self.uv_coords[i, j]

            # Center UV at 0.5
            u_c = u - 0.5
            v_c = v - 0.5

            # Rotate UV
            u_rot = u_c * cos_angle - v_c * sin_angle
            v_rot = u_c * sin_angle + v_c * cos_angle

            # Translate back and add offset with wrap-around
            u_final = (u_rot + 0.5 + offset_u) % 1.0
            v_final = (v_rot + 0.5 + offset_v) % 1.0

            tex_w, tex_h = self.texture_field.shape

            x = u_final * (tex_w - 1)
            y = v_final * (tex_h - 1)

            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, tex_w - 1), min(y0 + 1, tex_h - 1)
            tx, ty = x - x0, y - y0

            c00 = self.texture_field[x0, y0]
            c10 = self.texture_field[x1, y0]
            c01 = self.texture_field[x0, y1]
            c11 = self.texture_field[x1, y1]

            color_top = c00 * (1 - tx) + c10 * tx
            color_bottom = c01 * (1 - tx) + c11 * tx
            final_color = color_top * (1 - ty) + color_bottom * ty

            self.colors[i * self.n + j] = final_color