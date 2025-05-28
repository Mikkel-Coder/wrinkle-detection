import taichi as ti
from numpy import random
from src.sim.texture import UVTexture
from src.sim.cloth import ClothSimulation

@ti.data_oriented
class ImageRender:
    def __init__(self, texture: UVTexture, cloth: ClothSimulation):
        # Setup UI window and rendering components
        self.window = ti.ui.Window(
            name="cloth_sim",
            res=(1024, 1024),
            vsync=False,
            show_window=False,
        )
        self.texture = texture
        self.cloth = cloth

        self.camera = ti.ui.Camera()
        self.camera.position(0.0, 1.4, 0.0)
        self.camera.lookat(0.0, 0.0, 0.0)
        self.camera.up(0, 0, 1)

        self.canvas = self.window.get_canvas()

        self.scene = self.window.get_scene()
        self.scene.set_camera(self.camera)
    
    def random_pale_rgb(self):
        base = 0.7  # minimum lightness
        variance = 0.2
        r = base + random.uniform(0, variance)
        g = base + random.uniform(0, variance)
        b = base + random.uniform(0, variance)
        return (r, g, b)
        
    def render_mask_and_save(self, filename: str):
        self.scene.ambient_light((1.0, 1.0, 1.0))
        self.texture.apply_mask()
        self.scene.mesh(
            self.cloth.vertices,
            indices=self.cloth.indices,
            per_vertex_color=self.texture.colors,
            two_sided=True,
        )
        self.canvas.set_background_color((0.0, 0.0, 0.0)) 
        self.canvas.scene(self.scene)
        self.window.save_image(filename)

    def render_texture_and_save(self, filename: str, texture_index: int = None):
        # Set up the camera and lighting
        light_pos = (
            random.uniform(-1.0, 1.0),
            random.uniform(1.0, 2.0),
            random.uniform(1.5, 3.0)
        )
        self.scene.point_light(pos=light_pos, color=(1, 1, 1))
        self.scene.ambient_light((0.5, 0.5, 0.5))

        # Render the mesh
        self.texture.apply_texture(texture_index=texture_index)
        self.scene.mesh(
            self.cloth.vertices,
            indices=self.cloth.indices,
            per_vertex_color=self.texture.colors,
            two_sided=True,
        )
        self.canvas.set_background_color(self.random_pale_rgb())
        self.canvas.scene(self.scene)
        self.window.save_image(filename)
