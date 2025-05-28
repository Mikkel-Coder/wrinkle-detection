from pathlib import Path
import taichi as ti
from tqdm import tqdm
from src.sim.texture import UVTexture
from src.sim.cloth import ClothSimulation
from src.sim.render import ImageRender
from src.ml.precompute_points import generate_points

def main():
    N_TEXTURES = 10
    cloth_sim = ClothSimulation(n=256, time=1.2)
    texture = UVTexture(cloth=cloth_sim)
    render = ImageRender(texture=texture, cloth=cloth_sim)

    dataset_path = Path("dataset")
    dataset_train_path = dataset_path / "train"
    dataset_train_image_path = dataset_train_path / "images"
    dataset_train_mask_path = dataset_train_path / "masks"
    dataset_test_path = dataset_path / "test"
    dataset_test_image_path = dataset_test_path / "images"
    dataset_test_mask_path = dataset_test_path / "masks"

    dirs = [
        dataset_path,
        dataset_train_path,
        dataset_train_image_path,
        dataset_train_mask_path,
        dataset_test_path,
        dataset_test_image_path,
        dataset_test_mask_path,
    ]

    for dir in dirs:
        if not dir.is_dir():
            dir.mkdir()
    
    for i in range(N_TEXTURES):

        texture_path = dataset_test_image_path / str(i)
        if not texture_path.is_dir():
            texture_path.mkdir()

        mask_path = dataset_test_mask_path / str(i)
        if not mask_path.is_dir():
            mask_path.mkdir()

    train_images = 2_000
    print(f"Generating {train_images} training images...")
    for i in tqdm(range(train_images)):
        cloth_sim.reset_simulation()
        cloth_sim.simulate()
        render.render_texture_and_save(f"{dataset_train_image_path}/{i}.png")
        render.render_mask_and_save(f"{dataset_train_mask_path}/{i}.png")

    test_images = 10
    print(f"Generating {test_images} test images...")
    for texture in tqdm(range(N_TEXTURES)):
        for i in tqdm(range(test_images), leave=False):
            cloth_sim.reset_simulation()
            cloth_sim.simulate()
            render.render_texture_and_save(f"{dataset_test_image_path}/{texture}/{i}.png", texture_index=texture)
            render.render_mask_and_save(f"{dataset_test_mask_path}/{texture}/{i}.png")
    
    print("Generating points...")
    generate_points()


if __name__ == "__main__":
    ti.init(ti.vulkan, log_level=ti.WARN)
    main()
