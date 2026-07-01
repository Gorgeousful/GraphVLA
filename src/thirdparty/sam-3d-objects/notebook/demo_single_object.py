# %%
# Copyright (c) Meta Platforms, Inc. and affiliates.

# %% [markdown]
# ## 1. Imports and Model Loading

# %%
import os
import imageio
import uuid
from IPython.display import Image as ImageDisplay
from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer

# %%
PATH = os.getcwd()
TAG = "hf"
config_path = f"/data0/luokang/dataset/luokang/ckpts/sam-3d-objects/checkpoints/pipeline.yaml"
inference = Inference(config_path, compile=False)

# %% [markdown]
# ## 2. Load input image to lift to 3D (single object)

# %%
IMAGE_PATH = f"{PATH}/images/shutterstock_stylish_kidsroom_1640806567/image.png"
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))

image = load_image(IMAGE_PATH)
mask = load_single_mask(os.path.dirname(IMAGE_PATH), index=14)
display_image(image, masks=[mask])

# %% [markdown]
# ## 3. Generate Gaussian Splat

# %%
# run model
output = inference(image, mask, seed=42)

# export gaussian splat (as point cloud)
output["gs"].save_ply(f"{PATH}/gaussians/single/{IMAGE_NAME}.ply")

# %% [markdown]
# ## 4. Visualize Gaussian Splat
# ### a. Animated Gif

# %%
# render gaussian splat
scene_gs = make_scene(output)
scene_gs = ready_gaussian_for_video_rendering(scene_gs)

video = render_video(
    scene_gs,
    r=1,
    fov=60,
    pitch_deg=15,
    yaw_start_deg=-45,
    resolution=512,
)["color"]

# save video as gif
imageio.mimsave(
    os.path.join(f"{PATH}/gaussians/single/{IMAGE_NAME}.gif"),
    video,
    format="GIF",
    duration=1000 / 30,  # default assuming 30fps from the input MP4
    loop=0,  # 0 means loop indefinitely
)

# notebook display
ImageDisplay(url=f"gaussians/single/{IMAGE_NAME}.gif?cache_invalidator={uuid.uuid4()}")

# %% [markdown]
# ### b. Interactive Visualizer

# %%
# might take a while to load (black screen)
interactive_visualizer(f"{PATH}/gaussians/single/{IMAGE_NAME}.ply")


