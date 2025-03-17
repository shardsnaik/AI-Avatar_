import torch
import numpy as np
import imageio
from skimage.transform import resize
import sys
sys.path.append("first-order-model")  # Adjust this path if needed
from demo import load_checkpoints, make_animation

# Load Model
generator, kp_detector = load_checkpoints(config_path='first-order-model\\config\\vox-256.yaml',checkpoint_path='checkpoints/vox.pth')



# Load Image (Avatar or Face)
source_image = imageio.imread('image_src\\freepik__the-style-is-candid-image-photography-with-natural__24771.jpeg')
source_image = resize(source_image, (256, 256))[..., :3]  # Resize for model

# Load Driving Video (Face Movement)
driving_video = imageio.mimread('image_src\\avatar_vedio.mp4')
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

# Generate Animation
predictions = make_animation(source_image, driving_video, generator, kp_detector)

# Save Result
imageio.mimsave('output.mp4', predictions)