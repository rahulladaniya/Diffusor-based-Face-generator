import torch
from diffuisonmodel import obj_diffusionmodel
import matplotlib.pyplot as plt
import numpy as np


# Function to load the saved model
def load_model(model_path, img_size=64, timesteps=1000):
    checkpoint = torch.load(model_path)
    model = obj_diffusionmodel(img_size=checkpoint.get('img_size', img_size), 
                          timesteps=checkpoint.get('timesteps', timesteps))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

model=load_model("checkpoints/final_model.pt")

def show_images(images, title="Generated Images"):
    images = (images + 1) / 2  # Denormalize
    images = images.clamp(0, 1)
    images = images.cpu().numpy()

    plt.figure(figsize=(10, 10))
    for i in range(min(16, len(images))):
        plt.subplot(4, 4, i+1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

samples = model.sample(n_samples=8)
show_images(samples)