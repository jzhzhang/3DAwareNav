import numpy as np
import matplotlib.pyplot as plt
import torch
from constants import color_palette_array
from skimage import io

def plot_save_output(obs, output):
    
    boxes = output["boxes"] 
    labels = output["labels"]
    scores = output["scores"].detach().numpy()
    masks = output["masks"].permute(1,2,3,0).detach().numpy()

    if (scores.shape[0]==0):
        print("no prediction, continue")
        return
    plt.figure(figsize=(14*2, 14))    

    plt.subplot(1 ,2 ,1)
    plt.title("Input img")
    plt.axis("off")
    plt.imshow(imgs.permute(1,2,0).detach().numpy().astype(np.uint8))

    plt.subplot(1 ,2 ,2)
    plt.title("Mask"+str(scores[0]))
    plt.axis("off")
    plt.imshow(masks[0,:,:,0])


    plt.savefig(path)



def save_semantic(output, sem_seg):
    # non_zero_row = torch.abs(sem_seg).sum(dim=1) > 0

    sem_index = np.argmax(sem_seg, axis=2)
    sem_index = sem_index + 5
    sem_index[sem_index==11] = 1
    sem_seg = color_palette_array[sem_index]

    io.imsave(output, sem_seg)

