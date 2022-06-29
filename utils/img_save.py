import numpy as np
import matplotlib.pyplot as plt
import torch


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
