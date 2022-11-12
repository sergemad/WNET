import os, shutil
from config import Config
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

config = Config()


def save_progress_image(autoencoder, progress_images, progress_expected, epoch):
    if not torch.cuda.is_available():
        segmentations, reconstructions = autoencoder(progress_images)
    else:
        segmentations, reconstructions = autoencoder(progress_images.cuda())

    f, axes = plt.subplots(4, config.val_batch_size, figsize=(8,8))
    for i in range(config.val_batch_size):
        segmentation = segmentations[i]
        pixels = torch.argmax(segmentation, axis=0).float() / config.k # to [0,1]

        axes[0, i].imshow(progress_images[i].permute(1, 2, 0))
        axes[1, i].imshow(pixels.detach().cpu())
        axes[2, i].imshow(reconstructions[i].detach().cpu().permute(1, 2, 0))
        if config.variationalTranslation:
            axes[3, i].imshow(progress_expected[i].detach().cpu().permute(1, 2, 0))
    plt.savefig(os.path.join(config.segmentationProgressDir, str(epoch)+".png"))
    plt.close(f)