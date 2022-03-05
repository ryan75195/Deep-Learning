import time

import numpy as np
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")


def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

    #Computes the psnr for a batch of images
def compute_psnr( fake_imgs, real_imgs):
          batch_size = fake_imgs.shape[0]
          total_psnr_batch = 0
          # Loop through each pairs of (fake_img, real_img) in the batch
          for fake_img, real_img in zip(fake_imgs, real_imgs):
              mse = np.sum((fake_img - real_img)**2)
              mse /= float(fake_img.shape[0] * fake_img.shape[1] * fake_img.shape[2]) # Divide our sum of squares by the total number of pixels in the image (256 * 256 * 3)
              psnr = 10 * np.log10((1.0 ** 2) / mse) # Should it be 1.0 or 255?
              total_psnr_batch += psnr

          # Divide total psnr obtained by batch size and add it to total_psnr
          avg_psnr_batch = total_psnr_batch / batch_size 
          # print('Avg psnr of an image from this batch: {:.4f} dB'.format(avg_psnr_batch))
          return avg_psnr_batch

def compute_ssim( fake_imgs, real_imgs):
          batch_size = fake_imgs.shape[0]
          total_ssim_batch = 0
          for fake_img, real_img in zip(fake_imgs, real_imgs):

              # Plot fake and real image
              # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
              #       sharex=True, sharey=True)
              # ax = axes.ravel()
              # ax[0].imshow(fake_img)
              # ax[1].imshow(real_img)
              # plt.show()
              
              # Calculate SSIM
              ssimVal = ssim(real_img, fake_img, data_ramge = real_img.max() - fake_img.min(), multichannel=True) # data_range is 1.0
              # print('ssim for this image is: ', ssimVal)
              total_ssim_batch += ssimVal

          avg_ssim_batch = total_ssim_batch / batch_size
          return avg_ssim_batch
                
def compute_acc( net_G,dataloader):
          total_psnr = 0
          total_ssim = 0
          net_G.eval()
          with torch.no_grad():
              for data in dataloader:
                  L, ab = data['L'].to(device), data['ab'].to(device)
                  fake_color = net_G(L)

                  fake_imgs = lab_to_rgb(L,fake_color.detach()) # produces np array of shape (16, 256, 256, 3)
                  real_imgs = lab_to_rgb(L,ab) # produces np array of shape (16, 256, 256, 3)

                  avg_psnr_batch = compute_psnr(fake_imgs, real_imgs)
                  avg_ssim_batch = compute_ssim(fake_imgs, real_imgs)
                  total_psnr += avg_psnr_batch
                  total_ssim += avg_ssim_batch
              
              #print("===> Avg. PSNR: {:.4f} dB".format(total_psnr / len(dataloader))) # length of val_dl is 2000/16 = 125
              #print("===> Avg. SSIM: {:.4f} dB".format(total_ssim / len(dataloader))) # length of val_dl is 2000/16 = 125
          net_G.train()
          return total_psnr, total_ssim


