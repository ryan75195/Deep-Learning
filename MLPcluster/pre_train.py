from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torchvision.models.vgg import vgg11
from fastai.vision.models.unet import DynamicUnet
import torchvision.models as models
import torch
from utils import *
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_res_unet(resnet,n_input=1, n_output=2, size=256):
   # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resnet == 18:
        body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    elif resnet == 34:
        body = create_body(resnet34, pretrained=True, n_in=n_input, cut=-2)
    else:
        print("invalid resnet type")
        exit()
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)

    return net_G

def pretrain_generator(net_G, train_dl,val_dl, opt, criterion, epochs,resnet, lr,weight_decay):
    for e in range(epochs):
        train_loss_meter = AverageMeter()
        val_loss_meter = AverageMeter()
        i = 0 
        for data in train_dl:
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step() 
            train_loss_meter.update(loss.item(), L.size(0))

        net_G.eval()
        with torch.no_grad():
            for data in val_dl:
                L, ab = data['L'].to(device), data['ab'].to(device)
                preds = net_G(L)
                loss = criterion(preds.detach(), ab)
                val_loss_meter.update(loss.item(), L.size(0))
            
                 
        total_psnr, total_ssim = compute_acc(net_G,val_dl)
        net_G.train()
        psnr = total_psnr / len(val_dl)
        ssim = total_ssim / len(val_dl)	

        print(f"Epoch {e + 1}/{epochs}")
        print(f"Train Loss: {train_loss_meter.avg:.5f}")
        print(f"Val Loss: {val_loss_meter.avg:.5f}")
        print(f"PSNR: {psnr:.4f} dB")
        print(f"SSIM: {ssim:.4f} dB") 
        with open(f"rs{resnet}-{lr}-{weight_decay}-experiment", "a+") as file:
            file.write(f"{e+1}, Train loss,{train_loss_meter.avg:.5f}\n") 
            file.write(f"{e+1},Val loss,{val_loss_meter.avg:.5f}\n")
            file.write(f"{e+1},PSNR, {psnr:.4f}\n")
            file.write(f"{e+1},SSIM, {ssim:.4f}\n")

        i += 1
        if i % 10 == 0:
            name = "res"+str(resnet)+"epoch"+str(i)+"-unet.pt"
            torch.save(net_G.state_dict(), name)
