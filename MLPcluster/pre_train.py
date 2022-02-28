import glob

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torchvision.models.vgg import vgg13
from fastai.vision.models.unet import DynamicUnet
import torch
from utils import *
from tqdm.notebook import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G
    
def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    fileCounter = len(glob.glob1("data/accounts/Lossfiles", "*.txt"))
    for e in range(epochs):
        loss_meter = AverageMeter()
        i = 0 
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step() 
            loss_meter.update(loss.item(), L.size(0))

#        i += 1
#        if i % 10 == 0:
#            name = "res18"+str(i)+"-unet.pt"
#            torch.save(net_G.state_dict(), name)
###Modify here to write out result to file###
        with open(f"loss-{fileCounter}") as file:
            print(f"Epoch {e + 1}/{epochs}")
            print(f"L1 Loss: {loss_meter.avg:.5f}")
            file.write(f"{e + 1},{loss_meter.avg:.5f}\n")
