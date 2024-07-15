from torchvision.transforms.functional import resize as Tresize
import torch
from PIL import Image

from utils.networkutils import loadmodelweights, init_net
from model.discriminator import VOTEGAN
from model.saliencymodel import EMLNET
import numpy as np
from utils.datautils import get_transform
from dataloader.anydataset import AnyDataset

import argparse

class Options:
    def __init__(self):
        return


def calculate_realism(args, img1, img2, mask):
    #crop
    before = Image.open(img1).convert("RGB")
    after = Image.open(img2).convert("RGB")
    mask  = Image.open(mask).convert("RGB")
    options = {"load_size": 384, "crop_size": 384, "preprocess": "resize", "no_flip": True}
    setattr(args, "gpu_ids", [0])

    before = next(iter(torch.utils.data.DataLoader(
        AnyDataset(args, before, mask),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True
    )))
    after = next(iter(torch.utils.data.DataLoader(
        AnyDataset(args, after, mask),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True
    )))
    # initialize
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = init_net(VOTEGAN(args), [0])
    loadmodelweights(net, "./bestmodels/realismnet.pth", device)
    net.eval()

    net_eml = init_net(EMLNET(args), [0])

    before_img = before["rgb"].to(device)
    after_img = after["rgb"].to(device)
    mask_img = before["mask"].to(device)

    before_imgs = torch.cat((before_img, mask_img), 1).to(device)
    after_imgs = torch.cat((after_img, mask_img), 1).to(device)
    # realism
    before_realism = net(before_imgs).squeeze(1)
    after_realism = net(after_imgs).squeeze(1)
    realism_change = before_realism - after_realism


    # saliency
    before_saliency =Tresize(net_eml(before_img), (options["crop_size"], options["crop_size"]))
    after_saliency = Tresize(net_eml(after_img), (options["crop_size"], options["crop_size"]))
    saliency_change = (after_saliency-before_saliency) / (before_saliency + 1e-8)
    saliency_change = torch.sum(saliency_change * mask_img, axis=[1, 2, 3]) / torch.sum(mask_img, dim=[1, 2, 3])
    saliency_change = torch.clip(saliency_change, -1, 1)

    return realism_change.detach().cpu().numpy(), saliency_change.detach().cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument('--crop_size', type=int, default=384)
    parser.add_argument('--load_size', type=int, default=384)

    args = parser.parse_args()
    realism_change, saliency_change = calculate_realism(args, args.before, args.after, args.mask)
    print(f"Realism change:\t\t {realism_change.item()}\nSaliency change:\t {saliency_change.item()}")