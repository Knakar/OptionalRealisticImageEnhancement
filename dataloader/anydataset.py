import copy

import torch
import torchvision.transforms as transforms

from utils.datautils import get_transform
from PIL import Image


class AnyDataset:
    def __init__(self, args, base: Image, mask: Image):
        self.args = args

        self.base:Image = copy.deepcopy(base)
        self.mask = mask

        opt = {}
        opt['load_size'] = args.load_size
        opt['crop_size'] = args.crop_size
        opt['preprocess'] = 'resize'
        opt['no_flip'] = True

        self.rgb_transform = get_transform(opt, grayscale=False)
        self.mask_transform = get_transform(opt, grayscale=True)
        self.org_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return 1

    def __getitem__(self, index_):

        rgb_img = self.base
        mask_img = self.mask

        mask_img = mask_img.resize(tuple(self.base.size),Image.Resampling.NEAREST)

        rgb = self.rgb_transform(rgb_img)
        mask = self.mask_transform(mask_img)

        rgb_org = self.org_transform(rgb_img)
        mask_org = self.org_transform(mask_img)
        
        category = torch.Tensor([-1])
        
        return {'rgb': rgb, 'mask': mask, 'category':category, 'rgb_org':rgb_org, 'mask_org':mask_org}
        

class ResultDataset:
    def __init__(self, args, inputs: Image, mask: Image):
        self.args = args
        self.rgbs =  inputs
        self.mask = mask
        opt = {"load_size": args.load_size, "crop_size": args.crop_size, "preprocess": "resize", "no_flip": True}
        self.rgb_transform = get_transform(opt, grayscale=False)
        self.mask_transform = get_transform(opt, grayscale=True)
        self.org_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, index_):
        rgb_img = Image.fromarray(self.rgbs[index_], "RGB")
        mask_img = self.mask
        mask_img = mask_img.resize(tuple(rgb_img.size), Image.Resampling.NEAREST)
        rgb = self.rgb_transform(rgb_img)
        mask = self.mask_transform(mask_img)
        rgb_org = self.org_transform(rgb_img)
        mask_org = self.org_transform(mask_img)
        category = torch.Tensor([-1])
        return {"rgb": rgb, "mask": mask, "category": category, "rgb_org": rgb_org, "mask_org": mask_org}