import torch
import os 
import cv2
import numpy as np
from argumentsparser import args
import random


from model.editnettrainer import EditNetTrainer
from dataloader.anydataset import AnyDataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

str_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        args.gpu_ids.append(id)
if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])

if __name__ == '__main__':
    dataset_val = AnyDataset(args)

    dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    drop_last=True)
    
    direction_str = 'attenuation' if args.result_for_decrease else 'amplification'
    result_root = os.path.join(args.result_path, direction_str)
    os.makedirs(result_root, exist_ok=True)
    
    trainer = EditNetTrainer(args)

    trainer.setEval()

    for episode,data in enumerate(dataloader_val):
        mask_path = data['path'][0]
        image_name =  mask_path.split('/')[-1].split('.')[0]+'.jpg'

        print('({}/{})'.format(episode+1, len(dataloader_val)), '----->', image_name)
        
        trainer.setinput_hr(data)

        result_list = []
        with torch.inference_mode():
            for result in trainer.forward_allperm_hr():
                edited = (result[6][0,].transpose(1,2,0)[:,:,::-1] * 255).astype('uint8')
                result_list.append(edited.copy())

        for i, result in enumerate(result_list):
            cv2.imwrite(os.path.join(result_root, 'result_{}_{}'.format(i,image_name)), result)
