import torch
import os 
import cv2
import numpy as np
from argumentsparser import args
import random
import copy


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

    # amplification trainer
    args.init_parameters_weights = args.init_amplify_weights
    args.result_for_decrease = 0
    amplification_args = copy.deepcopy(args)
    amplification_trainer = EditNetTrainer(amplification_args)

    # attenuation trainer
    args.init_parameters_weights = args.init_attenuate_weights
    args.result_for_decrease = 1
    attenuation_args = copy.deepcopy(args)
    attenuation_trainer = EditNetTrainer(attenuation_args)

    attenuation_trainer.setEval()
    amplification_trainer.setEval()

    pick_strategy_list = ['others', 'best_realism' , 'best_saliency']
    for pick_strategy in pick_strategy_list:
        os.makedirs(os.path.join(result_root, 'picked_{}'.format(pick_strategy)), exist_ok=True)


    for episode,data in enumerate(dataloader_val):
        mask_path = data['path'][0]
        image_name =  mask_path.split('/')[-1].split('.')[0]+'.jpg'

        print('({}/{})'.format(episode+1, len(dataloader_val)), '----->', image_name)

        if image_name.endswith("amplification"):
            trainer = amplification_trainer
        elif image_name.endswith("attenuation"):
            trainer = attenuation_trainer
        else:
            trainer = None

        trainer.setinput_hr(data)

        sal_list = []
        realism_list = []
        result_list = []
        with torch.inference_mode():
            for result in trainer.forward_allperm_hr():
                sal_list.append(result[2])
                realism_list.append(result[1])
                edited = (result[6][0,].transpose(1,2,0)[:,:,::-1] * 255).astype('uint8')
                result_list.append(edited.copy())

        sal_list = [np.asarray(item).item() for item in sal_list]
        realism_list = [np.asarray(item).item() for item in realism_list]

        # Do the pick
        picked_list = []
        for pick_strategy in pick_strategy_list:
            if pick_strategy == 'random':
                picked_ind = random.randint(0, len(sal_list)-1)
            elif pick_strategy == 'best_realism':
                picked_ind = np.argmin(realism_list)
            elif pick_strategy == 'best_saliency':
                if args.result_for_decrease == 1:
                    picked_ind = np.argmin(sal_list)
                else:
                    picked_ind = np.argmax(sal_list)

            picked_list.append(picked_ind)
            # save picked result
            picked = result_list[picked_ind]
            picked_name = os.path.join('picked_{}'.format(pick_strategy),image_name) 
            cv2.imwrite(os.path.join(result_root, picked_name), picked)

    
                




