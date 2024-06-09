import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
from argumentsparser import args
import random
import copy
from PIL import Image
import gc

from model.editnettrainer import EditNetTrainer
from dataloader.anydataset import AnyDataset, ResultDataset

from model.discriminator import VOTEGAN
from utils.networkutils import init_net, loadmodelweights

import torch


def modulate_image(image: Image, masks):
    """
    Modulate the image with masks, and calculate realisms, and saliencies.

    Args:
        image (Image): The image.
        masks (List[np.ndarray]): The masks.

    Returns:
        List[np.ndarray]: The modulated images.
    """
    mask_path = masks.pop()
    mask_name = mask_path.split('/')[-1]
    mask = Image.open(mask_path)
    datasets = AnyDataset(args, image, mask)
    dataloader_val = torch.utils.data.DataLoader(
        datasets,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    data = next(iter(dataloader_val))
    print(f'(masks:\t{len(mask_paths)-len(masks)}/{len(mask_paths)+1}\timages:\t{episode+1}/{len(image_names)}) {mask_name}', '----->', image_name)
    if mask_name.endswith("amplification.jpg"):
        trainer = amplification_trainer
    elif mask_name.endswith("attenuation.jpg"):
        trainer = attenuation_trainer
    else:
        trainer = None
    trainer.setinput_hr(data)
    temp_images = []
    with torch.inference_mode():
        for result in trainer.forward_allperm_hr():
            edited = (result[6][0,].transpose(1,2,0) * 255).astype('uint8')
            temp_images.append(edited.copy())
    del datasets, dataloader_val, data
    gc.collect()
    torch.cuda.empty_cache()
    results = []
    for i, img in enumerate(temp_images):
        if masks:
            ret_image = modulate_image(Image.fromarray(img, "RGB"), copy.deepcopy(masks))
            results.extend(ret_image)
        else:
            results.append(img)

    return results

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= ", ".join(map(str, args.gpu_ids))

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

     # get images names
    image_names = sorted([path for path in os.listdir(args.rgb_root) if path.endswith('.jpg')])

    result_root = args.result_path
    os.makedirs(result_root, exist_ok=True)

    # amplification trainer
    setattr(args, "init_parameternet_weights", args.init_amplify_weights)
    args.result_for_decrease = 0
    amplification_args = copy.deepcopy(args)
    amplification_trainer = EditNetTrainer(amplification_args)

    # attenuation trainer
    setattr(args, "init_parameter_weights", args.init_attenuate_weights)
    args.init_parameters_weights = args.init_attenuate_weights
    args.result_for_decrease = 1
    attenuation_args = copy.deepcopy(args)
    attenuation_trainer = EditNetTrainer(attenuation_args)

    attenuation_trainer.setEval()
    amplification_trainer.setEval()

    pick_strategy_list = ['random', 'best_realism']
    for pick_strategy in pick_strategy_list:
        os.makedirs(os.path.join(result_root, 'picked_{}'.format(pick_strategy)), exist_ok=True)

    # initialize predict model
    realism_net = init_net(VOTEGAN(args), args.gpu_ids)

    predict_device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    loadmodelweights(realism_net, 'bestmodels/realismnet.pth', predict_device)
    realism_net.eval()

    for episode, image_name in enumerate(image_names):
        image_path = os.path.join(args.rgb_root,image_name.split('.')[0]+'.jpg')
        mask_root = os.path.join(args.mask_root, image_name.split('.')[0])
        image: Image = Image.open(image_path).convert("RGB")
        mask_paths = [os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.endswith('.jpg')]
        # get the overlapping mask
        overlapping_mask = np.maximum.reduce([np.array(Image.open(mask_path)) for mask_path in mask_paths])
        # generate the modulated image
        results = modulate_image(image, mask_paths)
        realisms = []
        # before realism score
        dataset_before_editing = AnyDataset(
            args,
            image,
            Image.fromarray(overlapping_mask)
        )
        dataset_before_editing_val = torch.utils.data.DataLoader(
            dataset_before_editing,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=True
        )
        data_before_editing = next(iter(dataset_before_editing_val))

        ## settings for predict realism
        rgb = data_before_editing['rgb'].to(predict_device)
        mask = data_before_editing['mask'].to(predict_device)
        category = data_before_editing['category'].to(predict_device)
        ishuman =  (category == 1).float()
        input_data = torch.cat((rgb, mask), 1).to(predict_device)

        before_D_value = realism_net(torch.cat(tuple((rgb, mask)), 1)).squeeze(1).to(predict_device)
        result_dataset = ResultDataset(args, results, Image.fromarray(overlapping_mask))
        result_dataloader = torch.utils.data.DataLoader(
            result_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=True
        )
        # predict the final realism score
        for i, data_after_editing in enumerate(result_dataloader):
            print(f"Evaluate the realism score ({i+1}/{len(results)})\t ----------->", end="\t")
            result = data_after_editing['rgb'].to(predict_device)
            # calculate the realism score
            D_value = realism_net(torch.cat(tuple((result, mask)), 1)).squeeze(1).to(predict_device)
            realism_change = before_D_value - D_value
            #realism_component_human = (1+args.human_weight_gan * F.relu(realism_change))
            #realism_component_other = (1+F.relu(realism_change - args.beta_r))
            #realism_loss = ishuman.squeeze(1) * realism_component_human + (1-ishuman.squeeze(1)) * realism_component_other
            realisms.append(realism_change.item())
            del result, D_value, realism_change
            gc.collect()
            torch.cuda.empty_cache()
            print(realisms[-1])

        # Do the pick
        picked_list = []
        for pick_strategy in pick_strategy_list:
            if pick_strategy == 'random':
                picked_idx = random.randint(0, len(results)-1)
            elif pick_strategy == 'best_realism':
                picked_idx = np.argmin(realisms)

            picked_list.append(picked_idx)
            # save picked result
            picked = results[picked_idx]
            picked = cv2.cvtColor(picked, cv2.COLOR_RGB2BGR)
            picked_name = os.path.join('picked_{}'.format(pick_strategy), f"{image_name.split('.')[0]}_{realisms[picked_idx]}.jpg")
            cv2.imwrite(os.path.join(result_root, picked_name), picked)

        #save all results
        for idx, result in enumerate(results):
            result_name = os.path.join('all', image_name.split('.')[0] + '_{}.jpg'.format(idx))
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(result_root, result_name), result)







