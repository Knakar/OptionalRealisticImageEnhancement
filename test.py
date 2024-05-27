import torch
import os
import cv2
import numpy as np
from argumentsparser import args
import random
import copy
from PIL import Image


from model.editnettrainer import EditNetTrainer
from dataloader.anydataset import AnyDataset


def modulate_image(image: Image, masks, realism, saliency):
    """
    Modulate the image with masks, and calculate realisms, and saliencies.

    Args:
        image (Image): The image.
        masks (List[np.ndarray]): The masks.
        realism (float): The realism.
        saliency (float): The saliency.

    Returns:
        List[np.ndarray]: The modulated images.
        List[float]: The modulated realisms.
        List[float]: The modulated saliencies.
    """
    mask_path = masks.pop()
    mask_name = mask_path.split('/')[-1]
    mask = Image.open(mask_path).convert("RGB")
    datasets = AnyDataset(args, image, mask)
    dataloader_val = torch.utils.data.DataLoader(
        datasets,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    print(f'(masks:\t{len(masks)}/{len(mask_paths)}\timages:\t{episode}/{image_paths})', '----->', mask_name)
    if mask_name.endswith("amplification.jpg"):
        trainer = amplification_trainer
    elif mask_name.endswith("attenuation.jpg"):
        trainer = attenuation_trainer
    else:
        trainer = None
    trainer.setinput_hr(dataloader_val)
    temp_saliencies = []
    temp_realisms = []
    temp_images = []
    with torch.inference_mode():
        for result in trainer.forward_allperm_hr():
            saliency *= result[2]; temp_saliencies.append(saliency.item())
            realism *= result[1]; temp_realisms.append(realism.item())

            edited = (result[6][0,].transpose(1,2,0)[:,:,::-1] * 255).astype('uint8')
            temp_images.append(edited.copy())
    del datasets, dataloader_val
    results = []
    saliencies = []
    realisms = []
    for i, img in enumerate(temp_images):
        if masks:
            ret_image, ret_saliency, ret_realism = modulate_image(img, copy.deepcopy(masks), temp_realisms[i], temp_saliencies[i])
            results.extend(ret_image)
            saliencies.extend(ret_saliency)
            realisms.extend(ret_realism)
        else:
            results.append(img)
            saliencies.append(temp_saliencies[i])
            realisms.append(temp_realisms[i])

    return results, saliencies, realisms

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

     # get images path
    image_paths = sorted([path for path in os.listdir(args.rgb_root) if path.endswith('.jpg')])

    result_root = args.result_path
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

    pick_strategy_list = ['ramdom', 'best_realism' , 'best_saliency']
    for pick_strategy in pick_strategy_list:
        os.makedirs(os.path.join(result_root, 'picked_{}'.format(pick_strategy)), exist_ok=True)

    for episode, image_path in enumerate(image_paths):
        image_name = image_path.split('.')[0]+'.jpg'
        mask_root = os.path.join(args.mask_root, image_path.split('.')[0])
        image: Image = Image.open(image_path).convert("RGB")
        mask_paths = [os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.endswith('.jpg')]
        results, saliencies, realisms = modulate_image(image, mask_paths, 1.0, 1.0)

        # Do the pick
        picked_list = []
        for pick_strategy in pick_strategy_list:
            if pick_strategy == 'random':
                picked_idx = random.randint(0, len(saliencies)-1)
            elif pick_strategy == 'best_realism':
                picked_idx = np.argmin(realisms)
            elif pick_strategy == 'best_saliency':
                if args.result_for_decrease == 1:
                    picked_idx = np.argmin(saliencies)
                else:
                    picked_idx = np.argmax(saliencies)
            else:
                picked_idx = None

            picked_list.append(picked_idx)
            # save picked result
            picked = results[picked_idx]
            picked_name = os.path.join('picked_{}'.format(pick_strategy),image_name)
            cv2.imwrite(os.path.join(result_root, picked_name), picked)







