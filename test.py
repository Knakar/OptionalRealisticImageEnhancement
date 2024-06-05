import torch
import os
import cv2
import numpy as np
from argumentsparser import args
import random
import copy
from PIL import Image
import gc

from model.editnettrainer import EditNetTrainer
from dataloader.anydataset import AnyDataset


def modulate_image(image: Image, masks):
    """
    Modulate the image with masks, and calculate realism, and saliency.

    Args:
        image (Image): The image.
        masks (List[np.ndarray]): The masks.

    Returns:
        np.ndarray: The modulated images( the best realism).
        np.ndarray: The modulated images( the best saliency).
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
    temp_saliencies = []
    temp_realisms = []
    temp_images = []
    with torch.inference_mode():
        for result in trainer.forward_allperm_hr():
            saliency = 1-(result[2].item()) if args.result_for_decrease else result[2].item(); temp_saliencies.append(saliency)
            realism = result[1].item(); temp_realisms.append(realism)

            edited = (result[6][0,].transpose(1,2,0) * 255).astype('uint8')
            temp_images.append(edited.copy())
    del datasets, dataloader_val, data
    gc.collect()
    torch.cuda.empty_cache()

    best_realism_idx = np.argmax(temp_realisms)
    best_saliency_idx = np.argmax(temp_saliencies)

    best_realism = temp_images[best_realism_idx]
    best_saliency = temp_images[best_saliency_idx]

    realisms = [temp_realisms[best_realism_idx]]
    saliencies = [temp_saliencies[best_saliency_idx]]
    if masks:
        ret_realism_img, _, ret_realism, _ = modulate_image(Image.fromarray(best_realism, "RGB"), copy.deepcopy(masks))
        _, ret_saliency_img , _, ret_saliency = modulate_image(Image.fromarray(best_saliency, "RGB"), copy.deepcopy(masks))
        realisms.extend(ret_realism)
        saliencies.extend(ret_saliency)
    else:
        ret_realism_img = best_realism
        ret_saliency_img = best_saliency

    return ret_realism_img, ret_saliency_img, realisms, saliencies

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

    pick_strategy_list = ['random', 'best_realism' , 'best_saliency']
    for pick_strategy in pick_strategy_list:
        os.makedirs(os.path.join(result_root, 'picked_{}'.format(pick_strategy)), exist_ok=True)

    for episode, image_name in enumerate(image_names):
        image_path = os.path.join(args.rgb_root,image_name.split('.')[0]+'.jpg')
        mask_root = os.path.join(args.mask_root, image_name.split('.')[0])
        image: Image = Image.open(image_path).convert("RGB")
        mask_paths = [os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.endswith('.jpg')]
        best_realism_img, best_saliency_img, realisms, saliencies = modulate_image(image, mask_paths)
        print("realism:", "_".join(map(str, realisms)), "\nsaliency:", "_".join(map(str, saliencies)))
        best_realism_img = cv2.cvtColor(best_realism_img, cv2.COLOR_RGB2BGR)
        best_saliency_img= cv2.cvtColor(best_saliency_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(result_root, 'picked_best_realism', f'{image_name.split(".")[0]}_{"_".join(map(str, realisms))}.jpg'), best_realism_img)
        cv2.imwrite(os.path.join(result_root, 'picked_best_saliency', f'{image_name.split(".")[0]}_{"_".join(map(str, saliencies))}.jpg'), best_saliency_img)






