import torch
import os
import cv2
import numpy as np
from argumentsparser import args
import random
import copy
from PIL import Image
from anytree import Node, findall, Walker, RenderTree
import gc

from model.editnettrainer import EditNetTrainer
from dataloader.anydataset import AnyDataset


def modulate_image(image: Image, masks, realism, saliency, realism_history:Node=None, saliency_history:Node=None):
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
    if realism_history is None or saliency_history is None:
        realism_history = Node("root")
        saliency_history = Node("root")

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
    temp_saliencies_history = []
    temp_realisms_history = []
    temp_images = []
    with torch.inference_mode():
        for result in trainer.forward_allperm_hr():
            generated_saliency = 1 - (result[2].item()) if args.result_for_decrease else result[2].item()
            temp_saliencies.append(saliency + generated_saliency)
            generated_realism = result[1].item()
            temp_realisms.append(realism + generated_realism)
            temp_saliencies_history.append(Node(generated_saliency, parent=saliency_history))
            temp_realisms_history.append(Node(generated_realism, parent=realism_history))
            edited = (result[6][0,].transpose(1,2,0) * 255).astype('uint8')
            temp_images.append(edited.copy())
    del datasets, dataloader_val, data
    gc.collect()
    torch.cuda.empty_cache()
    results = []
    saliencies = []
    realisms = []
    for i, img in enumerate(temp_images):
        if masks:
            ret_image, ret_saliency, ret_realism, _, _ = (
                modulate_image(Image.fromarray(img, "RGB"), copy.deepcopy(masks),
                                    temp_realisms[i], temp_saliencies[i], temp_saliencies_history[i], temp_realisms_history[i]))
            results.extend(ret_image)
            saliencies.extend(ret_saliency)
            realisms.extend(ret_realism)
        else:
            results.append(img)
            saliencies.append(temp_saliencies[i])
            realisms.append(temp_realisms[i])

    return results, saliencies, realisms, realism_history, saliency_history

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
        results, saliencies, realisms, saliency_history, realism_history = modulate_image(image, mask_paths, 0.0, 0.0)

        # search leaves
        saliency_leaves = list(findall(saliency_history, filter_=lambda node: not node.children))
        realism_leaves = list(findall(realism_history, filter_=lambda node: not node.children))

        # Do the pick
        picked_list = []
        for pick_strategy in pick_strategy_list:
            # Find the history of the evaluation score in the History Tree
            walker = Walker()

            if pick_strategy == 'random':
                picked_idx = random.randint(0, len(saliencies)-1)
                image_evaluation_history = "random"
            elif pick_strategy == 'best_realism':
                picked_idx = np.argmin(realisms)
                # saliency history
                _, saliency_history_path, path_down = walker.walk(saliency_history, saliency_leaves[picked_idx])
                saliency_history_path  = [saliency_history_path] + list(path_down)
                image_evaluation_history = "_".join([str(node.name) for node in saliency_history_path][1:])
            elif pick_strategy == 'best_saliency':
                picked_idx = np.argmax(saliencies)
                # realism history
                _, realism_history_path, path_down = walker.walk(realism_history, realism_leaves[picked_idx])
                realism_history_path = [realism_history_path] + list(path_down)
                image_evaluation_history = "_".join([str(node.name) for node in realism_history_path][1:])

            picked_list.append(picked_idx)
            # save picked result
            picked = results[picked_idx]
            picked = cv2.cvtColor(picked, cv2.COLOR_RGB2BGR)
            picked_name =  f"{image_name.split('.')[0]}_{image_evaluation_history}.jpg"
            cv2.imwrite(os.path.join(result_root,f"picked_{pick_strategy}", picked_name), picked)

        #save all results
        for idx, result in enumerate(results):
            saliency = saliencies[idx]
            realism = realisms[idx]

            result_name = os.path.join('all', image_name.split('.')[0] + '_{}.jpg'.format(idx))
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(result_root, result_name), result)







