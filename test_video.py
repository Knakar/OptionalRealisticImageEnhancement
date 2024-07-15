import torch
import os 
import cv2
import numpy as np
from argumentsparser import args
import random
import copy
from PIL import Image

from model.editnettrainer import EditNetTrainer
from model.discriminator import VOTEGAN
from utils.networkutils import init_net, loadmodelweights

from dataloader.anydataset import AnyDataset
from dataloader.resultdataset import ResultDataset



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
    mask_kind = mask_path.split('/')[-2]
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
    if mask_kind.endswith("amplification"):
        trainer = amplification_trainer
    elif mask_kind.endswith("attenuation"):
        trainer = attenuation_trainer
    else:
        trainer = None

    trainer.setinput_hr(data)
    temp_images = []
    temp_saliency = []
    with torch.inference_mode():
        for result in trainer.forward_allperm_hr():
            saliency = -(result[2].item())if args.result_for_decrease else result[2].item(); temp_saliency.append(saliency)
            edited = (result[6][0,].transpose(1,2,0) * 255).astype('uint8')
            temp_images.append(edited.copy())
    del datasets, dataloader_val, data
    torch.cuda.empty_cache()
    best_saliency_idx = np.argmax(temp_saliency)
    best_saliency = temp_images[best_saliency_idx]
    results_realism = []
    result_saliency = None
    saliency = 0
    for i, img in enumerate(temp_images):
        if masks:
            ret_images, ret_sal_image, ret_saliency = modulate_image(Image.fromarray(img, "RGB"), copy.deepcopy(masks))
            if i == best_saliency_idx:
                result_saliency = ret_sal_image
                saliency +=  ret_saliency
            results_realism.extend(ret_images)
        else:
            results_realism.append(img)
            result_saliency = best_saliency
            saliency += temp_saliency[best_saliency_idx]

    return results_realism, result_saliency, saliency

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

str_ids = args.gpu_ids.split(',')
args.gpu_ids = []
num_gpu = 0
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        args.gpu_ids.append(num_gpu)
        num_gpu+=1
if len(args.gpu_ids) > 0:
    print(args.gpu_ids[0])
    torch.cuda.set_device(args.gpu_ids[0])


if __name__ == '__main__':
    print("start")
    result_root = args.result_path
    os.makedirs(os.path.join(result_root), exist_ok=True)
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


    pick_strategy_list = ['first']
    for pick_strategy in pick_strategy_list:
        os.makedirs(os.path.join(result_root, 'picked_{}'.format(pick_strategy)), exist_ok=True)

    video_names = [path for path in os.listdir(args.rgb_root) if os.path.isdir(os.path.join(args.rgb_root, path))]

    realism_net = init_net(VOTEGAN(args), args.gpu_ids)
    predict_device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    loadmodelweights(realism_net, 'bestmodels/realismnet.pth', predict_device)
    realism_net.eval()

    for i, video_title in enumerate(video_names):
        video_root_path = os.path.join(args.rgb_root, video_title)
        video_mask_path = os.path.join(args.mask_root, video_title)
        frame_paths = sorted([path for path in os.listdir(video_root_path) if path.endswith('.jpg')])

        mask_kind = sorted([path for path in os.listdir(video_mask_path) if os.path.isdir(os.path.join(video_mask_path, path))])
        mask_paths = []
        for kind in mask_kind:
            mask_paths.append(sorted([os.path.join(video_mask_path, kind,path) for path in os.listdir(os.path.join(video_mask_path, kind)) if path.endswith('.jpg')]))
        mask_paths_per_frame = np.array(mask_paths).T.tolist()
        realism_sum = 0.0
        saliency_sum = 0.0
        # initialize the video writer
        output_codec = cv2.VideoWriter_fourcc(*'MJPG')
        output_writer_realism = cv2.VideoWriter(os.path.join(result_root, f"{video_title}_realism.avi"), output_codec, 5, Image.open(os.path.join(video_root_path, frame_paths[0])).size, True)
        output_writer_saliency = cv2.VideoWriter(os.path.join(result_root, f"{video_title}_saliency.avi"), output_codec, 5, Image.open(os.path.join(video_root_path, frame_paths[0])).size, True)
        for j, frame in enumerate(frame_paths):
            print(f"video: {video_title} frame: {j+1}/{len(frame_paths)}")
            # calculate the before realism score
            image_path = os.path.join(video_root_path, frame)
            mask_paths = mask_paths_per_frame[j]

            image: Image = Image.open(image_path).convert("RGB")
            # get the overlapping mask
            overlapping_mask = np.maximum.reduce([np.array(Image.open(mask_path)) for mask_path in mask_paths])
            # generate the modulated image
            results, saliency_result, saliency= modulate_image(image, mask_paths)
            torch.cuda.empty_cache()
            # predict before realism score
            data_before_editing = next(iter(torch.utils.data.DataLoader(
                AnyDataset(args, image, Image.fromarray(overlapping_mask)),
                batch_size=1,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=True)))
            rgb = data_before_editing['rgb'].to(predict_device)
            mask = data_before_editing['mask'].to(predict_device)
            before_realism_score = realism_net(torch.cat(tuple((rgb, mask)), 1)).squeeze(1)
            torch.cuda.empty_cache()
            result_dataloader = torch.utils.data.DataLoader(
                ResultDataset(args, results, Image.fromarray(overlapping_mask)),
                batch_size=1,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=True)

            realisms = []
            for result in result_dataloader:
                rgb = result['rgb'].to(predict_device)
                mask = result['mask'].to(predict_device)
                after_realism_score = realism_net(torch.cat(tuple((rgb, mask)), 1)).squeeze(1)
                torch.cuda.empty_cache()
                before_realism_score = before_realism_score.to(predict_device)
                after_realism_score = after_realism_score.to(predict_device)
                realism_score = (after_realism_score - before_realism_score).cpu().detach()
                realisms.append(realism_score)
            # pick the best result
            realisms = torch.cat(realisms).numpy()
            picked_realism_idx = np.argmin(realisms)
            realism_sum += realisms[picked_realism_idx]
            saliency_sum += saliency
            picked_realism = cv2.cvtColor(results[picked_realism_idx], cv2.COLOR_RGB2BGR)
            picked_saliency = cv2.cvtColor(saliency_result, cv2.COLOR_RGB2BGR)
            output_writer_realism.write(picked_realism)
            output_writer_saliency.write(picked_saliency)
            torch.cuda.empty_cache()
        realism_avg = realism_sum / len(frame_paths)
        saliency_avg = saliency_sum / len(frame_paths)
        print(f"Realism  avg:\t", realism_avg)
        print(f"Saliency avg:\t", saliency_avg)
