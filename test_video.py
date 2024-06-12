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
    #TODO cahange to directory namae detaction form file name
    if mask_kind.endswith("amplification"):
        trainer = amplification_trainer
    elif mask_name.endswith("attenuation"):
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

    video_names = [path for path in os.listdir(args.rgb_root) if os.path.isdir(path)]

    realism_net = init_net(VOTEGAN(args), args.gpu_ids)
    predict_device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    loadmodelweights(realism_net, 'bestmodels/realismnet.pth', predict_device)
    realism_net.eval()

    for i, video_title in enumerate(video_names):
        video_root_path = os.path.join(args.rgb_root, video_title)
        video_mask_path = os.path.join(args.mask_root, video_title)
        frame_paths = sorted([path for path in os.listdir(video_root_path) if path.endswith('.jpg')])

        mask_kind = sorted([path for path in os.listdir(video_mask_path) if os.path.isdir(path)])
        mask_paths = []
        for kind in mask_kind:
            mask_paths.append([path for path in os.listdir(os.path.join(video_mask_path, kind)) if path.endswith('.png')])
        mask_paths_per_frame = np.array(mask_paths).T.tolist()

        for j, frame in enumerate(frame_paths):
            # calculate the before realism score
            image_path = os.path.join(video_root_path, frame)
            mask_paths = mask_paths_per_frame[j]

            image: Image = Image.open(image_path).convert("RGB")
            # get the overlapping mask
            overlapping_mask = np.maximum.reduce([np.array(Image.open(mask_path)) for mask_path in mask_paths])
            # generate the modulated image
            results = modulate_image(image, mask_paths)
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
            before_realism_score = realism_net((rgb, mask), 1).squeeze(1)
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
                after_realism_score = realism_net((rgb, mask), 1).squeeze(1)
                realism_score = after_realism_score - before_realism_score
                realisms.append(realism_score)

            # pick the best result
            picked_idx = np.argmin(realisms)
            picked = results[picked_idx]



    video_param = None
    for frame,data in enumerate(dataloader_val):
        mask_path = data['path'][0]
        image_name =  mask_path.split('/')[-1].split('.')[0]+'.jpg'

        print('({}/{})'.format(frame+1, len(dataloader_val)), '----->', image_name)
        
        trainer.setinput_hr(data)

        sal_list = []
        realism_list = []
        result_list = []
        param_list = []
        with torch.inference_mode():
            for result in trainer.forward_allperm_hr(video_param):
                sal_list.append(result[2])
                realism_list.append(result[1])
                edited = (result[6][0,].transpose(1,2,0)[:,:,::-1] * 255).astype('uint8')
                result_list.append(edited.copy())
                param_list.append(result[9])

        if video_param is None:
            video_param = param_list[0]
            print('Video param selected as params from the first frame')
            

        sal_list = [np.asscalar(item) for item in sal_list]
        realism_list = [np.asscalar(item) for item in realism_list]

        # Do the pick
        picked_ind = 0
        pick_strategy = 'first'
        # save picked result
        picked = result_list[picked_ind]
        picked_name = os.path.join('picked_{}'.format(pick_strategy),image_name) 
        cv2.imwrite(os.path.join(result_root, picked_name), picked)

    
                




