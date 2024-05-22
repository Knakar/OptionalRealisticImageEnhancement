## structure of the project
### before(forked code)
project_root/
- mask_inputs/
    - mask1.jpg
    - image2.jpg
- rgb_inputs/
    - image1.jpg
    - image2.jpg
- results/
    - mydataset/
        - amplification/
            ...
        - attenuation/
            ...

### after(proposal)
project_root/
- mask_inputs/
    - image1/
        - mask1_attenuation.jpg
        - mask2_amplification.jpg
        - \*(attenuation|amplification).jpg
    - image2/
        - mask1_attenuation
        - mask2_amplification
        - \*(attenuation|amplification).jpg
- rgb_inputs/
    - image1.jpg
    - image2.jpg
- results/
    - mydataset/
        - amplification/
            ...
        - attenuation/
            ...

## how to run
first, Download our model weights from [here](https://drive.google.com/file/d/1NUN9xmD3p8G7n-HpD03UY9LHEF6J82-Q/view?usp=drive_link) and place them inside `"./bestmodels/"` folder.

```bash
rgb_root="" # path to rgb_inputs
mask_root="" # path to mask_inputs
result_root="././result/mydataset" # path to results
```

```bash
python test.py --mask_root "$mask_root" --rgb_root "$rgb_root" --result_path "$result_path" \
    --init_attenuate_weights "bestmodels/editnet_attenuate.pth" \
    --init_amplify-weights "bestmodels/editnet_amplify.pth" \
    --result_for_decrease 1 --batch_size 1
```

## todo
- [] load both weights(attention, amplification)
    - argumentparser.py:: argument name: attenuate_weight, amplify_weight
- [] update image loading
    - [] change timing of initial weight loading
        - [] change timing of dataloder
    - [] load image from rgb_inputs
    - [] load image from mask_inputs
- [] detect mask inputs attenuation or amplification
- [] edit(attenuate or amplify) the image from mask inputs
    - use recursive function
        - def edit_image(images, masks?):
            // some code
            if(masks is not None):
                // some code
                edit_image(images, masks):
                    ...

- Adapt the mask to each (amplify|attenuate) image.
    - $(Total number of generated images)=(input mask)^{(batch size)}$

## Citation
```

```