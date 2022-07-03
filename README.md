# Egocentric Scene Understanding via Multimodal Spatial Rectifier

This repository contains the source code for our paper:

**Egocentric Scene Understanding via Multimodal Spatial Rectifier**  
Tien Do, Khiem Vuong, and Hyun Soo Park  
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022  
[Project webpage](https://tien-d.github.io/egodepthnormal_cvpr22.html) 

![epick_supp_qualitative_small.png](media/epick_supp_qualitative_small.png)
<b>Qualitative results for depth, surface normal, and gravity 
prediction on EPIC-KITCHENS dataset.</b>

## Installation

[//]: # (:star2: Demo code and installation instructions will be available soon! :star2:)

To activate the docker environment, run the following command:

```
nvidia-docker run -it --rm --ipc=host -v /:/home nvcr.io/nvidia/pytorch:21.12-py3
```

where `/` is the directory in the local machine (in this case, the root folder), and `/home` is the reflection of that directory in the docker. 
This has also specified NVIDIA-Docker with PyTorch version 21.12 which is required to ensure the compatibility 
between the packages used in the code (at the time of submission).

Inside the docker, change the working directory to this repository: 
```
cd /home/PATH/TO/THIS/REPO/EgoDepthNormal
```

## Quick Inference
Please follow the below steps to extract depth and surface normals from some RGB images using our provided pre-trained model:

1) Make sure you have the following `.ckpt` files inside [`./checkpoints/`](./checkpoints) folder: 
`edina_depth_baseline.ckpt`, `edina_normal_baseline.ckpt`.
You can use this command to download these checkpoints:

    ```
    wget -O edina_depth_baseline.ckpt https://edina.s3.amazonaws.com/edina_depth_baseline.ckpt && mv edina_depth_baseline.ckpt ./checkpoints/
    
    wget -O edina_normal_baseline.ckpt https://edina.s3.amazonaws.com/edina_normal_baseline.ckpt && mv edina_normal_baseline.ckpt ./checkpoints/
    ```
   
2) Our demo RGB images are stored in [`demo_data/color`](./demo_data/color)
   
4) Run [`demo.sh`](./demo.sh) to extract the results in [`./demo_visualization/`](./demo_visualization).

    ```
    sh demo.sh
    ```

## Benchmark Evaluation
You can evaluate depth/surface normal predictions quantitatively and qualitatively on EDINA dataset using our provided pre-trained models. Make sure you have the corresponding depth/normal checkpoints inside [`./checkpoints/`](./checkpoints) folder and the dataset split (pickle file) inside [`./pickles/`](./pickles) folder. Please refer to [dataset](README_dataset.md) on how to download the pickle file. 


Run:
```
sh eval.sh
```
Specifically, inside the bash script, multiple arguments are needed, e.g. path to dataset/dataset pickle files, path to the pre-trained model, batch size, network architecture, test dataset, etc. Please refer to the actual code for the exact supported arguments options.

For instance, the following sample codeblock can be used to evaluate depth estimation on EDINA test set:

```
python main_depth.py --train 0 --model_type 'midas_v21' \
--test_usage 'edina_test' \
--checkpoint ./checkpoints/edina_depth_baseline.ckpt \
--dataset_pickle_file ./pickles/scannet_edina_camready_final_clean.pkl \
--batch_size 8 --skip_every_n_image_test 40 \
--data_root PATH/TO/EDINA/DATA \
--save_visualization ./eval_visualization/depth_results
```

## Egocentric Depth on everyday INdoor Activities (EDINA) Dataset

**:star2: EDINA data (train + test) set is now available to download! :star2:**

### Overview
EDINA is an egocentric dataset that comprises more than 500K synchronized RGBD frames and gravity directions. Each instance in the dataset is a triplet: RGB image, depths and surface normals, and 3D gravity direction.

![edina2.gif](media/edina2.gif)

**Please refer to [dataset](README_dataset.md) for more details, including downloading instructions and dataset organization.** 

## Citation
If you find our work to be useful in your research, please consider citing our paper:
```
@InProceedings{Do_2022_EgoSceneMSR,
    author     = {Do, Tien and Vuong, Khiem and Park, Hyun Soo},
    title      = {Egocentric Scene Understanding via Multimodal Spatial Rectifier},
    booktitle  = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month      = {June},
    year       = {2022}
}
```

## Contact
If you have any questions/issues, please create an issue in this repo or contact us at [this email](doxxx104@umn.edu). 


