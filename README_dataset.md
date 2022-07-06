# Egocentric Depth on everyday INdoor Activities (EDINA) Dataset

**:star2: Full EDINA dataset is now available to download! :star2:**

## Overview
EDINA is an egocentric dataset that comprises more than 500K synchronized RGBD frames and gravity directions. Each instance in the dataset is a triplet: RGB image, depths and surface normals, and 3D gravity direction.

![edina2.gif](media/edina2.gif)

## Capturing Process
The data were collected using Azure Kinect cameras that provide RGBD images with inertial signals (rotational velocity and linear acceleration). Eighteen participants were asked to perform diverse daily indoor activities, e.g., cleaning, sorting, cooking, eating, doing laundry, training/playing with pet, walking, shopping, vacuuming, making bed, exercising, throwing trash, watering plants, sweeping, wiping, while wearing a head-mounted camera. More information can be found in our main paper and supplementary materials.

## Dataset Downloading

### Raw data
We provide a Python script [download_edina.py](download_edina.py) to download the dataset conveniently. More information on dataset format can be found in the [Data Organization](#data-organization) section.

To download the dataset to `pathToDataset` AND unzip, you can use the following command by specifying the `--split` argument to be either `train` or `test` to download the corresponding train/test data:

```
python3 download_edina.py --out_dir pathToDataset --split test --unzip
```

We also provide optional functionalities where you can only download a specific scene (e.g., scene0016_01):

```
python3 download_edina.py --out_dir pathToDataset --id scene0016_01 --unzip
```

Please refer to [download_edina.py](download_edina.py) for more specific details.

### Dataset splits
We provide the `.pkl` file that specifies the train/test split of our data, specified by the dictionary keys `edina_train` and `edina_test`. The pickle file can be downloaded to [`./pickles/`](./pickles) folder by:

```     
wget -O scannet_edina_camready_final_clean.pkl https://edina.s3.amazonaws.com/pickles/scannet_edina_camready_final_clean.pkl && mv scannet_edina_camready_final_clean.pkl ./pickles/
```


## Data Organization
There is a separate directory for each RGB-D-(Normal-Depth) sequence (with varied length). Each sequence is named uniquely in the format of `scene<participantID>_<videoID>`, or `scene%04d_%02d`, sorted sequentially by `participantID` (from 0 to 17) and `videoID` (0-indexed) per participant. 
Within each sequence, `<frameID>` is also 0-indexed and of format `%06d`.

The general data hierarchy is described below:

```
scene<participantID>_<videoID>
├── color
│   └── color_<frameID>.png
├── depth
│   └── depth_<frameID>.png
├── normal
│   └── normal_<frameID>.png
├── gravity
│   └── gravity_<frameID>.txt
```