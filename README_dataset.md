# Egocentric Depth on everyday INdoor Activities (EDINA) Dataset

**:star2: EDINA test set is now available to download! :star2:**

## Overview
EDINA is an egocentric dataset that comprises more than 500K synchronized RGBD frames and gravity directions. Each instance in the dataset is a triplet: RGB image, depths and surface normals, and 3D gravity direction.

![edina2.gif](media/edina2.gif)

## Capturing Process
The data were collected using Azure Kinect cameras that provide RGBD images with inertial signals (rotational velocity and linear acceleration). Eighteen participants were asked to perform diverse daily indoor activities, e.g., cleaning, sorting, cooking, eating, doing laundry, training/playing with pet, walking, shopping, vacuuming, making bed, exercising, throwing trash, watering plants, sweeping, wiping, while wearing a head-mounted camera. More information can be found in our main paper and supplementary materials.

## Download
_Note: Currently, only the test set is available for downloading. Training data is coming soon!_

We provide a Python script [download_edina.py](download_edina.py) to download the dataset conveniently. More information on dataset format can be found in the [Data Organization](#data-organization) section.

To download the whole dataset to `pathToDataset` (only test set is available now):

```
python3 download_edina.py --out_dir pathToDataset
```

We also provide optional functionalities where you can unzip the downloaded file and/or only download a specific scene (e.g., scene0016_01):

```
python3 download_edina.py --out_dir pathToDataset --id scene0016_01 --unzip
```

Please refer to [download_edina.py](download_edina.py) for more specific details.

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