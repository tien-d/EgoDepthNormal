# Egocentric Scene Understanding via Multimodal Spatial Rectifier

This repository contains the source code for our paper:

**Egocentric Scene Understanding via Multimodal Spatial Rectifier**  
Tien Do, Khiem Vuong, and Hyun Soo Park  
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022  
[Project webpage](???) | [Dataset](???) | [arXiv](???) 

# Abstract

In this paper, we study a problem of egocentric scene understanding, i.e., predicting depths and surface normals from an
egocentric image. Egocentric scene understanding poses unprecedented challenges: (1) due to large head movements, the 
images are taken from non-canonical viewpoints (i.e., tilted images) where existing models of geometry prediction do 
not apply; (2) dynamic foreground objects including hands constitute a large proportion of visual scenes.

These challenges limit the performance of the existing models learned from large indoor datasets, such as ScanNet 
and NYUv2, which comprise predominantly upright images of static scenes. We present a multimodal spatial rectifier 
that stabilizes the egocentric images to a set of reference directions, which allows learning a coherent visual 
representation. Unlike unimodal spatial rectifier that often produces excessive perspective warp for egocentric 
images, the multimodal spatial rectifier learns from multiple directions that can minimize the impact of the 
perspective warp. To learn visual representations of the dynamic foreground objects, we present a new dataset called 
EDINA (Egocentric Depth on everyday INdoor Activities) that comprises more than 500K synchronized RGBD frames and 
gravity directions. Equipped with the multimodal spatial rectifier and the EDINA dataset, our proposed method on  
single-view depth and surface normal estimation significantly outperforms the baselines not only on our EDINA 
dataset, but also on other popular egocentric datasets, such as First Person Hand Action (FPHA) and 
EPIC-KITCHENS.


![epick_qualitative.jpg](media/epick_supp_qualitative_small.png)

<b>Figure 1: Qualitative results for depth, surface normal, and gravity 
prediction on EPIC-KITCHENS dataset.</b>

# Installation


# Egocentric Depth on everyday INdoor Activities (EDINA) Dataset

### Descriptions

![edina2.gif](media/edina2.gif)

<b>Figure 2: EDINA dataset.</b>

Participants, train? test?

Number of frames, train? test?

Links to download.

### Organization


# Citation
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

# Contact
If you have any questions/issues, please create an issue in this repo or contact us at [this email](doxxx104@umn.edu). 


