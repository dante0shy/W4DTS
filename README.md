# Weakly Supervised Segmentation on Outdoor 4D point clouds with Temporal Matching and Spatial Graph Propagation


## 1. Dataset Preparation
Download SemanticKITTI and unzip all the data. (should contain a directory, like`/some/thing/to/dataset/sequence`)

The inital super voxel and the random sampling in our paper is ready [here](https://drive.google.com/drive/folders/1d2AQjE-22F44fjtSLYo4ra_EfD5mD2Lk?usp=sharing) 
Then, unzip all the zip file to the SemanticKITTI directory. The directory should be like :

```
-/some/thing/to/dataset
    --sequence
    --sv_sample_point_v3
    --superV_v2
```
aa
In our paper, the rule of initial frames sampling is:
```
(i % sr == 0) and ((i+sr-1) < len(sequence)) or (len(sequence)-sr) 
# i is the index number of frame (0,1,...) 
# sr is the sampling rate (1 frame pre sr frames (100 or 20) ) 
# len(sequence) is the length of point cloud sequence
```

## 2. Environment & Installation

### 2.1 Conda Envirnemnt

Follow the installation guide of [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) 
to build an Anaconda envirnment. 

### 2.2 Requirements

- tqdm
- PyYAML
- scikit-learn

## TODO