# Weakly Supervised Segmentation on Outdoor 4D point clouds with Temporal Matching and Spatial Graph Propagation


## 1. Dataset Preparation

The inital super voxel and the random sampling in our paper is ready [here](https://drive.google.com/drive/folders/1d2AQjE-22F44fjtSLYo4ra_EfD5mD2Lk?usp=sharing) 

The rule of initial frames sampling is:
```
(i % sr == 0) and ((i+sr-1) < len(sequence)) or (len(sequence)-sr) 
# i is the index number of frame (0,1,...) 
# sr is the sampling rate (1 pre 100 or 20 frames) 
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