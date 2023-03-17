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


## 3. Evalution

The codes to validate the models are under the `train/validation`.
```
cd train/validation
```


Modify some parameters in the python scripts under `train/validation`:
- `data_base`: your postion of SemanticKITTI, which contains `sequences`. 

There are six trained models, which are reported in Table 1 of our submission.
- `pretrain-v1-42-ep200.pth`: Baseline-A 
  - Using only 0.001% initial annotations to train a segmentation
model
  - Validation: `python val_pretrain.py`
- `model-a-spp-tempGM.pth`: Model-A 
  - Using temporal matching module with greedy matching to update the pseudo label and train a new model
  - Validation: `python val_spp_tempGM.py`
- `model-b-spp-tempOT.pth`: Model-B
  - Using temporal matching module with optimal transport to update the pseudo label and train a new model  
  - Validation: `python val_spp_tempOT.py`
- `model-c-dsp-tempOT-sgp.pth`: Model-C
  - Without seed point propgation stage
  - Conbining temporal matching module with optimal transport and spatial graph propagation to update the pseudo label and train a new model  
  - Validation: `python val_dsp_tempOT_sgp.py`
- `model-d-spp-tempOT-dsp-tempGM-sgp.pth`: Model-D
  - Using the model-b-spp-tempOT.pth as the initial model to update the pseudo
  - Conbining temporal matching module with greedy matching and spatial graph propagation to update the pseudo label and train a new model  
  - Validation: `python val_spp_tempOT_dsp_tempGM_sgp.py`
- `model-e-spp-tempOT-dsp-tempOT-sgp.pth`: Model-E
  - Using the model-b-spp-tempOT.pth as the initial model to update the pseudo
  - Conbining temporal matching module with optimal transport and spatial graph propagation to update the pseudo label and train a new model  
  - Validation: `python val_spp_tempOT_dsp_tempOT_sgp.py`


## 3. Training

### 3.1 Supervoxel Segmentation & Sampple Point Generation

The codes are under the `W4DTS/tools` or you can derictly download our intial annotations in [here](https://drive.google.com/drive/folders/1d2AQjE-22F44fjtSLYo4ra_EfD5mD2Lk?usp=sharing).
```
cd W4DTS/tools
```
1. Modify `data_base` to your postion of SemanticKITTI, which contains `sequences`, in `W4DTS/tools/get_supervoxel_kitti.py` and `gen_sample_point_v2_kitti.py`

2. Generate the supervoxel with:
  ```
  python get_supervoxel_kitti.py
  ```

  The script depends on `W4DTS/tools/superV/supervoxel_vk`, which is a C++ program. 
  If the `W4DTS/tools/superV/supervoxel_vk` is not working on your workstation, you need to compile your own version with our inplementation:
  ```
  git clone https://github.com/20227469/Supervoxel-for-3D-point-clouds.git
  cd Supervoxel-for-3D-point-clouds
  g++ -o main_wrapper.cc supervoxel_vk
  #Replace W4DTS/tools/superV/supervoxel_vk with new compiled supervoxel_vk 
  ```

3. Generate the inital annotation sample:
    ```
    python gen_sample_point_v2_kitti.py
    ```
   
### 3.2 Train Model in Seed Point Propagation Stage
The code to train the models in SPP is under the `train/stage_SPP`.
```
cd train/stage_SPP
```

1. Modify `data_base` in YAML file to your postion of SemanticKITTI, which contains `sequences`, in `train/stage_SPP/train_tempGM_spp.py` and `train/stage_SPP/train_tempOT_spp.py`

2. Modify `log_pos` in YAML file for saving the training logs and model snapshots. The training logs and model snapshots are saved in the `log_pos`

3. Train the model with:
    ```
    # Training model with the temporal matching with greedy matching. (require 60 epoch)
    python train_tempGM_spp.py
    # Training model with the temporal matching with optimal transport. (require 90 epoch)
    python train_tempOT_spp.py
    ```
   
### 3.3 Train Model in Dense Scene Propagation Stage
The code to train the models in SPP is under the `train/stage_DSP`.
```
cd train/stage_DSP
```

1. Modify `data_base` in YAML file to your position of SemanticKITTI, which contains `sequences`, in `train/stage_DSP/train_tempGM_spp.py` and `train/stage_DSP/train_tempOT_spp.py`

2. Modify `log_pos` in YAML file for saving the training logs and model snapshots. The training logs and model snapshots are saved in the `log_pos`

3. Choose the initial model from previous stage with modifying `pretrain_dir`. 
   The default model is our `pretrained/model-b-spp-tempOT.pth`.

4. Train the model with:
    ```
    # Training model with the temporal matching with greedy matching. 
    python train_tempGM_sgp.py
    # Training model with the temporal matching with optimal transport. 
    python train_tempOT_sgp.py
    ```





