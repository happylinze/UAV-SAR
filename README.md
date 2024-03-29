# Readme

## Abstract
We have ensembles three models, including infoGCN+FR_Head, Skeleton-MixFormer, and STTFormer.

**In the end, we achieved the accuracy of 48.51% on v1, and the accuracy of 76.13% on v2.**

All prediction results are saved in the folder `ensemble_results`, just like `epoch1_test_score.pkl`, and you could use our prediction results for validation, or train these models by yourself according to `readme.md`.

## Dependencies
* python == 3.8
* pytorch == 1.1.3
* NVIDIA apex
* PyYAML, tqdm, tensorboardX, wandb

Run `pip install -e torchlight`.

You could find more details in the `requirements.txt`, or use the command `pip install -r requirements.txt`.

## Data Preparation
We used the data processing method based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN).
### Directory Structure
Put downloaded data into the following directory structure:
```
- data/
    - uav/
        - Skeleton/
            -all_sqe
            ... # txt data of UAV-Human
```
### Data Processing
We have removed two meaningless samples. If you need them, you could find them the specific samples in `./data/uav/statistics/missing_skes_name.txt`, and then put their names into `skes_available_name.txt`.

Generate UAV-Human dataset:
```shell
 cd ./data/uav
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton
 python get_raw_denoised_data.py
 # 1. Transform the skeleton to the center of the first frame
 # 2. Split the train set and the test set
 python seq_transformation.py
 # After that, you could get MMVRAC_CSv1.npz and MMVRAC_CSv2.npz
```
We have introduced a data processing method that uses calculating bone motion angles as a measure.
If you want to use them, ensure that you have got the `MMVRAC_CSv1.npz` and `MMVRAC_CSv2.npz` by the above-mentioned data processing.
But it could spend lots of time. 

```shell
# Make sure you have got MMVRAC_CSv1.npz and MMVRAC_CSv2.npz
cd ./data/uav
# Calculate the angle from the data
python gen_angle_data.py
# After that, you could get MMVRAC_CSv1_angle.npz and MMVRAC_CSv2_angle.npz
```
## Training & Testing
### Quick Inference
You could use `ensemble.py` to validate the accuracy on v1 and v2. The command as:
```shell
python ensemble.py
```

### Training
#### [InfoGCN](https://github.com/stnoah1/infogcn)
For example, these are commands for training InfoGCN on view1. Please change the arguments `--config` and `--work-dir` to custom your training. If you want to training on v2, we have prepared the arguments in `./infogcn(FR_Head)/config`.
1. we added the [FR_Head](https://github.com/zhysora/FR-Head) module into infoGCN when training the model. And we trained the k=1, k=2, and k=6.
You could use the commands as:
```shell
cd ./infogcn(FR_Head)

# k=1 use FR_Head
python main.py --config ./config/uav_csv1/FR_Head_1.yaml --work-dir <the save path of results> 

# k=2 use FR_Head
python main.py --config ./config/uav_csv1/FR_Head_2.yaml --work-dir <the save path of results>
    
# k=6 use FR_Head
python main.py --config ./config/uav_csv1/FR_Head_6.yaml --work-dir <the save path of results>
```
2. We also trained the motion by k=1, k=2 and k=6. You could use the commands as:
```shell
cd ./infogcn(FR_Head)

# By motion k=1
python main.py --config ./config/uav_csv1/motion_1.yaml --work-dir <the save path of results>

# By motion k=2
python main.py --config ./config/uav_csv1/motion_2.yaml --work-dir <the save path of results>

# By motion k=6
python main.py --config ./config/uav_csv1/motion_6.yaml --work-dir <the save path of results>
```
3. The default sample frames for model is 64, we also trained the 32 frames and the 128 frames.The commands as:
```shell
cd ./infogcn(FR_Head)

# use 32 frames
python main.py --config ./config/uav_csv1/32frame_1.yaml --work-dir <the save path of results>

# use 128 frames
python main.py --config ./config/uav_csv1/128frame_1.yaml --work-dir <the save path of results>
```
4. After get the `MMVRAC_CSv1_angle.npz` and `MMVRAC_CSv2_angle.npz`, we trained the data by the command as:
```shell
cd ./infogcn(FR_Head)

# use angle to train
python main.py --config ./config/uav_csv1/angle_FR_Head_1.yaml --work-dir <the save path of results>
```
5. We also tried the FocalLoss to optimize the model. The command as:
```shell
cd ./infogcn(FR_Head)

# use focalloss
python main.py --config ./config/uav_csv1/focalloss_1.yaml --work-dir <the save path of results>
```

#### [Skeleton-MixFormer](https://github.com/ElricXin/Skeleton-MixFormer)
For example, these are commands for training Skeleton-Mixformer on v1. Please change the arguments `--config` and `--work-dir` to custom your training. If you want to training on v2, we have prepared the arguments in `./mixformer/config`.

1. We trained the model in k=1, k=2 and k=6. You could use the commands as:
```shell
cd ./mixformer

# k=1
python main.py --config ./config/uav_csv1/_1.yaml --work-dir <the save path of results>

# k=2
python main.py --config ./config/uav_csv1/_2.yaml --work-dir <the save path of results>

# k=6
python main.py --config ./config/uav_csv1/_6.yaml --work-dir <the save path of results>
```

2. We also trained the model in k=1, k=2 and k=6 with motion. The commands as:
```shell
cd ./mixformer

# By motion k=1
python main.py --config ./config/uav_csv1/motion_1.yaml --work-dir <the save path of results>

# By motion k=2
python main.py --config ./config/uav_csv1/motion_2.yaml --work-dir <the save path of results>

# By motion k=6
python main.py --config ./config/uav_csv1/motion_6.yaml --work-dir <the save path of results>
```

3. And we tried the angle data to train the model. The command as:
```shell
cd ./mixformer

# use angle
python main.py --config ./config/uav_csv1/angle_1.yaml --work-dir <the save path of results>
```

#### [STTFormer](https://github.com/heleiqiu/STTFormer)
For example, these are commands for training STTFormer on v1. Please change the arguments `--config` and `--work-dir` to custom your training. If you want to training on v2, we have prepared the arguments in `./sttformer/config`.

We trained joint, bone and motion. The commands as follows:
```shell
cd ./sttformer

# train joint
python main.py --config ./config/uav_csv1/joint.yaml --work-dir <the save path of results>

# train bone
python main.py --config ./config/uav_csv1/bone.yaml --work-dir <the save path of results>

# train motion
python main.py --config ./config/uav_csv1/motion.yaml --work-dir <the save path of results>
```
### Testing
If you want to test any trained model saved in `<work_dir>`, run the following command: 
```shell
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt
```
It will get a file `epoch1_test_score.pkl` which save the prediction score, put them into the following directory structure:
```
- ensemble_results/
    - infogcn/
        - CSv1/
            - 32frame_1/
                - epoch1_test_score.pkl
            - 128frame_1/
                - epoch1_test_score.pkl
            - angle_1/
                - epoch1_test_score.pkl
            -FocalLoss_1/
                - epoch1_test_score.pkl
            - FR_Head_1/
                - epoch1_test_score.pkl
            - FR_Head_2/
                - epoch1_test_score.pkl
            - FR_Head_6/
                - epoch1_test_score.pkl
            - motion_1/
                - epoch1_test_score.pkl
            - motion_2/
                - epoch1_test_score.pkl
            - motion_6/
                - epoch1_test_score.pkl
            ...
        - CSv2/
            ...
    - mixformer/
        - CSv1/
            - angle_1/
                - epoch1_test_score.pkl
            - _1/
                - epoch1_test_score.pkl
            - _2/
                - epoch1_test_score.pkl
            - _6/
                - epoch1_test_score.pkl
            - motion_1/
                - epoch1_test_score.pkl
            - motion_2/
                - epoch1_test_score.pkl
            - motion_6/
                - epoch1_test_score.pkl
            ...
        - CSv2/
            ...
    - sttformer/
        - CSv1/
            - angle/
                - epoch1_test_score.pkl
            - b/
                - epoch1_test_score.pkl
            - j/
                - epoch1_test_score.pkl
            - m/
                - epoch1_test_score.pkl
            ...
        - CSv2/
            ...
```
Then run the command as:
```shell
python ensemble.py
```
## Acknowledgements
This repo is based on [Skeleton-MixFormer](https://github.com/ElricXin/Skeleton-MixFormer), [Info-GCN](https://github.com/stnoah1/infogcn) and [STTFormer](https://github.com/heleiqiu/STTFormer). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch).

Thanks to the original authors for their work! 

If you have any questions, please concat  [youwei](youwei_zhou@stu.jiangnan.edu.cn) and [linze](linze.li@stu.jiangnan.edu.cn).
