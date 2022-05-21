# CFBI: Collaborative Video Object Segmentation by Foreground-Background Integration
The PyTorch implementation of [Adaptive Selection of Reference Frames for Video Object Segmentation. ](https://ieeexplore.ieee.org/document/9665289)

Model Overview:
<div align=center><img src="https://github.com/z-x-yang/CFBI/raw/master/utils/overview.png" width="80%"/></div>


## Requirements
    1. Python3
    2. pytorch >= 1.3.0 and torchvision
    3. opencv-python and Pillow

## Getting Started
1. Prepare datasets:
    * Download the [training split](https://drive.google.com/file/d/13Eqw0gVK-AO5B-cqvJ203mZ2vzWck9s4/view?usp=sharing) of YouTube-VOS 2019, and decompress the file to `datasets/YTB/train`.
    * Download the [validation split](https://drive.google.com/file/d/1-QrceIl5sUNTKz7Iq0UsWC6NLZq7girr/view?usp=sharing) of YouTube-VOS 2018, and decompress the file to `datasets/YTB/valid`. If you want to evaluate CFBI on YouTube-VOS 2019, please download this [split](https://drive.google.com/file/d/1o586Wjya-f2ohxYf9C1RlRH-gkrzGS8t/view?usp=sharing) instead.
    * Download 480p [TrainVal](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) split of DAVIS 2017, and decompress the file to `datasets/DAVIS`. You can also download this [full-resolution TrainVal](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip), and decompress the file to `datasets/DAVIS` too. The training process will give priority to using the full-resolution data, which will slightly improve the performance when evaluating CFBI at a resolution greater than 480p.
    * Download pretrained backbone model, [ResNet101-DeepLabV3+](https://drive.google.com/file/d/1H3yUShfPqzxSt-nHJP-zYSbbu2o_RQQu/view?usp=sharing), to `pretrained_models`. The backbone is pretrained on COCO. If you want to pretrain a backbone by yourself, please refer to this [code](https://github.com/jfzhang95/pytorch-deeplab-xception). Besides, we also provide a light-weight backbone, [MobileNetV2-DeepLabV3+](https://drive.google.com/file/d/1vItBLa9bITy11cpcHHlRAXzDus0QNfjM/view?usp=sharing).
2. Training and Evaluating:
    * **YouTube-VOS**: Run `bash ytb_train_eval.sh`. The script will train CFBI using `4` GPUs. After the training process, the script will evaluate CFBI on the last `4` checkpoints. For each evaluation, the result will be packed into a Zip file, which you need to send to [official evaluation server](https://competitions.codalab.org/competitions/19544) to calculate the score. For 2019 version, use this [server](https://competitions.codalab.org/competitions/20127) instead.
    * **DAVIS**: After training on YouTube-VOS, run `bash davis_train_eval.sh` to finetune CFBI on DAVIS 2017. The script will evaluate and pack results into Zip as well. For calculating scores on validation split, please use [official code](https://github.com/davisvideochallenge/davis2017-evaluation).
    * **Specified evaluation**: If you want to specify the dataset and checkpoint in evaluation, please refer to `ytb_eval.sh`.
    * **Fast CFBI**: The default training batch size is `2` for each GPU, which takes about 13G memory. To reduce the memory usage, we also provide a fast setting in `ytb_train_eval_fast.sh` and `ytb_eval_fast.sh`. The fast setting enables using `float16` in the matching process of CFBI. Besides, we apply an `atrous strategy` in the global matching of CFBI for further efficiency (The discussion of atrous matching will be submitted to our Arxiv paper soon). Moreover, the local matching is changed from a parallel way into a `for-loop` way during training. The fast setting will save a large amount of memory and significantly improve the inference speed of CFBI. However, this will only lose very little performance (0.1~0.2 on YouTube-VOS 2018). **Notably**, the fast setting can be opened and closed dynamically during evaluation. You can train CFBI in the default setting but evaluate CFBI in the fast setting. This will only lose very little performance too.
    * Another way for saving memory is to increase the number of `--global_chunks`. This will not affect performance but will make the network speed slightly slower.

## Model Zoo
**We recorded the inference speed of CFBI by using one NVIDIA Tesla V100 GPU. Besides, we used a multi-object speed instead of a single-object. Almost every sequence in VOS datasets contains multiple objects, and CFBI is good at processing all of them simultaneously.**

`F16` denotes using `float16` in the matching process. `Fast` means using both `float16` and `atrous strategy` in the inference stage.

**YouTube-VOS** (Eval on Val 2018):

In the inference stage, we restricted the long edge of each frame to be no more than 1040 (800 * 1.3) pixels, which is the biggest random-scale size in the training and is smaller than the original size of YouTube-VOS (720p).

**Name** | **Backbone**  | **J Seen** | **F Seen** | **J Unseen** | **F Unseen** | **Mean** | **Multi-Obj** <br> **FPS** | **Link** 
---------| :-----------: | :--------: | :--------: | :----------: | :----------: | :------: | :------------------------: | :------:
ResNet101-CFBI | ResNet101-DeepLabV3+ | **81.9** | 86.3 | **75.6** | **83.4** | **81.8** | 3.48 | [Click](https://drive.google.com/file/d/1ZhoNOcDXGG-PpFXhCixs-L3yA255Wup8/view?usp=sharin) 
ResNet101-F16-CFBI | ResNet101-DeepLabV3+ |**81.9** | **86.4** | **75.6** | 83.3 | **81.8** | 4.62 (32.8%↑) | The same as above
ResNet101-Fast-CFBI | ResNet101-DeepLabV3+ | **81.9** | **86.4** | **75.6** | 83.1 |**81.8** | **7.61 (118.7%↑)** | The same as above
MobileNetV2-CFBI | MobileNetV2-DeepLabV3+ | 80.4 | 84.7 | 74.9 | 82.6 | 80.6 | 3.88 | [Click](https://drive.google.com/file/d/1L_pA2UzBbOWyyJyNmgnXqnqDvFfkdXpO/view?usp=sharing)
MobileNetV2-Fast-CFBI | MobileNetV2-DeepLabV3+ | 80.2 | 84.6 | 74.7 | 82.7 | 80.6 | **9.69 (150.0↑%)** | The same as above

**DAVIS** (Eval on Val 2017):

In the inference stage, we ran using the default size of DAVIS (480p).

**Name** | **Backbone**  | **J score** | **F score** | **Mean** | **Multi-Obj** <br> **FPS** | **Link** 
---------| :-----------: | :---------: | :---------: | :------: | :------------------------: | :------:
ResNet101-CFBI-DAVIS | ResNet101-DeepLabV3+ | **79.3** | **84.5** | **81.9** | 5.88 | [Click](https://drive.google.com/file/d/1ZhoNOcDXGG-PpFXhCixs-L3yA255Wup8/view?usp=sharin) 
ResNet101-F16-CFBI-DAVIS | ResNet101-DeepLabV3+ | 79.2 | 84.4 | 81.8 | 7.38 (25.5%↑) | The same as above
ResNet101-Fast-CFBI-DAVIS | ResNet101-DeepLabV3+ | 77.0 | 82.7 | 79.9 | **10.18 (73.1%↑)** | The same as above
MobileNetV2-CFBI-DAVIS | MobileNetV2-DeepLabV3+ | 76.5 | 80.3 | 78.4 | 6.94 | [Click](https://drive.google.com/file/d/1uuKlRqrPubJVRXKVr53cXuNFEQmmThGG/view?usp=sharing)
MobileNetV2-Fast-CFBI-DAVIS | MobileNetV2-DeepLabV3+ | 75.2 | 78.9 | 77.1 | **13.22 (90.5%↑)** | The same as above

Using `atrous strategy` on DAVIS leads to a obvious performance drop. The reason is that DAVIS is so small (only 30 videos in Train split) and easy to be overfitted. Training CFBI with `Fast` mode should significantly relief the performance drop.


## Citing
```
@ARTICLE{9665289,  
author={Hong, Lingyi and Zhang, Wei and Chen, Liangyu and Zhang, Wenqiang and Fan, Jianping},  
journal={IEEE Transactions on Image Processing},   
title={Adaptive Selection of Reference Frames for Video Object Segmentation},   
year={2022},  
volume={31},  
number={},  
pages={1057-1071},  
doi={10.1109/TIP.2021.3137660}
}
```

## Related Work
[CFBI](https://github.com/z-x-yang/CFBI)

