## Neural Reference Synthesis for Inter Frame Coding
DANdan Ding*, Xiang Gao*, Chenran Tang*, Zhan Ma**<br>
\* Hangzhou Normal University<br>
** Nanjing University<br>
___

## Introduction:
High-efficiency inter frame coding contributes significantly to the overall compression performance in hybrid block-based video coding solutions for decades. Its principle
behind is searching for the optimal reference block that leads to the minimal rate-distortion cost for motion estimation and motion compensation (MEMC). Thus, this work suggests the neural reference synthesis (NRS) to generate high-fidelity reference blocks for MEMC. The NRS has two submodules, namely reconstruction enhancement and reference generation, which directly impact the quality of reference blocks. Although existing works have developed numerous expensive learning-based methods for these two submodules, they basically deal with them separately without considering the inter dependencies in coding loop, resulting in limited coding gains.

By contrast, We propose to jointly optimize these two submodules to effectively exploit the spatiotemporal correlations for better characterization of structural and texture variations of pixel blocks. Specifically, we develop two deep neural networks (DNNs) based models, called EnhNet and GenNet, for reconstruction enhancement and reference generation respectively. The EnhNet model is mainly leveraging the spatial correlations within the current frame, and the GenNet is then augmented by further exploring the temporal correlations across multiple frames. Moreover, we devise a collaborative training strategy in these two neural models for practically avoiding the data over-fitting induced by iterative filtering propagated across temporal reference frames.Such jointly-optimized NRS not only offers state-of-the-art coding gains, e.g., >10% BD-Rate (Bjøntegaard Delta Rate) reduction against the High Efficiency Video Coding(HEVC) anchor for a variety of common test video sequences encoded at a wide bit range in both low-delay and random access settings, but also greatly reduces the complexity relative to existing learning-based methods by applying much lighter DNNs.


## Model
To download the pre-trained EnhNet models: https://drive.google.com/file/d/1q4H8xfgaGiVE4SqgtoNTbPa_RUOE9AmD/view?usp=sharing and pre-trained GenNet models: https://drive.google.com/file/d/1iS8fICkYK96x6VYtZ5wKM2lMaoCfPkf_/view?usp=sharing. Just unzip them and put them into folders ./EnhNet and ./GenNet, respectively. 

___

## Usage
Run the training script for EnhNet：
python3 TRAIN_CNNX.py

Run the testing script for EnhNet：
```python
python3 Test_CNNX.py
```

Run the training script for step 1 (GenNet):
```python
python3 GenNet_train_step1.py --subset=train
```

Run the training script for step 2 (GenNet):
```python
python3 GenNet_train_step2.py --subset=train --pretrained_model_checkpoint_path=./checkpoints/step1
```

Run the training script for step 3 (GenNet):
```python
python3 GenNet_train_step2.py --subset=train --pretrained_model_checkpoint_path=./checkpoints/step2
```

Run your own pair of frames (GenNet):
```python
python3 test.py --pretrained_model_checkpoint_path=./checkpoints/step2 --first=./first.png --second=./second.png --out=./out.png
```

___




