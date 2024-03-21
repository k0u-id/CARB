## Weakly Supervised Semantic Segmentation for Driving Scenes (AAAI 2024)

__Official pytorch implementation of "Weakly Supervised Semantic Segmentation for Driving Scenes"__

> [Weakly Supervised Semantic Segmentation for Driving Scenes](https://arxiv.org/abs/2312.13646) <br>
> Dongseob Kim<sup>\*,1 </sup>, Seungho Lee<sup>\*,1 </sup>, Junsuk Choe<sup>2 </sup>, Hyunjung Shim<sup>3 </sup> <br>
> <sup>1 </sup> Yonsei University, <sup>2 </sup> Sogang University, and <sup>3 </sup> Korea Advanced Institute of Science \& Technoloty <br>
> <sub>* </sub> indicates an equal contribution. <br>
>
> __Abstract__ _State-of-the-art techniques in weakly-supervised semantic segmentation (WSSS) using image-level labels exhibit severe performance degradation on driving scene datasets such as Cityscapes. To address this challenge, we develop a new WSSS framework tailored to driving scene datasets. Based on extensive analysis of dataset characteristics, we employ Contrastive Language-Image Pre-training (CLIP) as our baseline to obtain pseudo-masks. However, CLIP introduces two key challenges: (1) pseudo-masks from CLIP lack in representing small object classes, and (2) these masks contain notable noise. We propose solutions for each issue as follows. (1) We devise Global-Local View Training that seamlessly incorporates small-scale patches during model training, thereby enhancing the model's capability to handle small-sized yet critical objects in driving scenes (e.g., traffic light). (2) We introduce Consistency-Aware Region Balancing (CARB), a novel technique that discerns reliable and noisy regions through evaluating the consistency between CLIP masks and segmentation predictions. It prioritizes reliable pixels over noisy pixels via adaptive loss weighting. Notably, the proposed method achieves 51.8\% mIoU on the Cityscapes test dataset, showcasing its potential as a strong WSSS baseline on driving scene datasets. Experimental results on CamVid and WildDash2 demonstrate the effectiveness of our method across diverse datasets, even with small-scale datasets or visually challenging conditions._

## Updates

21 Mar, 2024: Initial upload



## Installation
**Step 0.** Install PyTorch and Torchvision following [official instructions](https://pytorch.org/get-started/locally/), e.g.,

```shell
pip install torch torchvision
# FYI, we're using torch==1.9.1 and torchvision==0.10.1
# We used docker image pytorch:1.9.1-cuda11.1-cudnn8-devel
```

**Step 1.** Install [MMCV](https://github.com/open-mmlab/mmcv).
```shell
pip install mmcv-full
# FYI, we're using mmcv-full==1.4.0 
```

**Step 2.** Install [CLIP](https://github.com/openai/CLIP).
```shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

**Step 3.** Install CARB.
```shell
git clone https://github.com/k0u-id/CARB.git
cd CARB
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

**Step 4.** Maybe you need. (if error occurs)
```shell
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
sudo apt-get install libmagickwand-dev
pip install yapf==0.40.1
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Dataset Preparation & Pretrained Checkpoint
In our paper, we experiment with Cityscapes, CamVid, and WildDash2.

- Example directory hierarchy
  ```
  CARB
  |--- data
  |    |--- cityscapes
  |    |    |---leftImg8bit
  |    |    |---gtFine
  |    |--- camvid11
  |    |    |---img
  |    |    |---mask
  |    |--- wilddash2
  |    |    |---img
  |    |    |---mask
  |--- work_dirs
  |    |--- output_dirs (config_name)
  |    | ...
  | ...
  ```

**Dataset**
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [Camvid](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
- [WildDash2](https://www.wilddash.cc/)

**Pretrained Checkpoint**
- [Cityscapes](https://drive.google.com/file/d/1acN1JK__LKzGV5TynQExiUVTot5PxE_D/view?usp=sharing)
- [CamVid](https://drive.google.com/file/d/1naC6bAfEmvoSaigPP3odNgPlra-ElODj/view?usp=sharing)
- [WildDash2](https://drive.google.com/file/d/1pmBRPKH8gvaC_ZsDQCYxjU9XNAOGw_e1/view?usp=sharing)

## training CARB
CARB trains segmentation model with single or dual path.
You need to prepair fixed-masks (pseudo-masks) for single path training.

**Step 0.** Download and convert the CLIP models, e.g.,
```shell
python tools/maskclip_utils/convert_clip_weights.py --model ViT16
# Other options for model: RN50, RN101, RN50x4, RN50x16, RN50x64, ViT32, ViT16, ViT14
```

**Step 1.** Prepare the text embeddings of the target dataset, e.g.,
```shell
python tools/maskclip_utils/prompt_engineering.py --model ViT16 --class-set city_carb
# Other options for model: RN50, RN101, RN50x4, RN50x16, ViT32, ViT16
# Other options for class-set: camvid, wilddash2
# Default option is ViT16, city_carb
```

**Train.** Here, we give an example of multiple GPUs on a single machine. 
```shell
# Please see this file for the detail of execution.
# You can change detailed configuration by changing config files (e.g., CARB/configs/carb/cityscapes_carb_dual.py)
bash tools/train.sh 
```

## Inference CARB
```shell
# Please see this file for the detail of execution.
bash tools/test.sh
```

## Acknoledgement
This is highly borrowed from [MaskCLIP](https://github.com/chongzhou96/MaskCLIP), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Thanks to Chong, zhou.

## Citation
If you use CARB or this code base in your work, please cite
```
@misc{kim2024weakly,
      title={Weakly Supervised Semantic Segmentation for Driving Scenes}, 
      author={Dongseob Kim and Seungho Lee and Junsuk Choe and Hyunjung Shim},
      year={2024},
      eprint={2312.13646},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```