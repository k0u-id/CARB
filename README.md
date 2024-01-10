## Weakly Supervised Semantic Segmentation for Driving Scenes (AAAI 2024)

__Official pytorch implementation of "Weakly Supervised Semantic Segmentation for Driving Scenes"__

> [__Official pytorch implementation of "Weakly Supervised Semantic Segmentation for Driving Scenes"__](https://arxiv.org/abs/2312.13646) <br>

> Dongseob Kim<sup>\*,1 </sup>, Seungho Lee<sup>\*,1 </sup>, Junsuk Choe<sup>2 </sup>, Hyunjung Shim<sup>3 </sup> <br>
> <sup>1 </sup> Yonsei University, <sup>2 </sup> Sogang University, and <sup>3 </sup> Korea Advanced Institute of Science \& Technoloty <br>
> <sub>* </sub> indicates an equal contribution. <br>
>
> __Abstract__ _State-of-the-art techniques in weakly-supervised semantic segmentation (WSSS) using image-level labels exhibit severe performance degradation on driving scene datasets such as Cityscapes. To address this challenge, we develop a new WSSS framework tailored to driving scene datasets. Based on extensive analysis of dataset characteristics, we employ Contrastive Language-Image Pre-training (CLIP) as our baseline to obtain pseudo-masks. However, CLIP introduces two key challenges: (1) pseudo-masks from CLIP lack in representing small object classes, and (2) these masks contain notable noise. We propose solutions for each issue as follows. (1) We devise Global-Local View Training that seamlessly incorporates small-scale patches during model training, thereby enhancing the model's capability to handle small-sized yet critical objects in driving scenes (e.g., traffic light). (2) We introduce Consistency-Aware Region Balancing (CARB), a novel technique that discerns reliable and noisy regions through evaluating the consistency between CLIP masks and segmentation predictions. It prioritizes reliable pixels over noisy pixels via adaptive loss weighting. Notably, the proposed method achieves 51.8\% mIoU on the Cityscapes test dataset, showcasing its potential as a strong WSSS baseline on driving scene datasets. Experimental results on CamVid and WildDash2 demonstrate the effectiveness of our method across diverse datasets, even with small-scale datasets or visually challenging conditions._

The code will be available soon.
