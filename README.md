<div align=center>
  
# **[CVPR 2025]** Adaptive Rectangular Convolution for Remote Sensing Pansharpening

</div>

<div align=center>
<img src=".Marsc-Net.png">
</div>

Abstract: Fine-grained ship detection in remote sensing imagery is critical for maritime safety, marine environmental protection, and traffic management. Despite recent advances, existing methods still face key challenges, including insufficient multi-scale salient feature representation, limited modeling of structural relationships, and weak cross-scale consistency. To address these issues, we propose Marsc-Net as a solution. First, we design a multi-scale saliency and causal-intervention module to enhance salient feature representations by highlighting ship regions while suppressing background interference. Second, an adaptive structured graph attention module, equipped with a novel top-$k$ node selection strategy and adaptive graph construction, captures informative part-level relations and suppresses noisy connections, thereby improving fine-grained discriminability. Third, we develop a dynamic scale consistency contrastive module that explicitly enforces cross scale consistency by coupling prototype alignment with supervised contrastive learning, promoting intra-class compactness across scales and enhancing inter-class separability. Extensive experiments on HRSC2016 and FGSD2021 demonstrate that Marsc-Net achieves 84.88\% mAP and 78.39\% mAP, respectively, outperforming state-of-the-art methods by 3.29 and 2.07 mAP points. The source is available at [github](https://github.com/wrkzzzz1/Marsc-Net).

## 🛠 Getting started

### Setup environment

1. clone the repository

```bash
git clone git@github.com:wrkzzzz1/Marsc-Net.git
cd Marsc-Net
```
2. install dependencies

```bash
pip install -r requirements.txt
```

### Prepare dataset

Datasets can be downloaded from the [HRSC2016](https://ieee-dataport.org/documents/hrsc2016-0) and [FGSD2021](https://www.modelscope.cn/datasets/wokaikaixinxin/FGSD2021).

## 🚀 Train the model

```bash
python mmrotate/train.py
```
## :e-mail: Contact

If you have any questions, please email [`202430310043@stu.shmtu.edu.cn`](mailto:202430310043@stu.shmtu.edu.cn).

