毕业设计

基于对比学习的遥感图像匹配

使用SimCLR模型，在Hpatches数据集上进行训练和测试验证性能，然后从遥感图像数据集中提取图像patch进行训练和测试。
其中遥感图像数据集来自于论文GAFM---A-Noval-Affine-Covariant-Mismatch-Removal-master，包括三个遥感图像数据集。

# SimCLR: A Simple Framework for Contrastive Learning of Visual Representations in PyTorch
Arxiv link for the SimCLR paper : [A Simple Framework for Contrastive Learning of Visual Representations][1]  

SimCLR Project Structure
---
The skeletal overview of this project is as follows:
```bash
.
├── utils/
│     ├── __init__.py
│     ├── model.py
│     ├── ntxent.py
│     ├── plotfuncs.py
│     └── transforms.py
├── results/
│    ├── model/
│    │     ├── lossesfile.npz
│    │     ├── model.pth
│    │     ├── optimizer.pth
│    ├── plots/
│    │     ├── training_losses.png
├── linear_evaluation.py
├── main.py
├── simclr.py
├── README.md



