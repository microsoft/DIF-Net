## Deformed Implicit Field: Modeling 3D Shapes with Learned Dense Correspondence ##

<p align="center"> 
<img src="/imgs/dif.png" width="800">
</p>

This is an official pytorch implementation of the following paper:

Y. Deng, J. Yang, and X. Tong, **Deformed Implicit Field: Modeling 3D Shapes with Learned Dense Correspondence**, IEEE Computer Vision and Pattern Recognition (CVPR), 2021.

[Paper](https://arxiv.org/abs/2011.13650) | [Slides](https://github.com/YuDeng/DIF-Net_/blob/main/slides/deformed%20implicit%20field_slides.pdf)

Abstract: _We propose a novel **Deformed Implicit Field (DIF)** representation for modeling 3D shapes of a category and generating dense correspondences among shapes. With DIF, a 3D shape is represented by a template implicit field shared across the category, together with a 3D deformation field and a correction field dedicated for each shape instance. Shape correspondences can be easily established using their deformation fields. Our neural network, dubbed DIF-Net, jointly learns a shape latent space and these fields for 3D objects belonging to a category without using any correspondence or part label. The learned DIF-Net can also provides reliable correspondence uncertainty measurement reflecting shape structure discrepancy. Experiments show that DIF-Net not only produces high-fidelity 3D shapes but also builds high-quality dense correspondences across different shapes. We also demonstrate several applications such as texture transfer and shape editing, where our method achieves compelling results that cannot be achieved by previous methods._

## Features

### ● Surface reconstruction
We achieve comparable results on surface reconstruction task with other implicit-based shape representation.
<p align="center"> 
<img src="/imgs/recon.png" width="450">
</p>

### ● Dense correspondence reasoning
Our model gives reasonable dense correspondence between shapes in a category.

<p align="center"> 
<img src="/imgs/corres.png" width="700">
</p>

### ● Awareness of structure discrepancy
Our model predicts correspondence uncertainty between shapes in a category, which depicts structure discrepancies.
<p align="center"> 
<img src="/imgs/uncertainty.png" width="700">
</p>

## Installation
To run the code and models, you need to first download the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/microsoft/DIF-Net.git --recursive
cd DIF-Net
conda env create -f environment.yml
source activate dif
```

## Generating shapes with pre-trained models
1. Download the pre-trained models from this [link](https://drive.google.com/file/d/1j74W9KGAYIMDfEkAP50YJGpPKf1whmTz/view?usp=sharing). Unzip all files into ./models subfolder and organize the directory structure as follows:
```
DIF-Net
│
└─── models
    │
    └─── car
    │   │
    |   └─── checkpoints
    |       |
    |       └─── *.pth
    │
    └─── plane
    │   │
    |   └─── checkpoints
    |       |
    |       └─── *.pth
    ...
```

2. Run the following script to generate 3D shapes using a pre-trained model:
```
# generate 3D shapes of certain subjects in certain category
python generate.py --config=configs/test/<category>.yml --subject_idx=0,1,2
```
The script should generate meshes with color-coded template coordinates (in ply format) into ./recon subfolder. The color of a surface point records the 3D location of its corresponding point in the template space, which indicates dense correspondence information. We recommand using MeshLab to visualize the meshes.
<p align="center"> 
<img src="/imgs/generate.png" width="400">
</p>

## Training a model from scratch
### Data preparation
Pre-processed data from [ShapeNet-v2](https://shapenet.org/) can be download from this [link]() (410 GB in total for four categories). The data contains surface points along with normals, and randomly sampled free space points with their SDF values. The data should be organized as the following structure:
```
DIF-Net
│
└─── datasets
    │
    └─── car
    │   │
    |   └─── surface_pts_n_normal
    |   |   |
    |   |   └─── *.mat
    │   |
    |   └─── free_space_pts
    |       |
    |       └─── *.mat    
    |
    └─── plane
    │   │
    |   └─── surface_pts_n_normal
    |   |   |
    |   |   └─── *.mat
    │   |
    |   └─── free_space_pts
    |       |
    |       └─── *.mat    
    ...
```
Alternatively, you can pre-process the data by yourself. We use [mesh_to_sdf](https://github.com/marian42/mesh_to_sdf) provided by [marian42](https://github.com/marian42) to extract surface points as well as calculate SDF values for ShapeNet meshes. Please follow the instruction of the repository to install it.
### Training networks
Run the following script to train a network from scratch using the pre-processed data:
```
# train dif-net of certain category
python train.py --config=configs/train/<category>.yml
```
By default, we train the network with a batchsize of 256 for 60 epochs on 8 Tesla V100 GPUs, which takes around 4 hours. Please adjust the batchsize according to your own configuration.
## Evaluation
To evaluate the trained models, run the following script:
```
# evaluate dif-net of certain category
python eval.py --config=configs/eval/<category>.yml
```
The script will first embed test shapes of certain category into DIF-Net latent space, then calculate chamfer distance between embedded shapes and ground truth point clouds. We use [Pytorch3D](https://github.com/facebookresearch/pytorch3d) for chamfer distance calculation. Please follow the instruction of the repository to install it.

## Contact
If you have any questions, please contact Yu Deng (t-yudeng@microsoft.com) and Jiaolong Yang (jiaoyan@microsoft.com)

## License

Copyright &copy; Microsoft Corporation.

Licensed under the MIT license.

## Citation

Please cite the following paper if this work helps your research:

    @inproceedings{deng2021deformed,
		title={Deformed Implicit Field: Modeling 3D Shapes with Learned Dense Correspondence},
    	author={Yu Deng and Jiaolong Yang and Xin Tong},
	    booktitle={IEEE Computer Vision and Pattern Recognition},
	    year={2021}
	}

## Acknowledgement
This implementation takes [SIREN](https://github.com/vsitzmann/siren) as a reference. We thank the authors for their excellent work. 
