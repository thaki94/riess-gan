# riess-gan

## Introduction
This repository contains the supplementary online materials of our work **All Weather Radar Image Enhancement and Semantic Segmentation Method for Autonomous Vehicles**. The proposed method is a generative adversarial network (GAN)-based approach for the enhancement and semantic segmentation of radar images for autonomous navigation applications. The inference phase of the proposed method is shown in Figure 1. For more information about this work, please see our <a href="https://yettobeadded.com" target="_blank">paper</a>.

<div align="center">
  <img src="images/Fig3.svg" alt="Figure 1" width="700">
</div>

This repository contains the following:

1. Sample testing data
2. Inference code
3. Link to the weight files of the trained GAN models

## Folder Structure

```
riess-gan/ 
├── README.md
├── modules/                  
│ ├── gan.py                  # code for gan models
│ ├── utils.py                # code for loading testing data, visualizing and saving the results
├── results/                  # directory for saving results
├── sample_testing_data/      # directory for sample testing data
│ ├── dataset_1               # testing data of dataset 1 (good weather - seen data sequence)
│ │ ├── test_data                # testing data
│ │ ├── en_test_target           # ground truth - enhancement
│ │ ├── ss_test_target           # ground truth - semantic segmentation 
│ ├── dataset_2               # testing data of dataset 2 (good weather - unseen data sequence)
│ │ ├── test_data                # testing data
│ │ ├── en_test_target           # ground truth - enhancement
│ │ ├── ss_test_target           # ground truth - semantic segmentation                 
│ ├── dataset_3              # testing data of dataset 3 (snow - no ground truth)
│ │ ├── test_data              # testing data
│ ├── dataset_4              # testing data of dataset 4 (rain - no ground truth)
│ │ ├── test_data              # testing data  
├── inference.ipynb          # inference code (Jupyter notebook)
├── inference.py             # inference code (python file)
└── requirements.txt         # required python modules list.
```

## Instructions to run the code

1. Clone this repository using ```git clone https://github.com/thaki94/riess-gan.git``` or download the repository manually to your computer
2. Download the weights files from the following <a href="https://drive.google.com/drive/folders/12X2MGjICpSVb8BPJ7NqRO1D_Su81U6w-?usp=sharing" target="_blank">link </a> and place them in a folder named ```models``` in the home location of the repository.
3. Install the required modules using the ```requirements.txt```. We suggest you create a separate environment to install the modules. This code is tested in ```Anaconda``` with ```python 3.11```

Run this code in the home directory of the repository.

```
conda create -n riess python=3.11
conda activate riess
pip install requirements.txt
```
4. Open the ```inference.ipynb``` using ```jupyter notebook``` and run the code. If you do not have ```jupyter notebook``` just run the ```inference.py```. The results will be saved in the ```results``` folder.
