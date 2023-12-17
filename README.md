# This repo is for the CSE5825 class project

## 1. Download the dataset 

` wget https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset/download?datasetVersionNumber=2`

## 2. Reorgnize the dataset

Use `python Reorgnize_data.py` file to reorgnize the data based on the different features and put them into one folder

Take the features `age` as an example, the final folder structure would be like this:

``` 
.
├── Adult
├── Baby
├── Senior
└── Young
```

Each subfolder will contain all the pictures in this category

## 3. Train and test

Run `python ResNet.py`. The result will be shown in the terminal