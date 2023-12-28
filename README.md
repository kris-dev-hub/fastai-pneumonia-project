# FastAI pneumonia detection

This is a project to detect pneumonia from chest x-ray images using the fastai. Based on [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset that was manually reviewed.

## Getting Started

Download dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract it to `../dataset_xray` folder. 

Create ../trained_model folder to store trained models

```shell
pip install -r requirements.txt.
```

## Train model

Check how much ram do you have to tune model settings.

```shell
nvidia-smi 
```

```shell
python3 train_model.py - train model
```

2263MiB /  4096MiB - my graphic card has 4GB of ram but only ~2GB is free, so I will use batch size 16 and image size 80.

Form my GTX 1070ti with 8GB ram I can use batch size 16 and image size 450. , you can also fit 32 batch with smaller images.

Model trained with 80x80 images and batch size 8 is less accurate than the model 380x380 16 batch size.

Going over 380x380 image size was not achieving better results.

Going over 16 batch size with smaller images was also less accurate.

Going on 8 batch with 690x690 images was also less accurate.

My optimal setting was bs=16 and img_size=380 for 8GB GPU ram.

Setup for 2GB - size 80, min_scale=0.2, bs=8

Setup for 8GB - size 380, min_scale=0.4, bs=16

## Things you can tune

item_tfms=[Resize(380, method='squish')] - this is image resize with squish method, you can also use crop or pad. It was achieving worst results than RandomResizedCrop.

item_tfms=[RandomResizedCrop(380, min_scale=0.4)] - I used RandomResizeCrop with image size 380, randomly cropped with a minimum scale of 40%

splitter=RandomSplitter(valid_pct=0.2, seed=42), - 20% of data is used for validation

learn.fine_tune(6) - I used 6 epochs to train model, there was no improvement after 6 epochs.

## Errors

If you get error like this:

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate
```

You need to lower batch size or image size.

## Validation

It is a good idea to keep 20% of data for validation. You can check how your model is doing on validation set.

Model randomly split data into train and validation set, but you should have a separate validation set for final validation that was not used for training.

I put part of my data in `../dataset_xray_test` folder and used it for validation with predict.py script.

This directory should contain the same structure as `../dataset_xray` folder.

```shell
python3 predict.py
```

Script output:

```shell
Random file path: ../dataset_xray_test/NORMAL/NORMAL2-IM-0027-0001.jpeg                                                          
This is a: NORMAL.
Probability it's a normal: 0.6500
Random file path: ../dataset_xray_test/PNEUMONIA/person103_bacteria_490.jpeg                                                     
This is a: PNEUMONIA.
Probability it's a normal: 0.0306
Random file path: ../dataset_xray_test/NORMAL/IM-0087-0001.jpeg                                                                  
This is a: NORMAL.
Probability it's a normal: 0.6822
Random file path: ../dataset_xray_test/PNEUMONIA/person152_bacteria_723.jpeg                                                     
This is a: PNEUMONIA.
Probability it's a normal: 0.0060

```

## Tuning the model

You can tune the model based on pre-trained model. tuning_model.py is an example how to do it.

## Results

I got >97% accuracy with 380x380 image size and batch size 16.