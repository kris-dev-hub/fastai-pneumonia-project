import random
from fastcore.all import *
from fastai.vision.all import *

path = Path('../dataset_xray/')
from time import sleep

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[RandomResizedCrop(380, min_scale=0.4)]
).dataloaders(path, bs=16)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(6)

learn.export('../trained_model/xray_pneumonia_model_8_80_randomcrop.pkl')
