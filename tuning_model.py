import random
from fastcore.all import *
from fastai.vision.all import *

learn = load_learner('../trained_model/xray_pneumonia_model_8_80_randomcrop.pkl')

path = Path('../dataset_xray/')
from time import sleep

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[RandomResizedCrop(450, min_scale=0.2)]
).dataloaders(path, bs=16)

learn.dls = dls
learn.fine_tune(3)

learn.export('../trained_model/tune_v1.pkl')
