import random
from fastcore.all import *
from fastai.vision.all import *
import os
import random


def get_random_file_path(directory):
    # List all files in the given directory
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    # Return a random file path
    if files:  # Check if the list is not empty
        return random.choice(files)
    else:
        return None


learn = load_learner('../trained_model/xray_pneumonia_model_8_80_randomcrop.pkl')

for i in range(1, 11):
    if i % 2 == 0:  # Even iterations
        random_file_path = get_random_file_path('../dataset_xray_test/PNEUMONIA')
    else:  # Odd iterations
        random_file_path = get_random_file_path('../dataset_xray_test/NORMAL')

    is_normal, _, probs = learn.predict(PILImage.create(random_file_path))

    print(f"Random file path: {random_file_path}")
    #   if probs[0] > 0.15:
    #       print(f"This is a: normal")
    #   else:
    print(f"This is a: {is_normal}.")

    print(f"Probability it's a normal: {probs[0]:.4f}")
