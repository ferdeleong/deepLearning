# -*- coding: utf-8 -*-
import fastbook

fastbook.setup_book()

# Exercise 1: Dog vs Cat

from fastai.vision.all import *

# Dowload and extract standard dataset
path = untar_data(URLs.PETS)/'images'

# If first charcater is uppercase letter
def is_cat(x): return x[0].isupper()

'''
- path: The root path where the dataset is located.
- get_image_files(path): Retrieve image files from the specified path.
- valid_pct=0.2: Random 20% of the data will be hold and used for validation. 80% training
- seed=42: Sets the random seed for reproducibility. Same validation set on each run
- label_func=is_cat: Determine the label for each image through the first character of the filename is uppercase.
- item_tfms=Resize(224): Applies a transformation to resize **each** image to a size of 224x224 pixels.
'''

dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

'''
- dls: The data loader for your task.
- resnet34: Convolutional Neural Network architecture.
- metrics=error_rate: The evaluation metric to monitor during training. Error rate is the fraction of incorrectly classified items.
'''

learn = vision_learner(dls, resnet34, metrics=error_rate)

'''
- epochs: 1 pass through to update the weights of the new random head
'''
learn.fine_tune(1)

# Upload image from device 

uploader = widgets.FileUpload()
uploader

img = PILImage.create(uploader.data[0])
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")

# Exercise 2: Semantic Segmentation

path = untar_data(URLs.CAMVID_TINY)


dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np.loadtxt(path/'codes.txt', dtype=str))

learn = unet_learner(dls, resnet34)
learn.fine_tune(8)

learn.show_results(max_n=6, figsize=(7,8))

# Exercise 3: NLP Text classification

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test', b
                                  s=32)
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)

learn.predict("I really liked that movie!")

'''
- Categorical columns (contain values that are one of a discrete set of choices)
- Continuous columns (contain a number that represents a quantity)

'''

from fastai.tabular.all import *

path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=accuracy)

learn.fit_one_cycle(3)

# Exercise 4: Predict number vs categories

from fastai.collab import *

path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv')
learn = collab_learner(dls, y_range=(0.5,5.5)) # Indicate target range
learn.fine_tune(10)

learn.show_results()

