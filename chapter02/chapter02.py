# -*- coding: utf-8 -*-
! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastbook import *

key = os.environ.get('AZURE_SEARCH_KEY', 'XXX')

search_images_bing

results = search_images_bing(key, 'grizzly bear')
ims = results.attrgot('contentUrl')
len(ims)

ims = ['http://3.bp.blogspot.com/-S1scRCkI3vY/UHzV2kucsPI/AAAAAAAAA-k/YQ5UzHEm9Ss/s1600/Grizzly%2BBear%2BWildlife.jpg']
dest = 'images/grizzly.jpg'
download_url(ims[0], dest)
im = Image.open(dest)
im.to_thumb(128,128)

# Download all the URLs for each of the search terms, each on a separate folder

bear_types = 'grizzly','black','teddy'
path = Path('bears')

if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} bear')
        download_images(dest, urls=results.attrgot('contentUrl'))

fns = get_image_files(path)
fns

# Check and delete corrupt images

failed = verify_images(fns)
failed.map(Path.unlink);

class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train,valid = add_props(lambda i,self: self[i])

bears = DataBlock(

    # Types for independent and dependent variables
    # X. Independent = set of images, prediction source
    # Y. Dependent = categories, target prediction
    blocks=(ImageBlock, CategoryBlock),

    # Underlying items, file paths
    get_items=get_image_files,

    # Random splitter with fixed seed
    splitter=RandomSplitter(valid_pct=0.2, seed=42),

    # Set labels using the file name
    get_y=parent_label,

    # Item transforms. Returns a DataBlock object
    item_tfms=Resize(128))

# Data source
# Includes validation and training
# Mini-batch of 64 items at a time in a single tensor
dls = bears.dataloaders(path)

dls.valid.show_batch(max_n=4, nrows=1)

bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

# Data augmentation. Randomized cropped image
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = bears.dataloaders(path)

# Same image, different versions of the crop transform
dls.train.show_batch(max_n=4, nrows=1, unique=True)

bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)