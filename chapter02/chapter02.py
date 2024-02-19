! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *

key = os.environ.get('AZURE_SEARCH_KEY', 'X')

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

# Data augmentation
bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)

# Standard size 244x244 pixel
# Train bear classifier
bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = bears.dataloaders(path)

# Create learner and fine-tune
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

# Confusion matrix to measure accuracy
# Diagonal correct classification

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

interp.plot_top_losses(5, nrows=1)

#hide_output
cleaner = ImageClassifierCleaner(learn)
cleaner

# Unlink all images selected for deletion
for idx in cleaner.delete(): cleaner.fns[idx].unlink()

# Different category
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)

learn.export()

# Check if file exists
path = Path()
path.ls(file_exts='.pkl')

# Simulating inference
learn_inf = load_learner(path/'export.pkl')

# Filename to predict
learn_inf.predict('images/grizzly.jpg')

# Vocab, or stored list of all possible categories
learn_inf.dls.vocab

# Ipywidget
btn_upload = widgets.FileUpload()
btn_upload

img = PILImage.create(btn_upload.data[-1])

# Display
out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl

# Get predictions
pred,pred_idx,probs = learn_inf.predict(img)

# Display
lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred

# Classification button
btn_run = widgets.Button(description='Classify')
btn_run

def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)

# Reset values
btn_upload = widgets.FileUpload()

VBox([widgets.Label('Select your bear!'),
      btn_upload, btn_run, out_pl, lbl_pred])

