
key = 'XXX'
key = os.environ['AZURE_SEARCH_KEY']


search_images_bing

results = search_images_bing(key, 'grizzly bear')
ims = results.attrgot('content_url')
len(ims)

dest = 'images/grizzly.jpg'
download_url(ims[0], dest)
im = Image.open(dest)
im.to_thumb(128,128)

bear_types = 'grizzly','black','teddy'
path = Path('bears')
if not path.exists(): path.mkdir()
for o in bear_types: dest = (path/o)
            dest.mkdir(exist_ok=True)
            results = search_images_bing(key, f'{o} bear')
            download_images(dest, urls=results.attrgot('content_url'))
            
fns = get_image_files(path)
fns

failed = verify_images(fns)

failed

failed.map(Path.unlink);


class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i] train,valid = add_props(lambda i,self: self[i])

bears = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128))

blocks=(ImageBlock, CategoryBlock)
