from multiprocessing import freeze_support

import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision.transforms import transforms
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np
import PIL
from pathlib import Path
import json
import glob
import io
import numpy

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

num_classes = 5
batch_size = 1
num_of_workers = 5

DATA_PATH_TRAIN = Path('/home/ikarandoha/Documents/projetMaster2/datacow/train/')
DATA_PATH_TEST = Path('/home/ikarandoha/Documents/projetMaster2/datacow/test')

trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

def load_json(train_folder):
    for path in glob.glob(train_folder + "/*.json"):
        print(path)
        d = json.load(open(path))
        rects = []
        image_filename = path.replace(".json", "." + d["imagePath"].split(".")[-1])
        img = PIL.Image.open(image_filename)
        images = []
        for shape in d["shapes"]:
            points = shape["points"]
            p1, p2 = points
            x1, y1 = p1
            x2, y2 = p2
            area = (x1, y1, x2, y2)
            img = img.crop(area)
            if numpy.min(img.size) > 10:
                yield trans(img), image_filename, area
            else:
                yield -1, "", ""

def from_string(data):
    img = PIL.Image.open(io.BytesIO(data))
    return trans(img)

def load_one(path):
    img = PIL.Image.open(path)
    return trans(img)

def fetch():
    # get some random training images
    dataset = ImageFolderWithPaths(root=DATA_PATH_TRAIN, transform=trans) # our custom dataset
    dataloader = DataLoader(dataset)
    return iter(dataloader)
