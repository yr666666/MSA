import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import yaml
from PIL import Image
import numpy as np
import json as jsonmod
from ipdb import set_trace
import random
from tqdm import tqdm,trange

import numpy as np
import cv2
from PIL import Image
from pylab import*
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, transform=None,ids_=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
       

        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                #五个句子
                self.ids += [(i, x) for x in range(len(d['sentences']))]
                #一个句子
                # self.ids += [(i, random.randint(0, len(d['sentences'])-1))]
                # self.ids += [(i, x) for x in range(len(d['sentences']))]


    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """

        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        cap_id = ann_id[1]
        caption = self.dataset[img_id]['sentences'][cap_id]['raw']
        # set_trace()
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        
        

        return image, caption ,img_id

    def __len__(self):
        return len(self.ids)




def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images,captions,iids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    return images,captions,iids

def get_loader_single(split, root, json, transform,
                      batch_size=100, shuffle=True,
                      num_workers=0, ids=None, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dataset = FlickrDataset(root=root,
                            split=split,
                            json=json,
                            transform=transform,
                            ids_=ids)
    print("-------------------- "+ split + ": " + str(len(dataset)) + " ------------------------------")
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def get_transform(split_name):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    t_list = [transforms.Resize(224)]
    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform



def get_loaders(batch_size, opt):
    # Build Dataset Loader

    transform = get_transform('train')
    train_loader = get_loader_single( 'train',
                                        opt["dataset"]["data_image"],
                                        opt["dataset"]["data_json"],
                                        transform,
                                        batch_size=batch_size, shuffle=True,
                                        collate_fn=collate_fn)

    transform = get_transform('val')
    val_loader = get_loader_single( 'val',
                                    opt["dataset"]["data_image"],
                                    opt["dataset"]["data_json"],
                                    transform,
                                    batch_size=batch_size, shuffle=False,
                                    collate_fn=collate_fn)
    transform = get_transform('test')
    test_loader = get_loader_single( 'test',
                                    opt["dataset"]["data_image"],
                                    opt["dataset"]["data_json"],
                                    transform,
                                    batch_size=1, shuffle=False,
                                    collate_fn=collate_fn)
    if opt["dataset"]["data_json"].split("/")[-2] == "RSITMD":
        # set_trace()
        val_loader = test_loader
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    options = parser_options()
    train_loader, val_loader, test_loader = get_loaders(options["dataset"]["batch_size"], options)
    for step, (image,text) in tqdm(enumerate(train_loader), leave=False):
        print(image.shape)
        # print(text)
    for step, (image,text) in tqdm(enumerate(test_loader), leave=False):
        print(image.shape)
        # print(text)