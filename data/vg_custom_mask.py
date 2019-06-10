#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import h5py
import PIL
import json
from torch.utils.data import DataLoader
from utils.data import imagenet_preprocess, Resize


class VgSceneGraphDataset(Dataset):
    def __init__(self, vocab, h5_path, image_dir, image_size=(256, 256),
                 normalize_images=True, max_objects=10, max_samples=None,
                 include_relationships=True, use_orphaned_objects=True):
        super(VgSceneGraphDataset, self).__init__()

        self.image_dir = image_dir
        self.image_size = image_size
        self.vocab = vocab
        self.num_objects = len(vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects-1
        self.max_samples = max_samples
        self.include_relationships = include_relationships

        transform = [Resize(image_size), T.ToTensor()]
        if normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)

        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

    def __len__(self):
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        img_path = os.path.join(self.image_dir, self.image_paths[index])

        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size

        # Figure out which objects appear in relationships and which don't
        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        if len(obj_idxs) > self.max_objects:
            obj_idxs = random.sample(obj_idxs, self.max_objects)
        if len(obj_idxs) < self.max_objects and self.use_orphaned_objects:
            num_to_add = self.max_objects - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)

        # random the obj_idxs
        random.shuffle(obj_idxs)

        O = len(obj_idxs)

        objs = torch.LongTensor(O).fill_(-1)

        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
        masks = torch.zeros(O, 1, H, W)
        # obj_idx_mapping = {}
        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx].item()
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
            x0 = float(x) / WW
            y0 = float(y) / HH
            x1 = float(x + w) / WW
            y1 = float(y + h) / HH
            boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
            masks[i, :, round(y0 * H):round(y1 * H), round(x0 * W):round(x1 * W)] = 1
            # obj_idx_mapping[obj_idx] = i

        # The last object will be the special __image__ object
        # objs[O - 1] = self.vocab['object_name_to_idx']['__image__']

        # triples = []
        # for r_idx in range(self.data['relationships_per_image'][index].item()):
        #     if not self.include_relationships:
        #         break
        #     s = self.data['relationship_subjects'][index, r_idx].item()
        #     p = self.data['relationship_predicates'][index, r_idx].item()
        #     o = self.data['relationship_objects'][index, r_idx].item()
        #     s = obj_idx_mapping.get(s, None)
        #     o = obj_idx_mapping.get(o, None)
        #     if s is not None and o is not None:
        #         triples.append([s, p, o])
        #
        # # Add dummy __in_image__ relationships for all objects
        # in_image = self.vocab['pred_name_to_idx']['__in_image__']
        # for i in range(O - 1):
        #     triples.append([i, in_image, O - 1])
        #
        # triples = torch.LongTensor(triples)
        return image, objs, boxes, masks


def vg_collate_fn(batch):
    """
    Collate function to be used when wrapping a VgSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triples: FloatTensor of shape (T, 3) giving all triples, where
      triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
      obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triples to images;
      triple_to_img[t] = n means that triples[t] belongs to imgs[n].
    """
    # batch is a list, and each element is (image, objs, boxes, triples)
    all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img = [], [], [], [], []
    # obj_offset = 0
    for i, (img, objs, boxes, masks) in enumerate(batch):
        all_imgs.append(img[None])
        O = objs.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_masks.append(masks)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        # obj_offset += O

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_masks = torch.cat(all_masks)
    all_obj_to_img = torch.cat(all_obj_to_img)

    out = (all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img)
    return out


def get_dataloader(batch_size=10, VG_DIR='/home/zhaobo/Data/vg'):
    vocab_json = os.path.join(VG_DIR, '178classes', 'vocab.json')
    train_h5 = os.path.join(VG_DIR, '178classes', 'train.h5')
    val_h5 = os.path.join(VG_DIR, '178classes', 'test.h5')
    vg_image_dir = os.path.join(VG_DIR, 'images')
    image_size = (64, 64)
    num_train_samples = None
    max_objects_per_image = 10
    vg_use_orphaned_objects = True
    include_relationships = False
    batch_size = batch_size
    shuffle_val = False

    # build datasets
    with open(vocab_json, 'r') as f:
        vocab = json.load(f)
    dset_kwargs = {
        'vocab': vocab,
        'h5_path': train_h5,
        'image_dir': vg_image_dir,
        'image_size': image_size,
        'max_samples': num_train_samples,
        'max_objects': max_objects_per_image,
        'use_orphaned_objects': vg_use_orphaned_objects,
        'include_relationships': include_relationships,
    }
    train_dset = VgSceneGraphDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    dset_kwargs['h5_path'] = val_h5
    del dset_kwargs['max_samples']
    val_dset = VgSceneGraphDataset(**dset_kwargs)

    # build dataloader
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 4,
        'shuffle': True,
        'collate_fn': vg_collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = shuffle_val
    loader_kwargs['num_workers'] = 1
    val_loader = DataLoader(val_dset, **loader_kwargs)

    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = get_dataloader(batch_size=32)

    # test reading data
    for i, batch in enumerate(train_loader):
        imgs, objs, boxes, masks, obj_to_img = batch

        print(imgs.shape, objs.shape, boxes.shape, masks.shape, obj_to_img.shape)

        if i == 20: break
