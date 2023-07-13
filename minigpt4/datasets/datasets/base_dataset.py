"""
 Copyright (c) 2022, salesforce.com, inc. 
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset 
from torch.utils.data.dataloader import default_collate


class BaseDataset(Dataset):
    def __init__(
        self, 
        vis_processor=None, 
        text_processor=None, 
        vis_root=None, 
        ann_paths=[], 
        labels=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        labels (list): 存储每个样本的匹配标签,1表示匹配,0表示不匹配
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        # 新增labels属性
        self.labels = labels

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)
            
    # 在__getitem__中返回匹配标签        
    def __getitem__(self, index):
        vis_data = self.vis_processor(self.annotation[index], self.vis_root)
        text_data = self.text_processor(self.annotation[index])
        
        output = {
            "vis_data": vis_data,
            "text_data": text_data,
            "ann": self.annotation[index],
            "label": self.labels[index]
        }
        
        return output


class ConcatDataset(ConcatDataset):

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.cummulative_sizes = np.cumsum([0] + [len(d) for d in datasets])
    
    def set_processors(self, vis_processor, text_processor):
        for d in self.datasets:
            if isinstance(d, BaseDataset):
                d.set_processors(vis_processor, text_processor)
