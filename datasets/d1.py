from torch.utils.data import Dataset
from typing import List
import os
import json
from easydict import EasyDict
from collections import defaultdict
import numpy as np
import cv2
import pandas as pd
import logging

class d1_dataset(Dataset):
    features: List[str]
    target: str
    mode: str
    imgs_train_ids: List[str]

    def __init__(self, root_:str):

        self.root_ = root_
        df_train= pd.read_parquet(os.path.join(root_, 'train.parquet'))
        imgs_train_ids = df_train['product_id'].apply(lambda x: str(x)).values
        train_text_attributes=df_train['text_fields'].apply(lambda x: EasyDict(json.loads(x))).values

        self.features = list(train_text_attributes[0].keys())
        self.target = 'category_name'

        train_targets = df_train['category_name'].values

        data_dict=  defaultdict(list)
        for attributes_ in train_text_attributes:
            obj= EasyDict(attributes_)
            for k,v in obj.items():
                data_dict[k].append(v)
        self.features = list(data_dict.keys())

        data_dict = data_dict
        
        all = pd.DataFrame(data=data_dict)
        # get category levels
        splitted = [path_.split('->') for path_ in train_targets]
        splitted = [el[1:] for el in splitted]
        max_len_ = max([len(x) for x in splitted])
        cat_dict = defaultdict(list)
        for row in splitted:
            for j in range(len(row)):
                cat_dict[f'cat_{j}'].append(row[j])
            for j in range(len(row), max_len_):
                cat_dict[f'cat_{j}'].append(None)
        cat_dict = pd.DataFrame(data=cat_dict)
        self.cat_levels  = list(cat_dict.keys())
        all = pd.concat([all, cat_dict],axis=1)
        all['product_id'] = imgs_train_ids 
        all[self.target] = ['->'.join(x) for x in splitted]
        self.all = all

        self.text_mode()
        
    def __getitem__(self, idx:int):
        row = self.all.iloc[idx:idx+1]
        
        if self.mode == 'text':
            text_ = row[self.features].values[0]
            target_ = row[self.target].values[0]
            return text_,target_
        elif self.mode == 'img':
            img_ = cv2.cvtColor(cv2.imread(os.path.join(self.root_, 'images', row['product_id'].values[0]+'.jpg')),
                            cv2.COLOR_RGB2BGR)
            target_ = row[self.target].values[0]
            return img_,target_
        elif self.mode == 'multi':
            text_ = row[self.features].values[0]
            img_ = cv2.cvtColor(cv2.imread(os.path.join(self.root_, 'images', row['product_id'].values[0]+'.jpg')),
                            cv2.COLOR_RGB2BGR)
            target_ = row[self.target].values[0]
            return text_,img_,target_


    def __len__(self):
        return len(self.imgs_train_ids)
                
    def text_mode(self):
        self.mode = 'text'
    def img_mode(self):
        self.mode = 'img'
    def multi_mode(self):
        self.mode = 'multi'

    def get_view(self, idx):
        try:
            import matplotlib.pyplot as plt
        except:
            logging.error('matplotlib is not installed')
            return
        if self.mode != 'multi':
            print(f'mode {self.mode} not supported yet')
        text,img,target = self[idx]
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(img)
        def add_new_lines(text_, max_line_length):
            o_ = ''
            n_ = len(text_)
            for i in range(int(n_/max_line_length)+1):
                o_ += text_[i*max_line_length:(i+1)*max_line_length] + '\n'
            return o_
        title_ = text[0]
        desc_ = text[1]
        final_ = 'title: '+add_new_lines(title_, 50) +'\n'+'desc: '+ add_new_lines(desc_, 50)

        ax.set_xlabel(final_,fontsize=10)
        ax.set_title('target:'+add_new_lines(target,50),fontsize=10)
        return fig,ax


class d1_adapter:
    def __init__(self):
        pass
    def __call__(self, text):
        # ['title', 'description', 'attributes', 'custom_characteristics', 'defined_characteristics', 'filters']
        if len(text.shape)==2:
            return np.asarray([' '.join(el) for el in text[:, :2]],dtype=object)
        elif len(text.shape)==1:
            return ' '.join(text[:2])
        else:
            logging.error(f'input with dims {text.shape} not supprted')
            return None



