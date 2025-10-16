# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:50:47 2021

@author: fakbarifar
"""

from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
from .utils import noisify

import scipy.signal as sps #FAR:
    
    
import random

from sklearn.model_selection import train_test_split

class KINARM_FEATS_3TASKS(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    
    raw_folder = 'raw/feats/3tasks'
    processed_folder = 'processed'
    training_file = 'training.pt'
    val_file = 'validation.pt'
    test_file = 'test.pt'

    def __init__(self, root, train='train', transform=None, target_transform=None, load=False,
                 noise_type=None, noise_rate=0.2, random_state=0, num_class=2):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='mnist'
        self.noise_type=noise_type
        self.num_class=num_class

                    
        # FAR:
        if load:
            self.load()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train=='train':
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))

            if noise_type != 'clean':
                self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state)
                self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                _train_labels=[i[0] for i in self.train_labels]
                self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)
        
        elif self.train=='val':
            self.val_data, self.val_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.val_file))
            
        elif self.train=='test':
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train=='train':
            #if self.noise_type is not None:
            if self.noise_type != 'clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        elif self.train=='val':
            img, target = self.val_data[index], self.val_labels[index]
        elif self.train=='test':
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='val':
            return len(self.val_data)
        elif self.train=='test':
            return len(self.test_data)

    def _check_exists(self): # FAR: returns True if both training.pt and test.pt files exist in their associated directories
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.val_file)) and \
                os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def load(self): #FAR:
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST: # FAR: if the directories already exist
                pass
            else:
                raise # FAR: throw an exception at any time. What does it mean here??

        # for url in self.urls:
        #     print('Downloading ' + url)
        #     data = urllib.request.urlopen(url)
        #     filename = url.rpartition('/')[2]
        #     file_path = os.path.join(self.root, self.raw_folder, filename)
        #     with open(file_path, 'wb') as f:
        #         f.write(data.read())
        #     with open(file_path.replace('.gz', ''), 'wb') as out_f, \
        #             gzip.GzipFile(file_path) as zip_f:
        #         out_f.write(zip_f.read())
        #     os.unlink(file_path)
        
        
        ###VGR+OH+OHA
        ###VGR
        nPATH = "D:\OneDrive - Queen's University\Co-teaching\data_time_exploration"
        read_vgr = np.load(os.path.join(nPATH,'VGR_binary_train,val,test_swap_indCMSA.npz'), allow_pickle=True)
        vgr_train_data_all = read_vgr['vgr_train_data_all']
        vgr_train_labels = read_vgr['vgr_train_labels']
        vgr_val_data_all = read_vgr['vgr_val_data_all']
        vgr_val_labels = read_vgr['vgr_val_labels']
        vgr_test_data_all = read_vgr['vgr_test_data_all']
        vgr_test_labels = read_vgr['vgr_test_labels']
        
        ###OH
        nPATH = "D:\OneDrive - Queen's University\Co-teaching\data_time_exploration"
        read_oh = np.load(os.path.join(nPATH,'OH_binary_train,val,test_swap_indCMSA.npz'), allow_pickle=True)
        oh_train_data_all = read_oh['oh_train_data_all']
        oh_train_labels = read_oh['oh_train_labels']
        oh_val_data_all = read_oh['oh_val_data_all']
        oh_val_labels = read_oh['oh_val_labels']
        oh_test_data_all = read_oh['oh_test_data_all']
        oh_test_labels = read_oh['oh_test_labels']
        
        ###OHA
        nPATH = "D:\OneDrive - Queen's University\Co-teaching\data_time_exploration"
        read_oha = np.load(os.path.join(nPATH,'OHA_binary_train,val,test_swap_indCMSA.npz'), allow_pickle=True)
        print('************************************************THESWAP')
        oha_train_data_all = read_oha['oha_train_data_all']
        oha_train_labels = read_oha['oha_train_labels']
        oha_val_data_all = read_oha['oha_val_data_all']
        oha_val_labels = read_oha['oha_val_labels']
        oha_test_data_all = read_oha['oha_test_data_all']
        oha_test_labels = read_oha['oha_test_labels']
        
        ###VGR
        indexes_vgr = np.concatenate([np.arange(6,20), np.arange(26,40)])
        train_data_vgr = vgr_train_data_all[:,indexes_vgr]
        val_data_vgr = vgr_val_data_all[:, indexes_vgr]
        test_data_vgr = vgr_test_data_all[:, indexes_vgr]
        
               
        ###OH
        indexes_oh = np.arange(6,20)
        train_data_oh = oh_train_data_all[:, indexes_oh]
        val_data_oh = oh_val_data_all[:, indexes_oh]
        test_data_oh = oh_test_data_all[:, indexes_oh]
        
        ###OHA
        indexes_oha = np.arange(6,26)
        train_data_oha = oha_train_data_all[:, indexes_oha]
        val_data_oha = oha_val_data_all[:, indexes_oha]
        test_data_oha = oha_test_data_all[:, indexes_oha]
        
        train_data = np.concatenate([train_data_vgr, train_data_oh, train_data_oha], axis=1)
        val_data = np.concatenate([val_data_vgr, val_data_oh, val_data_oha], axis=1)
        test_data = np.concatenate([test_data_vgr, test_data_oh, test_data_oha], axis=1)
        
        train_labels = np.concatenate([vgr_train_data_all[:, 0:6], np.expand_dims(vgr_train_labels, axis=1), np.expand_dims(oh_train_labels, axis=1)], axis=1)
        val_labels = np.concatenate([vgr_val_data_all[:, 0:6], np.expand_dims(vgr_val_labels, axis=1), np.expand_dims(oh_val_labels, axis=1)], axis=1)
        test_labels = np.concatenate([vgr_test_data_all[:, 0:6], np.expand_dims(vgr_test_labels, axis=1), np.expand_dims(oh_test_labels, axis=1)], axis=1)
        # train_labels = oh_train_labels
        # val_labels = oh_val_labels
        # test_labels = oh_test_labels
        
        
        #% Normalization, when using z-scores
        max_mat = np.max(train_data, axis=0)
        max_mat_train = np.tile(max_mat, (train_data.shape[0],1))
        min_mat = np.min(train_data, axis=0)
        min_mat_train = np.tile(min_mat, (train_data.shape[0],1))
        
        train_data = np.divide((train_data-min_mat_train), (max_mat_train-min_mat_train))
        
        max_mat_val = np.tile(max_mat, (val_data.shape[0],1))
        min_mat_val = np.tile(min_mat, (val_data.shape[0],1))
        val_data = np.divide((val_data-min_mat_val), (max_mat_val-min_mat_val))
        
        max_mat_test = np.tile(max_mat, (test_data.shape[0],1))
        min_mat_test = np.tile(min_mat, (test_data.shape[0],1))
        test_data = np.divide((test_data-min_mat_test), (max_mat_test-min_mat_test)) 
        
        
        
        
        train_data = torch.from_numpy(train_data)
        val_data = torch.from_numpy(val_data)
        test_data = torch.from_numpy(test_data)
        
        


        # process and save as torch files
        print('Processing...')

        # training_set = (
        #     read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
        #     read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        # )
        # test_set = (
        #     read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
        #     read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        # )
        # train_label = np.expand_dims(train_label[:, 1, 0], axis=1)
        # val_label = np.expand_dims(val_label[:, 1, 0], axis=1)
        # test_label = np.expand_dims(test_label[:, 1, 0], axis=1)
        
        if self.num_class!=2:
        
            train_cmsa = train_header[:, 1, -1]
            
            train_label = np.zeros(train_cmsa.shape)
            for ii in range(len(train_label)):
                train_label[ii] = float(train_cmsa[ii])
                
            
                
            val_cmsa = val_header[:, 1, -1]
            
            val_label = np.zeros(val_cmsa.shape)
            for ii in range(len(val_label)):
                val_label[ii] = float(val_cmsa[ii])
                
                
                
            test_cmsa = test_header[:, 1, -1]
            
            test_label = np.zeros(test_cmsa.shape)
            for ii in range(len(test_label)):
                test_label[ii] = float(test_cmsa[ii])

        ##################################################################torch.from_numpy(val_label[:, 1, 0]).view(-1,1)
        # train_label = torch.from_numpy(train_labels).view(-1,1)
        # val_label = torch.from_numpy(val_labels).view(-1,1)
        # test_label = torch.from_numpy(test_labels).view(-1,1)
        
        training_set = (train_data, train_labels)
        val_set = (val_data, val_labels)
        test_set = (test_data, test_labels)
        
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.val_file), 'wb') as f:
            torch.save(val_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        
        if self.train=='train':
            tmp = 'train' 
        elif self.train=='val':
            tmp = 'val' 
        elif self.train=='test':
            tmp = 'test'
            
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)
    
    
def data_resample(X): #FAR:
        X_rs = np.zeros(X.shape)
        X_rs = X_rs[:,:,:,:256]
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                x = sps.resample(X[i,:,j], 256, axis=1)
                X_rs[i,:,j]=x
        return X_rs    




