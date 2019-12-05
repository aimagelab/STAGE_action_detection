import torch
import torch.utils.data as data_utl
import h5py
import os
import random


def sort_function(filename):
    return (filename.split('/')[-2], int(filename.split('/')[-1].split('.')[-2]))

class AVADataset(data_utl.Dataset):

    def __init__(self, split='train', videodir='./train_features_I3D', objectsfile='./ava_objects_fasterrcnn.hdf5'):
        self.split = split
        self.objectsfile = objectsfile

        self.filenames = []
        for dirname in os.listdir(videodir):
            for filename in os.listdir(os.path.join(videodir, dirname)):
                self.filenames.append(os.path.join(videodir, dirname, filename))

        if self.split == "val":
            self.filenames.sort(key=sort_function)

    def __getitem__(self, index):
        filename = self.filenames[index]
        clip_id = filename.split('/')[-2]
        timestamp = filename.split('/')[-1].split('.')[0]

        hf_actors = h5py.File(filename, 'r')
        actors_features = torch.from_numpy(hf_actors.get("features").value)
        actors_labels = torch.from_numpy(hf_actors.get('labels').value)
        actors_boxes = torch.from_numpy(hf_actors.get('boxes').value)

        hf_objects = h5py.File(self.objectsfile, 'r')
        objects_features = torch.from_numpy(hf_objects.get(clip_id + '_' + timestamp.lstrip("0") + '_' + 'features').value)
        objects_boxes = torch.from_numpy(hf_objects.get(clip_id + '_' + timestamp.lstrip("0") + '_' + 'boxes').value)

        return actors_features, actors_labels, actors_boxes, [(clip_id, timestamp) for _ in range(actors_features.shape[0])], objects_features, objects_boxes, [(clip_id, timestamp) for _ in range(objects_features.shape[0])]

    def __len__(self):
        return len(self.filenames)

    def rotate(self, n):
        self.filenames = self.filenames[n:] + self.filenames[:n]

    def shuffle_filename_blocks(self, block_size, epoch):
        self.filenames.sort(key=sort_function)
        self.rotate(int(block_size/4)*(epoch-1))
        self.filenames = [self.filenames[i:i+block_size] for i in range(0,len(self.filenames),block_size)]
        random.shuffle(self.filenames)
        self.filenames[:] = [b for bs in self.filenames for b in bs]
