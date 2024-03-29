# STAGE: Spatio-Temporal Attention on Graph Entities
This repository contains the train and test code for the paper _[STAGE: Spatio-Temporal Attention on Graph Entities for Video Action Detection](https://arxiv.org/abs/1912.04316)_

<p align="center">
<img src="images/graph2.PNG" alt="STAGE" width=400 />
</p>

## Requirements
The required Python packages are:
* torch>=1.0.0
* h5py>=2.8.0
* tensorboardX>=1.6

## Features
In order to train and test the module, you need pre-computed actors and objects features coming from a pre-trained backbone on the [AVA dataset](https://research.google.com/ava/). Features must be organized in h5py files as follows:

**Actors features**
  
    actors_features_dir
          |
          |-> <clipID1_dir>
          |       |-> <timestamp1>.h5
          |       |-> <timestamp2>.h5
          |       |
          |
          |-> <clipID2_dir>
          |       |-> <timestamp1>.h5
          |       |-> <timestamp2>.h5
          |       |
          |
       
Each <timestamp>.h5 file should contain the following data:
  * "features" -> a torch tensor with shape (num_actors, feature_size, t, h, w) containing actors features
  * "boxes" -> a torch tensor with shape (num_actors, 4) containing bounding boxes coordinates for each actor
  * "labels" -> a torch tensor with shape (num_actors, 81) containing ones and zeros for performed/not performed actions
  
**Objects features**

    objects_features.h5
    
The file should contain the following data:
  * "\<clipID\>_\<timestamp\>_features" -> a torch tensor with shape (num_objects, feature_size) containing objects features
  * "\<clipID\>_\<timestamp\>_boxes" -> a torch tensor with shape (num_objects, 4) containing bounding boxes coordinates for each object
  * "\<clipID\>_\<timestamp\>_cls_prob" -> a torch tensor with shape (num_objects, num_classes) containing objects probabilities for each class
  
For example, the objects features of the clipID '-5KQ66BBWC4' at timestamp '902' will be in 
     
    objects_features["5KQ66BBWC4_902_features"]
    
I3D actors features are available at the following links:
  * [[I3D_actors_train]](https://drive.google.com/open?id=1RlciPLrEQcY0uYecS_cEWydrpvWg9DZv)
  * [[I3D_actors_val]](https://drive.google.com/open?id=1HCjezdcr2BkVUIEJgzBKPYSYLA0a9vxw)

Each tar.gz contains a directory, which corresponds to the "actors_features_dir" root. 

Faster-RCNN objects features are available at the following links:
  * [[Faster-RCNN_objects_train]](https://drive.google.com/file/d/13PrXvAR-Rw9MaTAJA5hInpJG4V_FLNuB/view?usp=sharing)
  * [[Faster-RCNN_objects_val]](https://drive.google.com/open?id=17_9NkM0kB_j0YEersD6y5WRPcKL6fiLp)

Each tar.gz contains an h5py file, which corresponds to the "objects_features.h5" file. 

The size of all the features is ~90 GB.

**SlowFast features**

You can find SlowFast features at the following links:
  * [[slowfast_ava2.1_32x2_features_train]](https://drive.google.com/file/d/1DW0b3Cc4d64P5Ir40cxpquGwYXkA_g-P/view?usp=sharing)
  * [[slowfast_ava2.1_32x2_features_val]](https://drive.google.com/file/d/1GbCLQ5jK8tk5FBj_DCKouQwyEApkkUfk/view?usp=sharing)
  * [[slowfast_ava2.2_32x2_features_train]](https://drive.google.com/file/d/1dV1F1wYDBl4M8BRp8_uKlGG4Vszzv9dH/view?usp=sharing)
  * [[slowfast_ava2.2_32x2_features_val]](https://drive.google.com/file/d/1C9m0DvE0rEyrmG36RggUCLlZOd2AqbVO/view?usp=sharing)
  * [[slowfast_ava2.2_64x2_features_train]](https://drive.google.com/file/d/1D-W7mJsWAt843GA0IwLaAD7qOpxIQESC/view?usp=sharing)
  * [[slowfast_ava2.2_64x2_features_val]](https://drive.google.com/file/d/1_Wd89_kQYtwL5IBmj7skJB0uVEUaW5V8/view?usp=sharing)
  
**Note**: These features are organized differently from I3D ones (you should write a specific dataloader or modify the provided one):
each 'h5py' file is a dictionary, each containing keys in the format "\<clip_number\>_\<timestamp\>". For each key, the corresponding value is another dictionary with keys "boxes", "features", "labels", containing actors' boxes coorindates, features extracted from the last SlowFast layer before classification and ground truth labels for that specific clip.


## Training

Run `python train.py` using the following arguments:

| Argument | Value |
|------|------|
| `--actors_dir` | Path to the train actors_features_dir |
| `--objects_file ` | Path to the train objects_features.h5 file |
| `--output_dir ` | Path to the directory where checkpoints will be stored |
| `--log_tensorboard_dir ` | Path to the directory where tensorboard logs will be stored |
| `--batch_size ` | The batch size. Must be > 1 to allow temporal connections |
| `--n_workers ` | The number of workers |
| `--lr ` | The learning rate |

For example, use:
```
python train.py --actors_dir "./actors_features_dir" --objects_file "./objects_features.h5" --output_dir "./out_checkpoints" --log_tensorboard_dir "./out_tensorboard" --batch_size 6 --n_workers 8 --lr 0.0000625 
```

## Testing

Run `python test.py` using the following arguments:

| Argument | Value |
|------|------|
| `--actors_dir` | Path to the val actors_features_dir |
| `--objects_file ` | Path to the val objects_features.h5 file |
| `--output_dir ` | Path to the directory where the checkpoint to load is stored |
| `--batch_size ` | The batch size. Must be > 1 to allow temporal connections |
| `--n_workers ` | The number of workers |

A "results.csv" file will be created under the "output_dir" directory, which should be used for evaluation as explained [here](https://research.google.com/ava/download.html)

For example, use:
```
python test.py --actors_dir "./actors_features_dir" --objects_file "./objects_features.h5" --output_dir "./out_checkpoints" --batch_size 6 --n_workers 8
```

