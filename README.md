# Contextual Prompt Learning








## Environment Setup
Refer to `requirements.txt` for installing all python dependencies. 

## Supervised Training

`train.py` 

### Dataset Preparation

We download the official version of Kinetics-400 from [here](https://github.com/cvdfoundation/kinetics-dataset) and videos are resized using code [here](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics).

We expect that `--train_list_path` and `--val_list_path` command line arguments to be a data list file of the following format
```
<path_1>,<label_1>
<path_2>,<label_2>
...
<path_n>,<label_n>
```
where `<path_i>` points to a video file, and `<label_i>` is an integer between `0` and `num_classes - 1`.
`--num_classes` should also be specified in the command line argument.

Additionally, `<path_i>` might be a relative path when `--data_root` is specified, and the actual path will be
relative to the path passed as `--data_root`.

The class mappings in the open-source weights are provided at [Kinetics-400 class mappings](data/k400_class_mappings.json)

### Download Pretrained CLIP Checkpoint

Download the pretrained CLIP checkpoint 

### Training Instruction

For supervised training on the Kinetics-400 dataset, use the train script in the `train_scripts` directory. Modify the `--train_list_path`, `--train_list_val` according to the data location and modify the `--backbone_path` according to location where the pretrained checkpoint was downloaded and stored.

## Pretrained Models




## Acknowledgements
Our code is based on [XCLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP)  and [CoOp](https://github.com/KaiyangZhou/CoOp) repositories. We thank the authors for releasing their code. If you use our model, please consider citing these works as well.