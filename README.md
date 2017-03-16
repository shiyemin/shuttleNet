# Video Action Recognition Toolkit with [TensorFlow](https://github.com/tensorflow/tensorflow)

*A TensorFlow implementation of the models described in [Shi et al. (2016)]
(https://arxiv.org/abs/1611.05216).* The code is modified based on [tensorflow/models]
(https://github.com/tensorflow/models).

Because that we are still tuning the model, the parameters are not exactly the same as our paper's
setting. But you should still be able to reproduce the experiments.

Training shuttleNet with GoogLeNet model needs 1-2 GPUs. However, it's recommended to have at least 4 GPUs to train shuttleNet with Inception-ResNet-v2 model.

The model used in our paper will be released soon. You can also ask for them before we actually make it available.

## Requirements
* Tensorflow (see tensorflow.org for installation instructions)

## Data
The data used to train this model is located
[here](http://crcv.ucf.edu/data/UCF101.php).

Convert the video with [dense_flow](https://github.com/yjxiong/dense_flow).

## Training the model

Convert the GoogLeNet with [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow).

```shell
# Where the pre-trained model is saved to.
npy_weights=models/googlenet_data.npy
```

To train the model, run the train_video_classifier.py file.
```shell
python train_video_classifier.py
```

There are several flags which can control the model that is trained, which are
exeplified below:
```shell
network=googlenet_rnn
train_batch_size=64
OUTPUT_DIR=UCF101_spatial_googlenet
DATASET_LIST=/home/shiyemin/data/ucf101/prepared_list/trainlist01_list.txt
DATASET_DIR=/home/shiyemin/data/ucf101/frames_tvl1
NUM_CLASSES=101
labels_offset=1
n_steps=16
modality=RGB
read_stride=5

GPU_ID=1,2,3,4
NUM_CLONES=4
train_steps=30000

CUDA_VISIBLE_DEVICES=$GPU_ID python ${ROOT}train_video_classifier.py \
  --mode=train \
  --num_clones=${NUM_CLONES} \
  --train_dir=${OUTPUT_DIR}/train \
  --dataset_list=${DATASET_LIST} \
  --dataset_dir=${DATASET_DIR} \
  --NUM_CLASSES=${NUM_CLASSES} \
  --n_steps=${n_steps} \
  --modality=${modality} \
  --read_stride=${read_stride} \
  --labels_offset=${labels_offset} \
  --resize_image_size=256 \
  --train_image_size=224 \
  --model_name=${network} \
  --npy_weights=${npy_weights} \
  --checkpoint_exclude_scopes=shuttleNet \
  --max_number_of_steps=${train_steps} \
  --batch_size=${train_batch_size} \
  --learning_rate_decay_type=piecewise \
  --learning_rate=0.01 \
  --decay_iteration=10000,20000,25000 \
  --learning_rate_decay_factor=0.1 \
  --label_smoothing=0.1 \
  --save_interval_secs=600 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=momentum \
  --weight_decay=0.00004
```

## Contact

To ask questions or report issues please open an issue on the shiyemin/shuttleNet
[issues tracker](https://github.com/shiyemin/shuttleNet/issues).

## Credits

This code was written by Yemin Shi.

You can also read more technical details in our [paper](https://arxiv.org/abs/1611.05216). If you use our code for research, please cite this paper as follows:

    @article{shi2016shuttlenet,
      title={shuttleNet: A biologically-inspired RNN with loop connection and parameter sharing},
      author={Shi, Yemin and Tian, Yonghong and Wang, Yaowei and Huang, Tiejun},
      journal={arXiv preprint arXiv:1611.05216},
      year={2016}
    }
