Fork of [PyTorch Deepspeech](https://github.com/SeanNaren/deepspeech.pytorch), as the trained models are for v1.1, the main branch in this repo is v1.1 and that tag is deleted.  
Does not depend on torchaudio, uses librosa directly.  

Key points are good implementation of [Baidu Warp-CTC](https://github.com/baidu-research/warp-ctc), good speed with 0.5x on cpu for audio even without openmp.  


# Installation

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu.

Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

Install this fork for Warp-CTC bindings:
```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
CC=gcc-8 CXX=gcc-8 make
cd ../pytorch_binding
CC=gcc-8 CXX=gcc-8 python setup.py install
```
in worst cases use, MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install. osx use clang which is not gcc so openmp and stdc99 options do not work out of box without additional work. With CC being set, it switches to gcc


If you want decoding to support beam search with an optional language model, install ctcdecode:
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
```

Finally clone this repo and run this within the repo:
```
pip install -r requirements.txt
```

# Usage
## Download Model  
```
mkdir model && cd model
wget https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v1.1/librispeech_pretrained.pth
```

## Transcribe  
```
python transcribe.py --model_path ./models/librispeech_pretrained.pth --audio_path /vz/wip/ai/deepspeech/training/hindi-splits/1499bf7e-9220-416c-b539-12699f91a14a-sha1-bf8c0f48d53ef4aaee7e1cc59fc5ad257511514f-0.wav
```

## Dataset
For dataset section refer to original repo [PyTorch Deepspeech](https://github.com/SeanNaren/deepspeech.pytorch)

### Custom Dataset
```
/path/to/audio1.wav,actual transcript 1
/path/to/audio2.wav,actual transcript 2
...
```
Yes the format is different from original.

### Merging multiple manifest files

To create bigger manifest files (to train/test on multiple datasets at once) we can merge manifest files together like below from a directory
containing all the manifests you want to merge. You can also prune short and long clips out of the new manifest.

```
cd data/
python merge_manifests.py --output_path merged_manifest.csv --merge_dir all_manifests/ --min_duration 1 --max_duration 15 # durations in seconds
```

## Training

Assuming tensorboard and [tensorboardx](https://github.com/lanpa/tensorboard-pytorch) support to visualise, point tensorboard to logdir 

```
python train.py --tensorboard --log_dir ./log_dir/ --train_manifest /vz/wip/ai/deepspeech/training/lm-audio-samples/pytorch/cv_valid_train.csv --val_manifest /vz/wip/ai/deepspeech/training/lm-audio-samples/pytorch/cv_valid_dev.csv --continue_from ./models/librispeech_pretrained.pth --batch_size 8 --epochs 8
```

For both visualisation tools, you can add your own name to the run by changing the `--id` parameter when training.

### Noise Augmentation/Injection
Refer original [readme](https://github.com/SeanNaren/deepspeech.pytorch/tree/v1.1)
```
python noise_inject.py --input_path /path/to/input.wav --noise_path /path/to/noise.wav --output_path /path/to/input_injected.wav --noise_level 0.5 # higher levels means more noise
```

### Checkpoints


To enable checkpoints every N batches through the epoch as well as epoch saving:

```
python train.py --checkpoint --checkpoint_per_batch N 
# N is the number of batches to wait till saving a checkpoint at this batch.
```

Note for the batch checkpointing system to work, you cannot change the batch size when loading a checkpointed model from it's original training
run.

To continue from a checkpointed model that has been saved:

```
python train.py --continue_from models/deepspeech_checkpoint_epoch_N_iter_N.pth
```

This continues from the same training state as well as recreates the visdom graph to continue from if enabled.

If you would like to start from a previous checkpoint model but not continue training, add the `--finetune` flag to restart training
from the `--continue_from` weights.

### Choosing batch sizes

For finding the batch size on the cuda h/w use 
```
python benchmark.py --batch_size 32
```


### Model details

Saved models contain the metadata of their training process. To see the metadata run the below command:

```
python model.py --model_path models/deepspeech.pth.tar
```

To also note, there is no final softmax layer on the model as when trained, warp-ctc does this softmax internally. This will have to also be implemented in complex decoders if anything is built on top of the model, so take this into consideration!

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py --model_path models/deepspeech.pth.tar --test_manifest /path/to/test_manifest.csv --cuda
```

An example script to output a transcription has been provided:

```
python transcribe.py --model_path models/deepspeech.pth.tar --audio_path /path/to/audio.wav
```

### Alternate Decoders
By default, `test.py` and `transcribe.py` use a `GreedyDecoder` which picks the highest-likelihood output label at each timestep. Repeated and blank symbols are then filtered to give the final output.

A beam search decoder can optionally be used with the installation of the `ctcdecode` library as described in the Installation section. The `test` and `transcribe` scripts have a `--decoder` argument. To use the beam decoder, add `--decoder beam`. The beam decoder enables additional decoding parameters:
- **beam_width** how many beams to consider at each timestep
- **lm_path** optional binary KenLM language model to use for decoding
- **alpha** weight for language model
- **beta** bonus weight for words

### Time offsets

Use the `--offsets` flag to get positional information of each character in the transcription when using `transcribe.py` script. The offsets are based on the size
of the output tensor, which you need to convert into a format required.
For example, based on default parameters you could multiply the offsets by a scalar (duration of file in seconds / size of output) to get the offsets in seconds.

## Pre-trained models

Pre-trained models can be found under releases [here](https://github.com/SeanNaren/deepspeech.pytorch/releases).

## Acknowledgements

Thanks to [SeanNaren](https://github.com/SeanNaren) for the original repo and the people he mentioned, [Egor](https://github.com/EgorLakomkin) and [Ryan](https://github.com/ryanleary) for their contributions!
