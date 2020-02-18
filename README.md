# RPNSD
PyTorch implementation of RPNSD. Our code is largely based on a Faster R-CNN implementation [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) by [jwyang](https://github.com/jwyang).

## Install
1. Clone this project
```bash
git clone https://github.com/HuangZiliAndy/RPNSD.git
cd RPNSD
```
2. Add your Python path to `PATH` variable in `path.sh`, the current default is `~/anaconda3/bin`.
3. Install PyTorch (0.4.0) and torchvision according to your CUDA version
```
conda install pytorch==0.4.0 cuda91 torchvision pillow"<7" -c pytorch
```
4. Install the packages in requirements.txt
```bash
pip install -r requirements.txt
```
5. Prepare Kaldi and Faster R-CNN library (You can specify a Kaldi root if you already have it)
```bash
cd tools
make KALDI=<path/to/a/compiled/kaldi/directory>
```
6. Set your backend computing environment to `cmd.sh`
```
# Select the backend used by run.sh from "local", "sge", "slurm", or "ssh"
cmd_backend='local'
```
## Data preparation
The purpose of this step includes
1. Prepare a large diarization dataset with Mixer6, SRE and SWBD. The majority of the dataset is two-channel telephone conversation of two people. We sum up the channels to create diarization style training data.
2. Prepare test set with CALLHOME dataset. Since the CALLHOME dataset doesn't specify train/dev/test, we use 5 folds cross validation.

```bash
./run_prepare_shared.sh
```

## Train
Training on the Mixer6 + SRE + SWBD dataset. Default setting uses single GPU and takes about 4 days.
```bash
./train.sh
```
Pretrained model is available at [pretrain-model](https://drive.google.com/file/d/1EYhTADveeeMlu2J3AqzkITcKXZhbNmUa/view?usp=sharing).

## Adapt
Adapt the model on in-domain data. Since we use 5 folds cross validation, each time we train on 400 utterances from CALLHOME dataset and test on 100.
```bash
./adapt.sh
```

## Inference
Inference stage. 
1. Forward the network to get speech region proposals, speaker embedding and background probability.
2. Post-processing with clustering and NMS.
3. Compute Diarization Error Rate (DER).
```bash
./inference.sh
```

## Citation
    
    
    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    }

    @inproceedings{huang2020speaker,
        Title={Speaker Diarization with Region Proposal Network},
        Author={Huang, Zili and Watanabe, Shinji and Fujita, Yusuke and Garcia, Paola and Shao, Yiwen and Povey, Daniel and Khudanpur, Sanjeev},
        Booktitle={Accepted to ICASSP 2020},
        Year={2020}
    }
