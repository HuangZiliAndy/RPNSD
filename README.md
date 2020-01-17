# RPNSD
PyTorch implementation of RPNSD

## Install
1. Clone this project
```bash
git clone https://github.com/HuangZiliAndy/RPNSD.git
cd RPNSD
```
2. Install PyTorch (0.4.0) and torchvision
3. Install the packages in requirements.txt
```bash
pip install -r requirements.txt
```
4. Prepare Kaldi and Faster-RCNN library (You can specify a Kaldi root if you already have it)
```bash
cd tools
make KALDI=<path/to/a/compiled/kaldi/directory>
```
## Data preparation
```bash
./run_prepare_shared.sh
```

## Train
```bash
./train.sh
```

## Adapt
```bash
./adapt.sh
```

## Inference
```bash
./inference.sh
```
