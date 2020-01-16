export KALDI_ROOT=`pwd`/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export PATH=/home/hzili1/tools/anaconda3/envs/rpnsd/bin:$PATH
CUDAROOT=/opt/NVIDIA/cuda-9.0
CUDNNROOT=/home/yfujita/cudnn-7.5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDAROOT/lib64:$CUDNNROOT/lib64:$CUDNNROOT/include
