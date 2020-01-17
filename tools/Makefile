KALDI =

.PHONY: all clean

all: kaldi faster_rcnn

ifneq ($(strip $(KALDI)),)
kaldi:
	ln -s $(KALDI) kaldi
	ln -s `pwd`/kaldi/egs/wsj/s5/steps ../.
	ln -s `pwd`/kaldi/egs/wsj/s5/utils ../.
else
kaldi:
	test -d kaldi || git clone https://github.com/kaldi-asr/kaldi.git
	cd kaldi/tools; $(MAKE) all
	cd kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
endif

faster_rcnn:
	git clone https://github.com/jwyang/faster-rcnn.pytorch.git
	pip install -r faster-rcnn.pytorch/requirements.txt
	cd faster-rcnn.pytorch/lib && sh make.sh
	ln -s `pwd`/faster-rcnn.pytorch/lib/model/nms ../scripts/model/.
	ln -s `pwd`/faster-rcnn.pytorch/lib/model/roi_align ../scripts/model/.
	ln -s `pwd`/faster-rcnn.pytorch/lib/model/roi_crop ../scripts/model/.
	ln -s `pwd`/faster-rcnn.pytorch/lib/model/roi_pooling ../scripts/model/.

clean:
	rm -rf kaldi
	rm -rf faster-rcnn.pytorch
