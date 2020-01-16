KALDI =

.PHONY: all clean

all: kaldi faster_rcnn

ifneq ($(strip $(KALDI)),)
kaldi:
	ln -s $(KALDI) kaldi
else
kaldi:
	test -d kaldi || git clone https://github.com/kaldi-asr/kaldi.git
	cd kaldi/tools; $(MAKE) all
	cd kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
endif

faster_rcnn:
	git clone https://github.com/jwyang/faster-rcnn.pytorch.git .
	cd faster-rcnn.pytorch && mkdir data || exit 1;
	pip install -r requirements.txt || exit 1;
	cd lib
	sh make.sh || exit 1;
	cd ../../..
	ln -s `pwd`/tools/faster-rcnn.pytorch/lib/model/nms scripts/model/. 
	ln -s `pwd`/tools/faster-rcnn.pytorch/lib/model/roi_align scripts/model/. 
	ln -s `pwd`/tools/faster-rcnn.pytorch/lib/model/roi_crop scripts/model/. 
	ln -s `pwd`/tools/faster-rcnn.pytorch/lib/model/roi_pooling scripts/model/. 

clean:
	rm -rf kaldi
	rm -rf tools/faster-rcnn.pytorch
