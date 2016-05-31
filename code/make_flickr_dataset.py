import pandas as pd
import numpy as np
import os
import cPickle
from cnn_util import *

vgg_model = '/home/kucho/library/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = '/home/kucho/library/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'

annotation_path = './data/results_20130124.token'
flickr_image_path = '../show_attend_and_tell/images/flickr30k-images/'
#feat_path = './data/feats.npy'
feat_path = './data/guitar.npy'

cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)

#ipdb.set_trace()

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))
print annotations['image'].values

annot = []
annot = ['/home/kucho/workspace/PycharmProjects/show_and_tell/show_and_tell.tensorflow/acoustic-guitar-player.jpg']
print annot

if not os.path.exists(feat_path):
    #feats = cnn.get_features(annotations['image'].values)
    feats = cnn.get_features(annot)
    np.save(feat_path, feats)
