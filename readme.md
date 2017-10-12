convolutional models for sequence tagging
---

This repo is a collection of Tensorflow models meant for sequence tagging.

### models

blstm -- your run-of-the-mill bidirectional lstm model with char embeddings.  
dconv2 -- dilated convolutional model using conv1d  
dconv3 -- dilated convolutional model using conv2d, rewritten from emma strubell's paper https://arxiv.org/pdf/1702.02098.pdf  
Strubells -- dilated convolutional model from emma's paper. This model was a copy/paste job, only changing what needed to be changed to fit into my coding paradigm.  
attentive_conv -- WIP, based on https://arxiv.org/pdf/1710.00519.pdf  

### refs
https://github.com/dpressel/baseline

https://github.com/iesl/dilated-cnn-ner
