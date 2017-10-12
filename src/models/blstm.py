#!/usr/bin/python
# -*- coding: utf8

from src.base.models import SequentialModel
import tensorflow as tf
import numpy as np
from src.util.tf import tensorToSeq, seqToTensor, revlut
import math
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import json
import os

def xform(arr, words_vocab, chars_vocab, mxlen, maxw):
    """
    transforms a single feature vector into a feed dict for the model.
    """
    b = 0
    idx = 0

    xs_ch = np.zeros((mxlen, maxw), dtype=np.int)
    xs = np.zeros((mxlen), dtype=np.int)
    ys = np.zeros((mxlen), dtype=np.int)

    v = arr

    length = mxlen
    for j in range(mxlen):

        if j == len(v):
            length = j
            break

        w = v[j]
        nch = min(len(w), maxw)

        xs[j] = words_vocab.get(w, 0)
        for k in range(nch):
            xs_ch[j,k] = chars_vocab.get(w[k], 0)
    return {"x":[xs],"y":[ys], "xch": [xs_ch], "id": [idx], "length": [length] }

class BLSTM(SequentialModel):
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

    def predict(self, batch, xform=True):
        if not isinstance(batch, dict):
            batch = xform(batch, self.word_vocab, self.char_vocab, self.maxlen, self.maxw)

        lengths = batch["length"]
        feed_dict = {self.x: batch["x"], self.xch: batch["xch"], self.pkeep: 1.0}


        # We can probably conditionally add the loss here
        preds = []
        with tf.variable_scope(self.name):
            if self.crf is True:
                probv, tranv = self.sess.run([self.probs, self.A], feed_dict=feed_dict)

                for pij, sl in zip(probv, lengths):
                    unary = pij[:sl]
                    viterbi, _ = tf.contrib.crf.viterbi_decode(unary, tranv)
                    preds.append(viterbi)
            else:
                # Get batch (B, T)
                bestv = self.sess.run(self.best, feed_dict=feed_dict)
                # Each sentence, probv
                for pij, sl in zip(bestv, lengths):
                    unary = pij[:sl]
                    preds.append(unary)

        if xform:
            return [self.y_lut[i] for sent in preds for i in sent]
        else:
            return preds  #mostly to simplify training procedure.

    @classmethod
    def restore(cls, sess, indir, base, checkpoint_name=None):
        """
        this method NEEDS to know the base name used in training for the model.

        while i declare a variable scope, I still grab variables by names, so
        we see duplication in using the base name to get the variables out. It
        would be great to fix this at some point to be cleaner.
        """
        klass = cls(sess, base)
        basename = indir + '/' + base
        checkpoint_name = checkpoint_name or basename
        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)
            print('Loaded saver def')

        with gfile.FastGFile(basename + '.graph', 'r') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            print('Imported graph def')
            with tf.variable_scope(base):
                sess.run(saver_def.restore_op_name,
                         {saver_def.filename_tensor_name: checkpoint_name})
                klass.x = tf.get_default_graph().get_tensor_by_name(base + '/'+ 'x:0')
                klass.xch = tf.get_default_graph().get_tensor_by_name(base + '/'+ 'xch:0')
                klass.y = tf.get_default_graph().get_tensor_by_name(base + '/'+ 'y:0')
                klass.pkeep = tf.get_default_graph().get_tensor_by_name(base + '/'+ 'pkeep:0')
                klass.best = tf.get_default_graph().get_tensor_by_name(base + '/'+ 'output/ArgMax:0') # X
                klass.probs = tf.get_default_graph().get_tensor_by_name(base + '/'+ 'output/transpose:0') # X
                try:
                    klass.A = tf.get_default_graph().get_tensor_by_name(base + '/'+ 'Loss/transitions:0')
                    print('Found transition matrix in graph, setting crf=True')
                    klass.crf = True
                except:
                    print('Failed to get transition matrix, setting crf=False')
                    klass.A = None
                    klass.crf = False


        with open(basename + '.labels', 'r') as f:
            klass.labels = json.load(f)

        klass.word_vocab = {}
        if os.path.exists(basename + '-word.vocab'):
            with open(basename + '-word.vocab', 'r') as f:
                klass.word_vocab = json.load(f)

        with open(basename + '-char.vocab', 'r') as f:
            klass.char_vocab = json.load(f)

        with open(basename + '-params', 'r') as f:
            params = json.load(f)
            klass.maxlen = params['maxlen']
            klass.maxw = params['maxw']
            # self.name = params['model_name']


        klass.saver = tf.train.Saver(saver_def=saver_def)
        klass.y_lut = revlut(klass.labels)

        return klass

    def ex2dict(self, batch, pkeep):
        return {
            self.x: batch["x"],
            self.xch: batch["xch"],
            self.y: batch["y"],
            self.pkeep: pkeep
        }

    def params(self, labels, word_vec, char_vec, mxlen, maxw, rnntype, wsz, hsz, filtsz, crf=False):

        self.crf = crf
        char_dsz = char_vec.dsz
        nc = len(labels)
        self.x = tf.placeholder(tf.int32, [None, mxlen], name="x")
        self.xch = tf.placeholder(tf.int32, [None, mxlen, maxw], name="xch")
        self.y = tf.placeholder(tf.int32, [None, mxlen], name="y")
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")
        self.labels = labels
        self.y_lut = revlut(labels)
        
        self.word_vocab = {}
        if word_vec is not None:
            self.word_vocab = word_vec.vocab
        self.char_vocab = char_vec.vocab

        filtsz = [int(filt) for filt in filtsz.split(',') ]


        if word_vec is not None:
            with tf.name_scope("WordLUT"):
                Ww = tf.Variable(tf.constant(word_vec.weights, dtype=tf.float32), name = "W")

                we0 = tf.scatter_update(Ww, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, word_vec.dsz]))

                with tf.control_dependencies([we0]):
                    wembed = tf.nn.embedding_lookup(Ww, self.x, name="embeddings")

        with tf.name_scope("CharLUT"):
            Wc = tf.Variable(tf.constant(char_vec.weights, dtype=tf.float32), name = "W")

            ce0 = tf.scatter_update(Wc, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, char_dsz]))

            with tf.control_dependencies([ce0]):
                xch_seq = tensorToSeq(self.xch)
                cembed_seq = []
                for i, xch_i in enumerate(xch_seq):
                    cembed_seq.append(sharedCharWord(Wc, xch_i, maxw, filtsz, char_dsz, wsz, None if i == 0 else True))
                word_char = seqToTensor(cembed_seq)

            # List to tensor, reform as (T, B, W)
            # Join embeddings along the third dimension
            joint = word_char if word_vec is None else tf.concat( [wembed, word_char], 2)
            joint = tf.nn.dropout(joint, self.pkeep)

        with tf.name_scope("Recurrence"):
            embedseq = tensorToSeq(joint)

            if rnntype == 'blstm':
                rnnfwd = tf.contrib.rnn.BasicLSTMCell(hsz)
                rnnbwd = tf.contrib.rnn.BasicLSTMCell(hsz)

                # Primitive will wrap the fwd and bwd, reverse signal for bwd, unroll
                rnnseq, _, __ = tf.contrib.rnn.static_bidirectional_rnn(rnnfwd, rnnbwd, embedseq, dtype=tf.float32)
            else:
                rnnfwd = tf.nn.rnn_cell.BasicLSTMCell(hsz)
                # Primitive will wrap RNN and unroll in time
                rnnseq, _ = tf.nn.rnn(rnnfwd, embedseq, dtype=tf.float32)

        with tf.name_scope("output"):
            # Converts seq to tensor, back to (B,T,W)

            if rnntype == 'blstm':
                hsz *= 2

            W = tf.Variable(tf.truncated_normal([hsz, nc],
                                                stddev = 0.1), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[1,nc]), name="b")

            preds = [tf.matmul(rnnout, W) + b for rnnout in rnnseq]
            self.probs = seqToTensor(preds)
            self.best = tf.argmax(self.probs, 2)

            print(joint.get_shape())
            print(W.get_shape())
            print(rnnseq[0].get_shape())
            print(rnnseq[1].get_shape())
            print(self.probs.get_shape())
            # going back to sparse representation

def writeEmbeddingsTSV(word_vec, filename):
    idx2word = revlut(word_vec.vocab)
    with codecs.open(filename, 'w') as f:
        wrtr = UnicodeWriter(f, delimiter='\t', quotechar='"')

#        wrtr.writerow(['Word'])
        for i in range(len(idx2word)):
            row = idx2word[i]
            wrtr.writerow([row])

def _vizEmbedding(proj_conf, emb, outdir, which):
    emb_conf = proj_conf.embeddings.add()
    emb_conf.tensor_name = '%s/W' % which
    emb_conf.metadata_path = outdir + "/train/metadata-%s.tsv" % which
    writeEmbeddingsTSV(emb, emb_conf.metadata_path)

def vizEmbeddings(char_vec, word_vec, outdir, train_writer):
    print('Setting up word embedding visualization')
    proj_conf = projector.ProjectorConfig()
    _vizEmbedding(proj_conf, char_vec, outdir, 'CharLUT')
    if word_vec is not None:
        _vizEmbedding(proj_conf, word_vec, outdir, 'WordLUT')
    projector.visualize_embeddings(train_writer, proj_conf)

def charWordConvEmbeddings(char_vec, maxw, filtsz, char_dsz, wsz, padding):

    expanded = tf.expand_dims(char_vec, -1)

    mots = []
    for i, fsz in enumerate(filtsz):
        with tf.variable_scope('cmot-%s' % fsz):

            siglen = maxw + padding - fsz + 1
            kernel_shape =  [fsz, char_dsz, 1, wsz]
            
            # Weight tying
            W = tf.get_variable("W", kernel_shape, initializer=tf.random_normal_initializer())
            b = tf.get_variable("b", [wsz], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(expanded, 
                                W, strides=[1,1,1,1], 
                                padding="VALID", name="conv")
                
            activation = tf.nn.relu(tf.nn.bias_add(conv, b), "activation")

            mot = tf.nn.max_pool(activation,
                                 ksize=[1, siglen, 1, 1],
                                 strides=[1,1,1,1],
                                 padding="VALID",
                                 name="pool")
            mots.append(mot)
            
    wsz_all = wsz * len(mots)
    combine = tf.reshape(tf.concat(mots, 3), [-1, wsz_all])

    # Make a skip connection

#    with tf.name_scope("proj"):
    with tf.variable_scope("proj"):

        W_p = tf.get_variable("W_p", [wsz_all, wsz_all], initializer=tf.random_normal_initializer())
        b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
        proj = tf.nn.relu(tf.matmul(combine, W_p) + b_p, "proj")

    joined = combine + proj
    return joined


def sharedCharWord(Wch, xch_i, maxw, filtsz, char_dsz, wsz, reuse):

    with tf.variable_scope("SharedCharWord", reuse=reuse):
        # Zeropad the letters out to half the max filter size, to account for
        # wide convolution.  This way we don't have to explicitly pad the
        # data upfront, which means our Y sequences can be assumed not to
        # start with zeros
        mxfiltsz = np.max(filtsz)
        halffiltsz = int(math.floor(mxfiltsz / 2))
        #zeropad = tf.Print(xch_i, [tf.shape(xch_i)])
        zeropad = tf.pad(xch_i, [[0,0], [halffiltsz, halffiltsz]], "CONSTANT")
        cembed = tf.nn.embedding_lookup(Wch, zeropad)
        #cembed = tf.nn.embedding_lookup(Wch, xch_i)
        if len(filtsz) == 0 or filtsz[0] == 0:
            return tf.reduce_sum(cembed, [1])
        return charWordConvEmbeddings(cembed, maxw, filtsz, char_dsz, wsz, 2*halffiltsz)
