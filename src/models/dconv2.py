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
from collections import defaultdict

from src.models.initializers import identity_initializer, orthogonal_initializer

def _xform(arr, words_vocab, chars_vocab, mxlen, maxw):
    """
    transforms a single feature vector into a feed dict for the model.
    """
    batch = defaultdict(list)
    for i in arr:

	    xs_ch = np.zeros((mxlen, maxw), dtype=np.int)
	    xs = np.zeros((mxlen), dtype=np.int)
	    ys = np.zeros((mxlen), dtype=np.int)

	    v = i

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
	    batch['x'].append(xs)
	    batch['y'].append(ys)
	    batch['xch'].append(xs_ch)
	    batch['id'].append(i)
	    batch['length'].append(length)
    return batch


class DConv(SequentialModel):
    def __init__(self, sess, name, version='1'):
        self.sess = sess
        self.name = name
        self.version = version

    def predict(self, batch, xform=True, training_phase=False, word_keep=1.0):
        if not isinstance(batch, dict):
            batch = _xform(batch, self.word_vocab, self.char_vocab, self.maxlen, self.maxw)

        lengths = batch["length"]
        feed_dict = {self.x: batch["x"], 
                    self.xch: batch["xch"], 
                    self.pkeep: 1.0, 
                    self.word_keep: 1.0,
                    self.phase: training_phase}


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
            # print(preds)
            return [[self.y_lut[i] for i in sent] for sent in preds]
        else:
            return preds

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
                klass.word_keep = tf.get_default_graph().get_tensor_by_name(base + '/'+ 'word_keep:0')
                klass.phase = tf.get_default_graph().get_tensor_by_name(base + '/'+ 'phase:0')

                klass.best = tf.get_default_graph().get_tensor_by_name('output/ArgMax:0') # X
                klass.probs = tf.get_default_graph().get_tensor_by_name('output/transpose:0') # X
                try:
                    klass.A = tf.get_default_graph().get_tensor_by_name(base + '/'+ 'Loss/block/transitions:0')
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

    def ex2dict(self, batch, pkeep, phase, word_keep):
        return {
            self.x: batch["x"],
            self.xch: batch["xch"],
            self.y: batch["y"],
            self.pkeep: pkeep,
            self.word_keep: word_keep,
            self.phase: phase
        }

    # def identity_init():
    #     middle0 = int(shape[0] / 2)
    #     middle1 = int(shape[1] / 2)
    #     if shape[2] == shape[3]:
    #         array = np.zeros(shape, dtype='float32')
    #         identity = np.eye(shape[2], shape[3])
    #         array[middle0, middle1] = identity
    #     else:
    #         m1 = divisor / shape[2]
    #         m2 = divisor / shape[3]
    #         sigma = eps*m2
    #         array = np.random.normal(loc=0, scale=sigma, size=shape).astype('float32')
    #         for i in range(shape[2]):
    #             for j in range(shape[3]):
    #                 if int(i*m1) == int(j*m2):
    #                     array[middle0, middle1, i, j] = m2
    #     return tf.get_variable(name, initializer=array)

    def block(self, wembed, kernel_sz, num_filt, num_layers, reuse=False):
        # def bias_init():
        #     return tf.constant(0.0 if initialization == "identity" or initialization == "varscale" else 0.001
           
        dilation_rate = 2

        input_tensor = wembed
        with tf.variable_scope('iterated-block', reuse=reuse):
            for i in range(0, num_layers):
                if i == num_layers-1:
                    dilation_rate = 1
                input_tensor = tf.layers.conv1d(input_tensor, num_filt, kernel_sz, 
                                                padding='same', 
                                                dilation_rate=dilation_rate**i, 
                                                activation=None,
                                                kernel_initializer=tf.orthogonal_initializer(), #tf.contrib.layers.xavier_initializer(),
                                                reuse=reuse,
                                                name='conv-'+ str(i))
                tf.summary.histogram('conv-'+str(i), input_tensor)
                input_tensor = tf.nn.relu(input_tensor, name="relu-"+str(i))

            # input_tensor = tf.layers.conv1d(input_tensor, 
            #                         num_filt, 
            #                         kernel_sz, #kernel_sz
            #                         padding='same', 
            #                         dilation_rate=1, 
            #                         activation=tf.nn.relu,
            #                         kernel_initializer=tf.orthogonal_initializer(),
            #                         reuse=reuse,
            #                         name='conv-end')
            # input_tensor = tf.nn.dropout(input_tensor, self.pkeep)
            return input_tensor


    def params(self, labels, word_vec, char_vec, mxlen, 
                maxw, rnntype, wsz, hsz, filtsz, num_filt=64, 
                kernel_size=3, num_layers=4, num_iterations=3, 
                crf=False):

        self.crf = crf
        char_dsz = char_vec.dsz
        nc = len(labels)
        self.x = tf.placeholder(tf.int32, [None, mxlen], name="x")
        self.xch = tf.placeholder(tf.int32, [None, mxlen, maxw], name="xch")
        self.y = tf.placeholder(tf.int32, [None, mxlen], name="y")
        self.intermediate_probs = tf.placeholder(tf.int32, [None, mxlen, nc, num_iterations+2], name="y")
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")
        self.word_keep = tf.placeholder(tf.float32, name="word_keep")
        self.labels = labels
        self.y_lut = revlut(labels)
        self.phase = tf.placeholder(tf.bool, name="phase")
        
        self.word_vocab = {}
        if word_vec is not None:
            self.word_vocab = word_vec.vocab
        self.char_vocab = char_vec.vocab

        # if num_filt != nc:
        #     raise RuntimeError('number of filters needs to be equal to number of classes!')

        filtsz = [int(filt) for filt in filtsz.split(',') ]

        with tf.variable_scope('output/'):
            W = tf.Variable(tf.truncated_normal([num_filt, nc],
                                                stddev = 0.1), name="W")
            # W = tf.get_variable('W', initializer=tf.contrib.layers.xavier_initializer(), shape=[num_filt, nc])
            b = tf.Variable(tf.constant(0.0, shape=[1,nc]), name="b")

        intermediates = []

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
                    cembed_seq.append(shared_char_word(Wc, xch_i, filtsz, char_dsz, wsz, None if i == 0 else True))
                word_char = seqToTensor(cembed_seq)

        # List to tensor, reform as (T, B, W)
        # Join embeddings along the third dimension
        joint = word_char if word_vec is None else tf.concat([wembed, word_char], 2)
        joint = tf.nn.dropout(joint, self.word_keep)

        conv = tf.layers.conv1d(joint, 
                                num_filt, 
                                kernel_size, 
                                padding='same', 
                                dilation_rate=1,
                                kernel_initializer=tf.orthogonal_initializer(),
                                reuse=None,
                                name='conv-start')
        conv = tf.nn.relu(conv)

        # inter_preds = get_intermediate_probs(conv, num_filt, nc, W, b)
        # intermediates.append(inter_preds)

        # conv = self.block(conv, kernel_size, num_filt, num_layers, reuse=False)
        conv = tf.nn.dropout(conv, self.pkeep)

        for i in range(num_iterations):  # numIterations
            reuse = i != 0
            conv = self.block(conv, kernel_size, num_filt, num_layers, reuse=reuse)

            inter_preds = get_intermediate_probs(conv, num_filt, nc, W, b)
            intermediates.append(inter_preds)

            conv = tf.nn.dropout(conv, self.pkeep)

            

        

        # with tf.variable_scope('conv-end'):
        #     conv = tf.layers.conv1d(conv, 
        #                             num_filt, 
        #                             kernel_size, 
        #                             padding='same', 
        #                             dilation_rate=1, 
        #                             activation=tf.nn.relu,
        #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                             reuse=None,
        #                             name='conv-end')
        #     conv = tf.nn.relu(conv)
        #     conv = tf.nn.dropout(conv, self.pkeep)

        # inter_preds = get_intermediate_probs(conv, num_filt, nc, W, b)
        # intermediates.append(inter_preds)

        self.intermediate_probs = tf.stack(intermediates, axis=-1)


        out_seq = tensorToSeq(conv)
        with tf.variable_scope("output/"):
            # Converts seq to tensor, back to (B,T,W)
            
            # W = tf.get_variable('W')
            # b = tf.get_variable('b')

            # log(W)
            preds = [tf.matmul(word, W) + b for word in out_seq]
            self.probs = seqToTensor(preds)
            # log(self.probs)
            self.best = tf.argmax(self.probs, 2)
            # going back to sparse representation
            # self.probs = tf.matmul(conv, W) + b
            # self.best = tf.argmax(self.probs, 2)
        

def get_intermediate_probs(tensor, num_filt, nc, W, b):
    with tf.variable_scope('intermediate'):
        out_seq = tensorToSeq(tensor)
        # W = tf.Variable(tf.truncated_normal([num_filt, nc],
        #                                         stddev = 0.1), name="W")
        # b = tf.Variable(tf.constant(0.0, shape=[1,nc]), name="b")

        # log(W)
        preds = [tf.matmul(word, W) + b for word in out_seq]
        probs = seqToTensor(preds)
        # probs = tf.matmul(tensor, W) + b

        return probs

def log(tensor):
        print(tensor)

def highway_conns(inputs, wsz_all, n):
    for i in range(n):
        with tf.variable_scope("highway-%d" % i):
            W_p = tf.get_variable("W_p", [wsz_all, wsz_all])
            b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
            proj = tf.nn.relu(tf.matmul(inputs, W_p) + b_p, "relu-proj")

            W_t = tf.get_variable("W_t", [wsz_all, wsz_all])
            b_t = tf.get_variable("B_t", [1, wsz_all], initializer=tf.constant_initializer(-2.0))
            transform = tf.nn.sigmoid(tf.matmul(inputs, W_t) + b_t, "sigmoid-transform")

        inputs = tf.multiply(transform, proj) + tf.multiply(inputs, 1 - transform)
    return inputs

def skip_conns(inputs, wsz_all, n, reuse):
    for i in range(n):
        with tf.variable_scope("skip-%d" % i, reuse=reuse):
            W_p = tf.get_variable("W_p", [wsz_all, wsz_all])
            b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
            proj = tf.nn.relu(tf.matmul(inputs, W_p) + b_p, "relu")

        inputs = inputs + proj
    return inputs

def char_word_conv_embeddings(char_vec, filtsz, char_dsz, wsz, reuse):
    """
    char_vec: 
    filtsz: string of comma separated filter sizes. "1,2,3,"

    """
    expanded = tf.expand_dims(char_vec, -1)
    mots = []
    for i, fsz in enumerate(filtsz):
        with tf.variable_scope('cmot-%s' % fsz, reuse=reuse):

            kernel_shape = [fsz, char_dsz, 1, wsz]
            
            # Weight tying
            W = tf.get_variable("W", kernel_shape)
            b = tf.get_variable("b", [wsz], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(expanded, 
                                W, strides=[1,1,1,1], 
                                padding="VALID", name="conv")
                
            activation = tf.nn.relu(tf.nn.bias_add(conv, b), "activation")

            mot = tf.reduce_max(activation, [1], keep_dims=True)
            # Add back in the dropout
            mots.append(mot)
            
    wsz_all = wsz * len(mots)
    combine = tf.reshape(tf.concat(values=mots, axis=3), [-1, wsz_all])

    # joined = highway_conns(combine, wsz_all, 1)
    joined = skip_conns(combine, wsz_all, 1, reuse)
    return joined


def shared_char_word(Wch, xch_i, filtsz, char_dsz, wsz, reuse):

    with tf.variable_scope("SharedCharWord", reuse=reuse):
        # Zeropad the letters out to half the max filter size, to account for
        # wide convolution.  This way we don't have to explicitly pad the
        # data upfront, which means our Y sequences can be assumed not to
        # start with zeros
        mxfiltsz = np.max(filtsz)
        halffiltsz = int(math.floor(mxfiltsz / 2))
        zeropad = tf.pad(xch_i, [[0,0], [halffiltsz, halffiltsz]], "CONSTANT")
        cembed = tf.nn.embedding_lookup(Wch, zeropad)
        if len(filtsz) == 0 or filtsz[0] == 0:
            return tf.reduce_sum(cembed, [1])
    return char_word_conv_embeddings(cembed, filtsz, char_dsz, wsz, reuse)

def tensor2seq(tensor):
    return tf.unstack(tf.transpose(tensor, perm=[1, 0, 2]))


def seq2tensor(sequence):
    return tf.transpose(tf.stack(sequence), perm=[1, 0, 2])