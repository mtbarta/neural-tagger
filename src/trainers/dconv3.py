import tensorflow as tf
import numpy as np
import time
import math
from src.util.conll_util import batch
from src.util.tf import toSpans, fScore
import json
from tensorflow.contrib.tensorboard.plugins import projector
import codecs
import csv
import io
from sklearn.metrics import confusion_matrix


class Evaluator:
    
    def __init__(self, sess, model, idx2label, fscore):
        self.sess = sess
        self.model = model
        self.idx2label = idx2label
        self.fscore = fscore

    def _writeSentenceCONLL(self, handle, sentence, gold, txt):

        if len(txt) != len(sentence):
            txt = txt[:len(sentence)]

        try:
            for word, truth, guess in zip(txt, gold, sentence):
                handle.write('%s %s %s\n' % (word, self.idx2label[truth], self.idx2label[guess]))
            handle.write('\n')
        except:
            print('ERROR: Failed to write lines... closing file')
            handle.close()
            handle = None

    def _batch(self, batch, handle=None, txts=None):

        sentence_lengths = batch["length"]
        truth = batch["y"]
        feed_dict = self.model.ex2dict(batch, 1, False, 1)
        guess = self.model.predict(batch, xform=False, training_phase=False, word_keep=1.0)

        correct_labels = 0
        total_labels = 0

        # For fscore
        gold_count = 0
        guess_count = 0
        overlap_count = 0
        
        # For each sentence
        for b in range(len(guess)):
            length = sentence_lengths[b]
            assert(length == len(guess[b]), "lengths differ: length-- {}, len(guess[b])-- {} ".format(length, len(guess[b])))
            sentence = guess[b]
            # truth[b] is padded, cutting at :length gives us back true length
            gold = truth[b][:length]

            # cm = confusion_matrix(gold, sentence, labels=self.idx2label.keys())
            # below gives me a weird error:
            # unsupported operand type(s) for +=: 'int' and 'NotImplementedType'
            correct_labels += np.sum(np.equal(sentence, gold))
            total_labels += length

            if self.fscore > 0:
                gold_chunks = toSpans(gold, self.idx2label)
                gold_count += len(gold_chunks)

                guess_chunks = toSpans(sentence, self.idx2label)
                guess_count += len(guess_chunks)
            
                overlap_chunks = gold_chunks & guess_chunks
                overlap_count += len(overlap_chunks)

            # Should we write a file out?  If so, we have to have txts
            if handle is not None:
                idx = batch["id"][b]
                txt = txts[idx]
                self._writeSentenceCONLL(handle, sentence, gold, txt) 
                

        return correct_labels, total_labels, overlap_count, gold_count, guess_count

    def test(self, ts, batchsz=1, phase='Test', conll_file=None, txts=None):

        total_correct = total_sum = fscore = 0
        total_gold_count = total_guess_count = total_overlap_count = 0
        start_time = time.time()
    
        steps = int(math.floor(len(ts)/float(batchsz)))

        # Only if they provide a file and the raw txts, we can write CONLL file
        handle = None
        if conll_file is not None and txts is not None:
            handle = open(conll_file, "w")

        # total_cm = np.ndarray((len(self.idx2label), len(self.idx2label)))
        for i in range(steps):
            ts_i = batch(ts, i, batchsz)
            correct, count, overlaps, golds, guesses = self._batch(ts_i, handle, txts)
            total_correct += correct
            total_sum += count
            total_gold_count += golds
            total_guess_count += guesses
            total_overlap_count += overlaps
            # total_cm += cm

        duration = time.time() - start_time
        total_acc = total_correct / float(total_sum)


        # Only show the fscore if requested
        if self.fscore > 0:
            fscore = fScore(total_overlap_count,
                            total_gold_count,
                            total_guess_count,
                            self.fscore)
            print('%s (F%d = %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % 
                  (phase,
                   self.fscore,
                   float(fscore),
                   total_correct,
                   total_sum,
                   total_acc,
                   duration))

            # #show label specific metrics
            # for i in len(self.idx2label.keys()):
            #     label = self.idx2label[i]
            #     not_i = [x for x in self.idx2label.keys() if x != i]
            #     tp = total_cm[i,i]
            #     fp = np.sum(total_cm[i, not_i])
            #     fn = np.sum(total_cm[not_i, i])

            #     fscore_l = fScore(tp, tp+fn, tp+fp, 1)

            #     print ('%s -- %s (F%d = %.4f)' % (
            #         phase, label, self.fscore, fscore_l))
                        
        else:
            print('%s (Acc %d/%d = %.4f) (%.3f sec)' %
                  (phase,
                   total_correct,
                   total_sum, total_acc,
                   duration))

        if handle is not None:
            handle.close()

        return total_acc, fscore

class DConvTrainer:

    def __init__(self, sess, model, outdir):
        
        self.sess = None
        self.outdir = outdir
        
        self.model = model

        
    def saveUsing(self, saver):
        self.saver = saver
        
    def writer(self):
        return self.train_writer

    def checkpoint(self, name):
        self.saver.save(self.sess, self.outdir + "/train/" + name, global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint(self.outdir + "/train/")
        print("Reloading " + latest)
        self.saver.restore(self.sess, latest)

    def train(self, name, ts, f2i, vs, es,
                char_vec, 
                word_vec,
                eval_out,
                batchsz=50,
                epochs=100,
                dropout=0.5, 
                test_thresh=10, 
                patience=70, 
                rnn='blstm', 
                maxlen = 100,
                maxw = 40,
                wsz=30, 
                hsz=100, 
                cfiltsz='1,2,3,4,5,7', 
                optim='momentum', 
                eta=0.001,
                crf=False,
                fscore=1,
                viz=False,
                clip=5,
                kernel_size=3,
                num_layers=4,
                num_iterations=3,
                word_keep=0.85,
                num_filt=64):
        self.model_name = name
        self.maxlen = maxlen
        self.maxw = maxw

        #need to save into class for loss computation
        self.crf = crf
        try:
            with tf.Graph().as_default():
                self.sess = tf.Session()
                with self.sess.as_default():
                    tf.set_random_seed(1234)
                    with tf.variable_scope(name):
                        model = self.model(self.sess, name)
                        
                        model.params(f2i,
                                     word_vec,
                                     char_vec,
                                     maxlen,
                                     maxw,
                                     rnn,
                                     wsz,
                                     hsz,
                                     cfiltsz,
                                     kernel_size=kernel_size,
                                     num_layers=num_layers,
                                     num_iterations=num_iterations,
                                     num_filt=num_filt,
                                     crf=crf)


                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                        self.evaluator = Evaluator(self.sess, model, model.y_lut, fscore)
                        self.loss = self.createLoss(model)
            
                        self.global_step = tf.Variable(0, name='global_step', trainable=False)
                        tvars = tf.trainable_variables()
                        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clip)

                        with tf.control_dependencies(update_ops):
                            if optim == 'adadelta':
                                self.optimizer = tf.train.AdadeltaOptimizer(eta, 0.95, 1e-6)
                            elif optim == 'adam':
                                self.optimizer = tf.train.AdamOptimizer(eta, beta2=.9)
                            elif optim == 'sgd':
                                self.optimizer = tf.train.GradientDescentOptimizer(eta)
                            else:
                                self.optimizer = tf.train.MomentumOptimizer(eta, 0.9)

                            # self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
                        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars),
                                                                    global_step=self.global_step)

                        self.f1 = tf.Variable(0.0, name='f1')
                        self.f1_summary = tf.summary.scalar("f1", self.f1)
                        self.loss_summary = tf.summary.scalar("loss", self.loss)
                        self.summary_op = tf.summary.merge_all()
                        self.train_writer = tf.summary.FileWriter(self.outdir + "/train", self.sess.graph)

                        if viz:
                            vizEmbeddings(char_vec, word_vec, self.outdir, self.train_writer, model)

                        init = tf.global_variables_initializer()
                        self.sess.run(init)
                        self.saveUsing(tf.train.Saver())
                        print('Writing metadata')
                        self.saveMetadata(self.sess, self.outdir, name, model)

                        saving_metric = 0
                        metric_type = "F%d" % fscore if fscore > 0 else "acc"
                        last_improved = 0
                        for i in range(epochs):
                            print('Training epoch %d.' % (i+1))
                            if i > 0:
                                print('\t(last improvement @ %d)' % (last_improved+1))
                            self._train(ts, dropout, batchsz, model, self.sess, word_keep)
                            this_acc, this_f1 = self.test(vs, batchsz, 'Validation')

                            
                            self.sess.run(self.f1.assign(this_f1))
                            this_metric = this_f1 if fscore > 0 else this_acc

                            if this_metric > saving_metric:
                                saving_metric = this_metric
                                print('Highest dev %s achieved yet -- writing model' % metric_type)

                                if (i - last_improved) > test_thresh:
                                    self.test(es, 1, 'Test')
                                self.checkpoint(name)
                                last_improved = i
                                    
                                
                            if (i - last_improved) > patience:
                                print('Stopping due to persistent failures to improve')
                                break


                        print("-----------------------------------------------------")
                        print('Highest dev %s %.2f' % (metric_type, saving_metric * 100.))
                        print('=====================================================')
                        print('Evaluating best model on test data')
                        print('=====================================================')
                        self.recover_last_checkpoint()
                        this_acc, this_f = self.test(es, 1, 'Test', eval_out, None)
                        print("-----------------------------------------------------")
                        print('Test acc %.2f' % (this_acc * 100.))
                        if fscore > 0:
                            print('Test F%d %.2f' % (fscore, this_f * 100.))
                        print('=====================================================')
                        # Write out model, graph and saver for future inference
                        self.saveValues(self.sess, self.outdir, name)
        finally:
            self.sess.close()
        return model

    def _train(self, ts, dropout, batchsz, model, sess, word_keep):

        start_time = time.time()

        steps = int(math.floor(len(ts)/float(batchsz)))

        shuffle = np.random.permutation(np.arange(steps))

        total_loss = total_err = total_sum = 0

        for i in range(steps):
            si = shuffle[i]
            ts_i = batch(ts, si, batchsz)
            feed_dict = model.ex2dict(ts_i, 1.0-dropout, True, word_keep)
        
            _, step, summary_str, lossv = sess.run([self.train_op, self.global_step, self.summary_op, self.loss], feed_dict=feed_dict)
            self.train_writer.add_summary(summary_str, step)
        
            total_loss += lossv

        duration = time.time() - start_time
        print('Train (Loss %.4f) (%.3f sec)' % (float(total_loss)/len(ts), duration))

    def test(self, ts, batchsz, phase='Test', conll_file=None, txts=None):
        return self.evaluator.test(ts, batchsz, phase, conll_file, txts)

    def createLoss(self, model):
        
        with tf.variable_scope("Loss", reuse=None):
            gold = tf.cast(model.y, tf.float32)
            mask = tf.sign(gold)

            lengths = tf.reduce_sum(mask, name="lengths",
                                    reduction_indices=1)

            all_total = tf.reduce_sum(lengths, name="total")

            block_probs = tf.unstack(model.intermediate_probs, axis=-1)

            all_loss = []
            for i, block in enumerate(block_probs):
                reuse = i != 0
                with tf.variable_scope('block', reuse=reuse):
                    if self.crf is True:
                        print('crf=True, creating SLL')
                        all_loss.append(self._computeSentenceLevelLoss(gold, mask, lengths, model, block))
                    else:
                        print('crf=False, creating WLL')
                        all_loss.append(self._computeWordLevelLoss(gold, mask, model, block))

        return tf.reduce_mean(all_loss)

    def _computeSentenceLevelLoss(self, gold, mask, lengths, model, probs):
        ll, model.A = tf.contrib.crf.crf_log_likelihood(probs, model.y, lengths)
        # print(model.probs)
        all_total = tf.reduce_sum(lengths, name="total")
        return tf.reduce_mean(-ll)

    def _computeWordLevelLoss(self, gold, mask, model, probs):

        nc = len(model.labels)
        # Cross entropy loss
        cross_entropy = tf.one_hot(model.y, nc, axis=-1) * tf.log(
            tf.clip_by_value(tf.nn.softmax(probs), 1e-10, 5.0))
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        all_loss = tf.reduce_mean(cross_entropy, name="loss")
        return all_loss

    def saveUsing(self, saver):
        self.saver = saver

    def saveValues(self, sess, outdir, base):
        basename = outdir + '/' + base
        self.saver.save(sess, basename)

    def saveMetadata(self, sess, outdir, base, model):
        
        basename = outdir + '/' + base
        tf.train.write_graph(sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        with open(basename + '.labels', 'w') as f:
            json.dump(model.labels, f)

        if len(model.word_vocab) > 0:
            with open(basename + '-word.vocab', 'w') as f:
                json.dump(model.word_vocab, f)

        with open(basename + '-char.vocab', 'w') as f:
            json.dump(model.char_vocab, f)

        with open(basename + '-params', 'w') as f:
            maxw = self.maxw
            maxlen = self.maxlen
            params = {
                'model_name':self.model_name,
                'maxw': maxw,
                'maxlen': maxlen
            }
            json.dump(params, f)
        
    def save(self, sess, outdir, base, model):
        self.saveMetadata(sess, outdir, base, model)
        self.saveValues(sess, outdir, base)

def _vizEmbedding(proj_conf, emb, outdir, which, model):
    emb_conf = proj_conf.embeddings.add()
    emb_conf.tensor_name = '%s/W' % which
    emb_conf.metadata_path = outdir + "/train/metadata-%s.tsv" % which
    writeEmbeddingsTSV(emb, emb_conf.metadata_path, model)

def vizEmbeddings(char_vec, word_vec, outdir, train_writer, model):
    print('Setting up word embedding visualization')
    proj_conf = projector.ProjectorConfig()
    _vizEmbedding(proj_conf, char_vec, outdir, 'CharLUT', model)
    if word_vec is not None:
        _vizEmbedding(proj_conf, word_vec, outdir, 'WordLUT', model)
    projector.visualize_embeddings(train_writer, proj_conf)

def writeEmbeddingsTSV(word_vec, filename, model):
    idx2word = model.y_lut
    with codecs.open(filename, 'wb') as f:
        wrtr = UnicodeWriter(f, delimiter='\t', quotechar='"')

#        wrtr.writerow(['Word'])
        for i in range(len(idx2word)):
            row = idx2word[i]
            wrtr.writerow([row])

class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = io.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = str(data)
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)