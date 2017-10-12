#!/usr/bin/python
# -*- coding: utf8

"""
This file holds classes to help get data into
a format that tensorflow models can understand.

tensorflow wants dictionaries of arrays called feeddicts.
	ex: {"x":xs,"y":ys, "xch": xs_ch, "id": i, "length": length }

from text and labels, we need to find max sentence length (or provide)
"""

def SentsToIndices( texts, lbls, words_vocab, chars_vocab, mxlen, maxw, f2i, label_idx=-1):

    b = 0
    ts = []
    idx = 0

    for i in range(len(txts)):

        xs_ch = np.zeros((mxlen, maxw), dtype=np.int)
        xs = np.zeros((mxlen), dtype=np.int)
        ys = np.zeros((mxlen), dtype=np.int)
        
        lv = lbls[i]
        v = txts[i]
        
        length = mxlen
        for j in range(mxlen):
            
            if j == len(v):
                length = j
                break
            
            w = v[j]
            nch = min(len(w), maxw)
            label = lv[j]

            if not label in f2i:
                idx += 1
                f2i[label] = idx

            ys[j] = f2i[label]
            xs[j] = words_vocab.get(cleanup(w), 0)
            for k in range(nch):
                xs_ch[j,k] = chars_vocab.get(w[k], 0)
        ts.append({"x":xs,"y":ys, "xch": xs_ch, "id": i, "length": length })

    return ts, f2i, txts

def validSplit(data, splitfrac):
    train = []
    valid = []
    numinst = len(data)
    heldout = int(math.floor(numinst * (1-splitfrac)))
    return data[1:heldout], data[heldout:]


def batch(ts, start, batchsz):
    ex = ts[start]
    siglen = ex["x"].shape[0]
    maxw = ex["xch"].shape[1]
    
    xs_ch = np.zeros((batchsz, siglen, maxw), dtype=np.int)
    xs = np.zeros((batchsz, siglen), dtype=np.int)
    ys = np.zeros((batchsz, siglen), dtype=np.int)
    ids = np.zeros((batchsz), dtype=np.int)
    length = np.zeros((batchsz), dtype=np.int)
    sz = len(ts)
    idx = start * batchsz
    for i in range(batchsz):
        if idx >= sz: idx = 0
        
        ex = ts[idx]
        xs_ch[i] = ex["xch"]
        xs[i] = ex["x"]
        ys[i] = ex["y"]
        ids[i] = ex["id"]
        length[i] = ex["length"]
        idx += 1
    return {"x": xs, "y": ys, "xch": xs_ch, "length": length, "id": ids }