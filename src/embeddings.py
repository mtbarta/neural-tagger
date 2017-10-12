#word embeddings

import numpy as np
import io

def readtospc(f):

    s = bytearray()
    ch = f.read(1)

    while ch != b'\x20':
        s.extend(ch)
        ch = f.read(1)
    s = s.decode('utf-8')
    return s.strip()

class Word2VecModel:

    def __init__(self, filename, knownvocab=None, unifweight=None):

        uw = 0.0 if unifweight == None else unifweight
        self.vocab = {}
        self.vocab["<PADDING>"] = 0
        with open(filename, "rb") as f:
            header = f.readline()
            vsz, self.dsz = map(int, header.split())

            if knownvocab is not None:
                self.vsz = 0
                for v in knownvocab:
                    self.vsz += 1
            else:
                self.vsz = vsz

            self.weights = np.random.uniform(-uw, uw, (self.vsz+1, self.dsz))
            width = 4 * self.dsz
            k = 1
            # All attested word vectors
            for i in range(vsz):
                word = readtospc(f)
                raw = f.read(width)
                # If vocab list, not in: dont add, in:add, drop from list
                if word in self.vocab:
                    continue

                if knownvocab is not None:
                    if word not in knownvocab:
                        continue

                    # Otherwise drop freq to 0, for later
                    knownvocab[word] = 0
                vec = np.fromstring(raw, dtype=np.float32)
                self.weights[k] = vec
                self.vocab[word] = k
                k = k + 1

            # Anything left over, unattested in w2v model, just use a random
            # initialization
        if knownvocab is not None:
            unknown = {v: cnt for v,cnt in knownvocab.items() if cnt > 0}
            for v in unknown:
                self.vocab[v] = k
                k = k + 1

        self.nullv = np.zeros(self.dsz, dtype=np.float32)
        self.weights[0] = self.nullv

    def lookup(self, word, nullifabsent=True):
        if word in self.vocab:
            return self.weights[self.vocab[word]]
        if nullifabsent:
            return None
        return self.nullv

class GloVeModel:

    def __init__(self, filename, known_vocab=None, unif_weight=None, keep_unused=False):
        uw = 0.0 if unif_weight is None else unif_weight
        self.vocab = {}
        idx = 1

        word_vectors = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                if keep_unused is False and word not in known_vocab:
                    continue

                # Otherwise add it to the list and remove from knownvocab
                if known_vocab and word in known_vocab:
                    known_vocab[word] = 0
                vec = np.asarray(values[1:], dtype=np.float32)
                word_vectors.append(vec)
                self.vocab[word] = idx
                idx += 1
            self.dsz = vec.shape[0]
            self.nullv = np.zeros(self.dsz, dtype=np.float32)
            word_vectors = [self.nullv] + word_vectors
            self.vocab["<PAD>"] = 0

        if known_vocab is not None:
            unknown = {v: cnt for v, cnt in known_vocab.items() if cnt > 0}
            for v in unknown:
                word_vectors.append(np.random.uniform(-uw, uw, self.dsz))
                self.vocab[v] = idx
                idx += 1

        self.weights = np.array(word_vectors)
        self.vsz = self.weights.shape[0] - 1

    def lookup(self, word, nullifabsent=True):
        if word in self.vocab:
            return self.weights[self.vocab[word]]
        if nullifabsent:
            return None
        return self.nullv

class RandomInitVecModel:

    def __init__(self, dsz, knownvocab, unifweight=None):

        uw = 0.0 if unifweight == None else unifweight
        self.vocab = {}
        self.vocab["<PADDING>"] = 0
        self.dsz = dsz
        self.vsz = 0

            
        attested = {v: cnt for v,cnt in knownvocab.items() if cnt > 0}
        for k,v in enumerate(attested):
            self.vocab[v] = k + 1
            k = k + 1
            self.vsz += 1

        self.weights = np.random.uniform(-uw, uw, (self.vsz+1, self.dsz))


        self.nullv = np.zeros(self.dsz, dtype=np.float32)
        self.weights[0] = self.nullv

    def lookup(self, word, nullifabsent=True):
        if word in self.vocab:
            return self.weights[self.vocab[word]]
        if nullifabsent:
            return None
        return self.nullv

  

