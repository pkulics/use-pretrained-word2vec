#-*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
'''本程序只是对word2vec进行了简单的预处理，应用到复杂模型中还需要根据实际情况做必要的改动'''

class Wordlist(object):
    def __init__(self, filename, maxn = 100000):
        lines = map(lambda x: x.split(), open(filename).readlines()[:maxn])
        self.size = len(lines)

        self.voc = [(item[0][0], item[1]) for item in zip(lines, xrange(self.size))]
        self.voc = dict(self.voc)

    def getID(self, word):
        try:
            return self.voc[word]
        except:
            return 0

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    i=0
    word_vecs = {}
    pury_word_vec = []
    with open(fname, "rb") as f:
        header = f.readline()
        print 'header',header
        vocab_size, layer1_size = map(int, header.split())
        print 'vocabsize:',vocab_size,'layer1_size:',layer1_size
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                #print ch
                if ch == ' ':
                    word = ''.join(word)
                    #print 'single word:',word
                    break
                if ch != '\n':
                    word.append(ch)
                    #print word
            #print word
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
               pury_word_vec.append(word_vecs[word])
               if i==0:
                   print 'word',word
                   i=1
            else:
                f.read(binary_len)
       #np.savetxt('googleembedding.txt',pury_word_vec)
    return word_vecs,pury_word_vec

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

if __name__=="__main__":
    w2v_file = "GoogleNews-vectors-negative300.bin"#Google news word2vec bin文件
    print "loading data...",
    vocab = Wordlist('vocab.txt')#自己的数据集要用到的词表
    w2v,pury_word2vec = load_bin_vec(w2v_file, vocab.voc)
    add_unknown_words(w2v, vocab.voc)
    W, word_idx_map = get_W(w2v)

    '''embedding lookup简单应用'''
    Wa = tf.Variable(W)
    embedding_input = tf.nn.embedding_lookup(Wa, [0,1,2])#正常使用时要替换成相应的doc

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input = sess.run(Wa)
        #print np.shape(Wa)
