import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

# build dataset, return train, valid, word_vocab, char_vocab
def build_dataset(data_folder):
    train, valid = [], []
    # only build vocab based on the train set, take the unknown word in valid as unknown
    train_word_path = data_folder[0]
    valid_word_path = data_folder[1]
    train = []
    valid = []
    word_vocab = defaultdict(float)
    char_vocab = defaultdict(float)
    with open(train_word_path, 'rb') as f:
        for line in f:
            vec = line.strip().split()
            label = vec[0]
            pair_word = vec[1]
            pair_char = vec[2]
            words = set(pair_word.split())
            for word in words:
                word_vocab[word] += 1
            chars = set(pair_char.split())
            for char in char:
                char_vocab[char] += 1
            datum = {'y': label,
                     'pair_word': pair_word,
                     'pair_char': pair_char,
                     'length_word': len(pair_word.split()),
                     'length_char': len(pair_char.split())
                    }
            train.append(datum)
    with open(valid_word_path, 'rb') as f:
        for line in f:
            vec = line.strip().split()
            label = vec[0]
            pair_word = vec[1]
            pair_char = vec[2]
            words = set(pair_word.split())
            for word in words:
                word_vocab[word] += 1
            chars = set(pair_char.split())
            for char in char:
                char_vocab[char] += 1
            datum = {'y': label,
                     'pair_word': pair_word,
                     'pair_char': pair_char,
                     'length_word': len(pair_word.split()),
                     'length_char': len(pair_char.split())
                    }
            valid.append(datum)
    return train, valid, word_vocab, char_vocab          

# pay attention to the float32 type
# i don't know if if will goes wrong on 64 bit OS
def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)  
            # only assign know words to the w2v vector, else run add_unknown_words
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

# should add to train and valid, but the first run is only for training data
def add_unknown_words(word_vecs, vocab, min_df=1, k=100):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    counter = 0
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
            counter += 1
    return counter
    
# add NIL as index 0 and generate numpy matrix for embedding   
def get_W(word_vecs, k=100):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word2id = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')  
    # the NULL word, i.e. </s>
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word2id[word] = i
        i += 1
    return W, word2id


if __name__=="__main__":    
    # given directories
    w2v_file = './embedding/word2vec_cn_qa.bin'
    c2v_file = './embedding/char2vec_cn_qa.bin'
    data_folder = ['./data/train_word.txt', './data/train_char.txt', './data/valid_word.txt', './data/valid_word.txt']
    # 1. Load data from disk and build dataset
    print 'loading data...'
    train, valid, word_vocab, char_vocab = build_dataset(data_folder, clean_string=True)
    # better to set a maximum length such as 200 words or 400 chars
    max_l_word = np.max(pd.DataFrame(train)['length_word'])
    max_l_char = np.max(pd.DataFrame(train)['length_char'])
    print ("data loaded!")
    print ("number of training sentences: %d " % len(train))
    print ("vocab size: %d" % len(vocab))
    print ("max sentence length on word level: %d" % max_l_word)
    print ("max sentence length on char level: %d" % max_l_char)
    # 2. Load pretrained embeddings from disk and assign them to vectors
    print ("loading word2vec vectors...")
    w2v = load_bin_vec(w2v_file, word_vocab)
    print ("word2vec loaded!")
    print ("number of words already in word2vec is %d" % len(w2v))
    c2v = load_bin_vec(c2v_file, char_vocab)
    print ("char2vec loaded!")
    print ("number of words already in char2vec is %d" % len(c2v))
    # 3. Add unkown words to vectors, pay attention when validating should also add unknown
    counter = add_unknown_words(w2v, word_vocab)
    print ("adding %d unknown words from dataset to w2v" % counter)
    counter = add_unknown_words(c2v, char_vocab)
    print ("adding %d unknown chars from dataset to c2v" % counter)
    # 4. build numpy format embeddings
    Embed_word, word2id = get_W(w2v)
    Embed_char, char2id = get_W(c2v)
    # 5. Store data to disk
    cPickle.dump([train, valid, word_vocab, char_vocab, Embed_word, word2id, Embed_char, char2id], open('matching_dataset.pkl', 'wb'))
    print "dataset created!"
    
