# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from matching_model import LSTMTextMatching
import os
import word2vec
import pickle

# hyper parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes", 2, "number of labels")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate of algorithm")
tf.app.flags.DEFINE_integer("batch_size", 20, "batch size for training/evaluating")
tf.app.flags.DEFINE_integer("decay_steps", 10000, "How many steps before decay learning rate") # necessary?
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Speed of decay for learning rate")
tf.app.flags.DEFINE_string("ckpt_dir", "bilstm_text_matching_checkpoint/", "directory for storing checkpoint of the model") # what for?
tf.app.flags.DEFINE_integer("word_sequence_length", 50, "maximum sentence length in word level")
tf.app.flags.DEFINE_integer("word_embed_size", 100, "word embedding dimension")
tf.app.flags.DEFINE_integer("word_hidden_size", 100, "word hidden layer dimension")
tf.app.flags.DEFINE_integer("char_sequence_length", 100, "maximum sentence length in char level")
tf.app.flags.DEFINE_integer("char_embed_size", 100, "char embedding dimension")
tf.app.flags.DEFINE_integer("char_hidden_size", 100, "char hidden layer dimension")
tf.app.flags.DEFINE_integer("hidden_size", 100, "hidden layer dimension")
tf.app.flags.DEFINE_boolean("is_training", True, "distinguish train and test")
tf.app.flags.DEFINE-integer("num_epochs", 20, "number of training iterations")
tf.app.flags.DEFINE_integer("validate_every", 1, "validate in every epoch(s)")
tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use pretrained embedding or not")
tf.app.flags.DEFINE_string("training_path", "./data/training.txt", "path of training data")
tf.app.flags.DEFINE_string("word_embedding_path", "./embedding/word_embedding.bin", "path of pretrained word embedding")
tf.app.flags.DEFINE_string("char_embedding_path", "./embedding/char_embedding.bin", "path of pretrained char embedding")

# from word to idx for a sentence pair
def make_sent(words, word2id):
    x = []
    for word in words:
        if word == 'EOS':
            x.append(0)
        elif word in word2id:
            x.append(word2id[word])
        else:
            x.append(0)
    return x
    

# convert raw sentences to tf input: word/char idx.
def make_data(data):
    XW, XC, Y = [],[],[]
    for datum in data:
        sent_word = make_sent(datum['pair_word'])
        sent_char = make_sent(datum['pair_char'])
        XW.append(sent_word)
        XC.append(sent_char)
        Y.append(datum['y'])
    return XW, XC, Y

# main function
# 1. load data (XW: list of int, XC: list of int, y: list of int)
# 2. create running session for tf
# 3. feed data
# 4. training; validation and prediction
def train():
    #1. load raw data and make feed-ins for TF function
    # train, valid, word_vocab, char_vocab, Embed_word, word2id, Embed_char, char2id
    if os.path.exists(FLAGS.training_path):
        with open(FLAGS.training_path,'r') as data_f:
            train, valid, word_vocab, char_vocab, Embed_word, word2id, Embed_char, char2id = pickle.load(data_f)
    # random shuffle the training data
    np.random.shuffle(train)
    trainXW, trainXC, trainY = make_data(train)
    validXW, validXC, validY = make_data(valid)
    word_vocab_size = len(word2id)
    char_vocab_size = len(char2id)
    num_of_train = len(trainXW)
    
    #2. session run
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # allow gpu resource from few to more
    with tf.Session(config=config) as sess:
        # initialize model
        model = LSTMTextMatching(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.word_sequence_length, word_vocab_size, FLAGS.word_embed_size, FLAGS.word_hidden_size, FLAGS.char_sequence_length, char_vocab_size, FLAGS.char_embed_size, FLAGS.char_hidden_size, FLAGS.hidden_size, FLAGS.is_training)
        
        # assign embeddings from disk to model, by running tf.assign, assign values to tf variables
        assign_word_embedding = tf.assign(model.Word_Embedding, Embed_word)
        sess.run(assign_word_embedding)
        assign_char_embedding = tf.assign(model.Char_Embedding, Embed_char)
        sess.run(assign_char_embedding)
        
        # initialize saver
        saver = tf.train.Saver()
        # load model
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print ("Restoring Variables from Checkpoint for the model.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print ("Initializing Variables")
            sess.run(tf.global_variables_initializer()) 
         
         # 3. feed the data
         curr_epoch = sess.run(model.epoch_step) # a variable , why run? initializer can't?
         batch_size = FLAGS.batch_size
         for epoch in range(curr_epoch, FLAGS.num_epochs):
            # counter is to count how many batches
            loss, acc, counter = 0.0, 0.0, 0
            # traverse the training data
            for start, end in zip(range(0, num_of_train, batch_size),
                                  range(batch_size, num_of_train, batch_size):
                # for test input?                  
                if epoch == 0 and counter == 0:
                    print ("TrainXW[start:end]:", trainXW[start:end])
                curr_loss, curr_acc, _ = sess.run([model.loss_val, model.accuracy, model.train_op], 
                                                  feed_dict={model.input_xw:trainXW[start:end], model.input_xc:trainXC[start:end], model.input_y:trainY[start:end],
                                                  model.dropout_keep_prob:1.0})
                loss, acc, counter = loss+curr_loss, acc+curr_acc, counter+1
                # output the performance instantly
                if counter % 500 == 0:
                    print ("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" % (epoch, counter, loss/float(counter), acc/float(counter)))
                # epoch increment: store the current epoch step for the model   
                sess.run(model.epoch_increment)
                
                # 4. validation
                if epoch % FLAGS.validate_every == 0:
                    eval_loss, eval_acc = evaluate(sess, model, validXW, validXC, validY, batch_size, id2word, id2char)
                    print ("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))
                    # save model to checkpoint
                    save_path = FLAGS.ckpt_dir + 'model.ckpt'
                    if not os.path.exists(FLAGS.ckpt_dir):
                        os.mkdir(FLAGS.ckpt_dir)
                    saver.save(sess, save_path, global_step=epoch)
                
# the evaluation function, where dropout is set to 0
def evaluate(sess, model, evalXW, evalXC, evalY, batch_size, id2word, id2char):
    num_eval = len(evalXW)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start,end in zip(range(0,number_eval,batch_size),
                         range(batch_size,number_eval,batch_size)):
        curr_eval_loss, logits, curr_eval_acc = sess.run([model.loss_val, model.logits, model.accuracy],
                                                         feed_dict={model.input_xw:evalXW[start:end], model.input_xc:evalXC[start:end], model.input_y: evalY[start:end],
                                                         model.dropout_keep_prob:0.0})
        eval_loss, eval_acc, eval_counter = eval_loss+curr_eval_loss, eval_acc+curr_eval_acc, eval_counter+1
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)
                
                
