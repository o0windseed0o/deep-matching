# -*- coding: utf-8 -*-
"""
LSTM-based class for sentence matching
    Description: binary classification with 1 matching, 0 no-matching
    Input format: sentence1 EOS sentence2
    Graph construction: embedding layer --> Bi-LSTM layer --> (pooling layer) --> 2 fully connected layer --> softmax
    Note: for word embedding and char embedding, the outputs of two graphs are concatenated in fc layer
    Author: Xiang,Yang
    Contact: xiangyang.hitsz@gmail.com
"""
    
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random

class LSTMTextMatching:
    # why using sequence length, how initializer uses
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, word_sequence_length, word_vocab_size, word_embed_size, word_hidden_size, char_squence_length, char_vocab_size, char_embed_size, char_hidden_size, hidden_size, is_training, initializer=tf.random_normal_initializer(stddev=0.1)):
        # init all hyperparameters here
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.word_sequence_length = word_sequence_length # no. of words
        self.char_sequence_length = char_sequence_length # no. of chars
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.word_hidden_size = word_hidden_size
        self.char_hidden_size = char_hidden_size
        self.hidden_size = hidden_size # before softmax, after the concat of word and char LSTMs
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.initializer = initializer # what for?

        # add placeholder as input
        self.input_xw = tf.placeholder(tf.int32, [None, self.word_sequence_length], name='input_xw') # concat of two word sequences
        self.input_xc = tf.placeholder(tf.int32, [None, self.char_sequence_length], name='input_xc') # concat of two char sequences
        self.input_y = tf.placeholder(tf.int32, [None], name='input_y') #y: [None, num_classes]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") # for differentiate train and test
        
        self.global_step = tf.Variable(0, trainable=False, name='Global_Step') # what for?
        self.epoch_step = tf.Variable(0, trainable=False, name='Epoch_Step')  # from this epoch
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1))) # what for?
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        
        self.initialize_weights() 
        self.logits = self.inference()
        # why return ?
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name='predictions') # shape:[None,] --> batch size
        # what is cast func? what is recuce mean
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')
        
        
    def initialize_weights(self):
        """
        """
        with tf.name_scope('embedding'):
            # need input to the initializer, need two for word or char?, what's get_variable
            self.Word_Embedding = tf.get_variable('Word_Embedding', shape=[self.word_vocab_size, self.word_embed_size], initializer=self.initializer)
            self.Char_Embedding = tf.get_variable('Char_Embedding', shape=[self.char_vocab_size, self.char_embed_size], initializer=self.initializer)
            self.W_hidden = tf.get_variable('W_hidden', shape=[self.word_hidden_size*2+self.char_hidden_size*2, self.hidden_size], initializer=self.initializer)
            self.b_hidden = tf.get_variable('b_hidden', shape=[self.hidden_size])
            self.W_projection = tf.get_variable('W_projection', shape=[self.hidden_size, self.num_classes], initializer=self.initializer)
            self.b_projection = tf.get_variable('b_projection', shape=[self.num_classes])


    def inference(self):
        """
        The computational graph starts here
        """
        # Embedding layer
        self.embedded_words = tf.nn.embedding_lookup(self.Word_Embedding, self.input_xw)
        self.embedded_chars = tf.nn.embedding_lookup(self.Char_Embedding, self.input_xc)

        # Bi-LSTM layer for word and char
        word_lstm_fw_cell = rnn.BasicLSTMCell(self.word_hidden_size)
        word_lstm_bw_cell = rnn.BasicLSTMCell(self.word_hidden_size)
        char_lstm_fw_cell = rnn.BasicLSTMCell(self.char_hidden_size)
        char_lstm_bw_cell = rnn.BasicLSTMCell(self.char_hidden_size)
        
        # dropout when training
        if self.dropout_keep_prob != 0.0:
            word_lstm_fw_cell = rnn.DropoutWrapper(word_lstm_fw_Cell, output_keep_prob=self.dropout_keep_prob)
            word_lstm_bw_cell = rnn.DropoutWrapper(word_lstm_bw_Cell, output_keep_prob=self.dropout_keep_prob)
            char_lstm_fw_cell = rnn.DropoutWrapper(char_lstm_fw_Cell, output_keep_prob=self.dropout_keep_prob)
            char_lstm_bw_cell = rnn.DropoutWrapper(char_lstm_bw_Cell, output_keep_prob=self.dropout_keep_prob)

        word_outputs, _ = tf.nn.bidirectional_dynamic_rnn(word_lstm_fw_cell, word_lstm_bw_cell, self.embedded_words, dtype=tf.float32)
        char_outputs, _ = tf.nn.bidirectional_dynamic_rnn(char_lstm_fw_cell, char_lstm_bw_cell, self.embedded_chars, dtype=tf.float32)
        word_lstm_outputs = tf.concat(word_outputs, axis=2) # [batch_size, sequence_length, word_hidden_size*2]
        char_lstm_outputs = tf.concat(char_outputs, axis=2) # [batch_size, sequence_length, char_hidden_size*2]
       
        # fc layer for dimension reduction and softmax
        fc_inputs = tf.concat([word_lstm_outputs, char_lstm_outputs], axis=2) # [batch_size, sequence_length, word_hidden_size*2+char_hidden_size*2]
        with name_scope('fully_connected'):
            fc_layer = tf.matmul(fc_inputs, self.W_hidden) + self.b_hidden
            logits = tf.matmul(fc_layer, self.W_projection) + self.b_projection
        return logits

    
    def loss(self, l2_lambda=0.0001):
        with tf.name_scope('loss'):
            # what difference?
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            # what for?
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss += l2_losses
        return loss


    def train(self):
        # look up the function??
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True) 
        # fixed ?
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer='Adam')
        return train_op
    


