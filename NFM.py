# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:58:40 2019

@author: SY
"""

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import log_loss
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import sys
sys.path.append('../')
sys.path.append('./')
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from random import shuffle
from sklearn.model_selection import KFold


class NFM():
    '''
    提取某几列类别特征的embedding
    仅使用内存
    '''
    def __init__(self, df, 
                 label_name,
                 onehot_feature=[], 
                 numerical_feature=[], 
                 vector_feature=[],
                 epoch=1,
                 embedding_size=4,
                 verbose=False,
                 batch_size=512,
                 random_state=3,
                 use_model='fm',
                 n_class=2,
                 get_prob=False,
                 df_test=None):
        assert use_model in ['fm', 'ffm'] , "use_model must in ['fm', 'ffm']"
        assert get_prob==False or (get_prob==True and df_test is not None)
        self.df = df[onehot_feature+
                     numerical_feature+
                     vector_feature+
                     [label_name]].copy()
        self.label_name = label_name
        self.onehot_feature = onehot_feature
        self.numerical_feature = numerical_feature
        self.vector_feature = vector_feature
        self.epoch = epoch
        self.embedding_size = embedding_size
        self.verbose = verbose
        self.batch_size = batch_size
        self.random_state = random_state
        self.use_model = use_model
        self.n_class = n_class
        self.get_prob = get_prob
        if self.get_prob:
            self.df_test = df_test[onehot_feature+
                                   numerical_feature+
                                   vector_feature+
                                   [label_name]].copy()
        
        self.dynamic_max_len = 10
        
        self.len_train = self.df.shape[0]
        # self.n_class = self.df[self.label_name].nunique()
        
        self.__run__()
    
    def __run__(self):
        self.preprocess()
        
        if self.verbose:
            print('======== training ========')
        
        if self.get_prob:
            oof = np.zeros((len(self.train_ix), self.n_class))
            prediction = np.zeros((len(self.test_ix), self.n_class))
            kf = KFold(n_splits=5, random_state=575)
            for fold, (train_ix, val_ix) in enumerate(kf.split(self.y[:len(self.train_ix)]), 1):
                print(f'fold {fold}')
                if self.use_model == 'fm':
                    model = Model_FM(
                            field_sizes=self.field_sizes, 
                            total_feature_sizes=self.total_feature_sizes,
                            embedding_size=self.embedding_size,
                            onehot_feature = self.onehot_feature + self.numerical_feature,
                            vector_feature = self.vector_feature,
                            n_class = self.n_class,
                                  )
                elif self.use_model == 'ffm':
                    model = Model_FFM(
                            field_sizes=self.field_sizes, 
                            total_feature_sizes=self.total_feature_sizes,
                            embedding_size=self.embedding_size,
                            onehot_feature = self.onehot_feature + self.numerical_feature,
                            vector_feature = self.vector_feature,
                            n_class = self.n_class,
                                  )
                shuffle(train_ix)
                for epoch in range(self.epoch):
                    t1 = time()
                    label_lst, out_lst = None, None
                    label_e, out_e = None, None
                    for ix, batch_ix in enumerate(range(0, len(train_ix), self.batch_size)):
                        _lst = train_ix[ix*self.batch_size:min(len(train_ix), (ix+1)*self.batch_size)]
                        batch_static, batch_dynamic, batch_dynamic_len, batch_y = self.get_batch(_lst)
                        batch_static = np.array(batch_static)
                        batch_dynamic = np.array(batch_dynamic.tolist())
                        batch_dynamic_len = np.array(batch_dynamic_len)
                        batch_y = np.array(batch_y.tolist())
                        f_dict = {model.label:batch_y,
                                  model.static_index: batch_static, 
                                  model.dropout_keep_fm:[1.0, 1.0],
                                  model.dropout_keep_deep:[1.0, 1.0, 1.0],
                                  model.train_phase: True}
                        if len(self.vector_feature) != 0:
                            f_dict.update({model.dynamic_index:batch_dynamic,
                                           model.dynamic_len:batch_dynamic_len})
                        loss_, _ ,label_, out_= model.sess.run(
                                (model.loss, model.optimizer, model.label, model.out),
                                feed_dict=f_dict)
                        out_ = res_normalization(out_)
                        if label_e is None:
                            label_e = label_
                            out_e = out_
                        else:
                            label_e = np.concatenate((label_e, label_), axis=0)
                            out_e = np.concatenate((out_e, out_), axis=0)
                        if label_lst is None:
                            label_lst = label_
                            out_lst = out_
                        else:
                            label_lst = np.concatenate((label_lst, label_), axis=0)
                            out_lst = np.concatenate((out_lst, out_), axis=0)
                        if self.verbose and ix % 100 == 99:
                            print('[%d/%d] logloss = %.4f [ %ds ]'
                                  %(ix+1, len(self.train_ix)//self.batch_size, 
                                    log_loss(label_e, out_e), 
                                    int(time()-t1)))
                            label_e, out_e = None, None
                self.embedding = model.get_embeddings(self.dic_LabelEncoder)
                for ix, batch_ix in enumerate(range(0, len(val_ix), self.batch_size)):
                    _lst = val_ix[ix*self.batch_size:min(len(self.y), (ix+1)*self.batch_size)]
                    batch_static, batch_dynamic, batch_dynamic_len, batch_y = self.get_batch(_lst)
                    batch_static = np.array(batch_static)
                    batch_dynamic = np.array(batch_dynamic.tolist())
                    batch_dynamic_len = np.array(batch_dynamic_len)
                    batch_y = np.array(batch_y.tolist())
                    
                    f_dict = {model.static_index: batch_static, 
                              model.dropout_keep_fm:[1.0, 1.0],
                              model.dropout_keep_deep:[1.0, 1.0, 1.0],
                              model.train_phase: True}
                    if len(self.vector_feature) != 0:
                        f_dict.update({model.dynamic_index:batch_dynamic,
                                       model.dynamic_len:batch_dynamic_len})
                    out_= model.sess.run(
                            (model.out),
                            feed_dict=f_dict)
                    out_ = res_normalization(out_)
                    oof[_lst] = out_
                for ix, batch_ix in enumerate(range(0, len(self.test_ix), self.batch_size)):
                    _lst = self.test_ix[ix*self.batch_size:min(len(self.y), (ix+1)*self.batch_size)]
                    batch_static, batch_dynamic, batch_dynamic_len, batch_y = self.get_batch(_lst)
                    batch_static = np.array(batch_static)
                    batch_dynamic = np.array(batch_dynamic.tolist())
                    batch_dynamic_len = np.array(batch_dynamic_len)
                    batch_y = np.array(batch_y.tolist())
                    
                    f_dict = {model.static_index: batch_static, 
                              model.dropout_keep_fm:[1.0, 1.0],
                              model.dropout_keep_deep:[1.0, 1.0, 1.0],
                              model.train_phase: True}
                    if len(self.vector_feature) != 0:
                        f_dict.update({model.dynamic_index:batch_dynamic,
                                       model.dynamic_len:batch_dynamic_len})
                    out_= model.sess.run(
                            (model.out),
                            feed_dict=f_dict)
                    out_ = res_normalization(out_)
                    prediction[np.array(_lst)-len(self.train_ix)] += out_ / 5
            self.oof = oof
            self.prediction = prediction
        else:   
            if self.use_model == 'fm':
                model = Model_FM(
                        field_sizes=self.field_sizes, 
                        total_feature_sizes=self.total_feature_sizes,
                        embedding_size=self.embedding_size,
                        onehot_feature = self.onehot_feature + self.numerical_feature,
                        vector_feature = self.vector_feature,
                        n_class = self.n_class,
                              )
            elif self.use_model == 'ffm':
                model = Model_FFM(
                        field_sizes=self.field_sizes, 
                        total_feature_sizes=self.total_feature_sizes,
                        embedding_size=self.embedding_size,
                        onehot_feature = self.onehot_feature + self.numerical_feature,
                        vector_feature = self.vector_feature,
                        n_class = self.n_class,
                              )
            train_ix = list(range(len(self.y)))
            shuffle(train_ix)
            for epoch in range(self.epoch):
                t1 = time()
                label_lst, out_lst = None, None
                label_e, out_e = None, None
                for ix, batch_ix in enumerate(range(0, self.len_train, self.batch_size)):
                    batch_static, batch_dynamic, batch_dynamic_len, batch_y = self.get_batch(
                            train_ix[ix*self.batch_size:min(len(self.y), (ix+1)*self.batch_size)])
                    batch_static = np.array(batch_static)
                    batch_dynamic = np.array(batch_dynamic.tolist())
                    batch_dynamic_len = np.array(batch_dynamic_len)
                    batch_y = np.array(batch_y.tolist())
                    
                    f_dict = {model.label:batch_y,
                              model.static_index: batch_static, 
                              model.dropout_keep_fm:[1.0, 1.0],
                              model.dropout_keep_deep:[1.0, 1.0, 1.0],
                              model.train_phase: True}
                    if len(self.vector_feature) != 0:
                        f_dict.update({model.dynamic_index:batch_dynamic,
                                       model.dynamic_len:batch_dynamic_len})
                    loss_, _ ,label_, out_= model.sess.run(
                            (model.loss, model.optimizer, model.label, model.out),
                            feed_dict=f_dict)
                    # 
                    out_ = res_normalization(out_)
                    if label_e is None:
                        label_e = label_
                        out_e = out_
                    else:
                        label_e = np.concatenate((label_e, label_), axis=0)
                        out_e = np.concatenate((out_e, out_), axis=0)
                    if label_lst is None:
                        label_lst = label_
                        out_lst = out_
                    else:
                        label_lst = np.concatenate((label_lst, label_), axis=0)
                        out_lst = np.concatenate((out_lst, out_), axis=0)
                    if self.verbose and ix % 100 == 99:
                        print('[%d/%d] logloss = %.4f [ %ds ]'
                              %(ix+1, self.len_train//self.batch_size, 
                                log_loss(label_e, out_e), 
                                int(time()-t1)))
                        label_e, out_e = None, None
            
            self.embedding = model.get_embeddings(self.dic_LabelEncoder)
    
    def preprocess(self):
        if self.verbose:
            print('preprocessing...')
        if self.get_prob:
            self.train_ix = list(range(len(self.df)))
            self.test_ix = list(range(len(self.df), len(self.df)+len(self.df_test)))
            df = self.df.append(self.df_test).reset_index(drop=True)
        dic_LabelEncoder = {}
        bins = 10
        n_features = [0, 0]
        '''ONEHOT_COL'''
        if len(self.onehot_feature) != 0:
            for col in self.onehot_feature:
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col].astype(str).fillna('-1').values) + n_features[0]
                temp_dic = {}
                for ix, item in enumerate(enc.classes_):
                    temp_dic[item] = n_features[0] + ix
                n_features[0] += np.int32(len(set(df[col])))
                dic_LabelEncoder[col] = temp_dic
        '''NUMERIC_COL'''
        if len(self.numerical_feature) != 0:
            '''preprocessing'''
            for col in self.numerical_feature:
                df[col] = df[col].astype(np.float32)
                _mean = np.mean(df[col])
                _percentile_99 = np.percentile(df[col], 0.99)
                _percentile_01 = np.percentile(df[col], 0.01)
                if _percentile_99 > 100 * _mean:
                    df[col].loc[df[col] > _percentile_99] = _percentile_99
                if _percentile_01 < 0.01 * _mean:
                    df[col].loc[df[col] < _percentile_01] = _percentile_01
            '''to bins, normalization'''
            for col in self.numerical_feature:
                df[col] = pd.cut(df[col].fillna(-1), bins, labels=False).astype(np.int32)+np.int32(n_features[0])
                n_features[0] += np.int32(len(set(df[col])))
                if type(df[col][0]) != np.int32:
                    df[col] = df[col].astype(np.int32)
        '''VECTOR_COL'''
        if len(self.vector_feature) != 0:
            for col in self.vector_feature:
                df[col] = df[col].fillna('-1')
                df[col] = df[col].apply(lambda x:x.replace(',', ' '))
                lst = list(set(' '.join(df[col]).split(' ')))
                temp_dic = {}
                for i, j in enumerate(lst):
                    temp_dic[j] = n_features[1] + i
                n_features[1] += len(lst)
                df[col] = df[col].apply(lambda x:' '.join([str(temp_dic[j]) for j in x.split(' ')]))
                dic_LabelEncoder[col] = temp_dic
        df[self.label_name] = df[self.label_name].astype(np.float32)
        
        # df = df.sample(frac=1).reset_index(drop=True)
        
        if len(self.vector_feature) != 0:
            for col in self.vector_feature:
                df[col] = df[col].apply(lambda x:[int(i) for i in x.split(' ')])
        
        self.field_sizes = [len(self.onehot_feature)+len(self.numerical_feature), len(self.vector_feature)]
        self.total_feature_sizes = n_features
        ''''''
        self.static = df[self.onehot_feature + self.numerical_feature].values.copy()
        
        self.y = df[self.label_name].fillna(-1).astype(int).apply(
                lambda x:[1 if i == x else 0 for i in range(self.n_class)]).values.copy()
        
        self.dynamic_len = df[self.vector_feature].copy()
        for col in self.vector_feature:
            self.dynamic_len[col] = self.dynamic_len[col].apply(lambda x :len(x))
        self.dynamic_len = self.dynamic_len.values
        
        
        self.dynamic = df[self.vector_feature].copy()
        for col in self.vector_feature:
            self.dynamic[col] = self.dynamic[col].apply(lambda x:[x[i] if i < len(x) else 0 for i in range(self.dynamic_max_len)])
        self.dynamic = self.dynamic.values
        
        self.dic_LabelEncoder = dic_LabelEncoder
        del self.df, self.df_test

    
    def get_batch(self, ix):
        return (self.static[ix], 
                self.dynamic[ix], 
                self.dynamic_len[ix], 
                self.y[ix])
        
    
        

def res_normalization(arr):
    for i in range(len(arr)):
        arr[i] = arr[i]/np.sum(arr[i])
    return arr


class Model_FM(BaseEstimator, TransformerMixin):
    def __init__(self, field_sizes, total_feature_sizes,
                 onehot_feature=[], 
                 vector_feature=[],
                 n_class=2,
                 dynamic_max_len=10, extern_lr_size=0, extern_lr_feature_size=0,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[256, 128], dropout_deep=[1.0, 1.0],
                 val_batch_size=128,
                 deep_layers_activation=tf.nn.relu,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=1, batch_norm_decay=0.995,
                 random_state=465,
                 loss_type="logloss", 
                 l2_reg=0.0,
                 verbose=False):
        self.field_sizes = field_sizes
        self.total_field_size = sum(field_sizes)
        self.total_feature_sizes = total_feature_sizes
        self.embedding_size = embedding_size
        self.extern_lr_size = extern_lr_size
        self.dynamic_max_len = dynamic_max_len
        self.extern_lr_feature_size = extern_lr_feature_size
        self.onehot_feature = onehot_feature
        self.vector_feature = vector_feature
        self.n_class = n_class
        self.verbose = verbose
        

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers

        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation

        self.val_batch_size = val_batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.random_state = random_state
        self.loss_type = loss_type
        self.train_result, self.valid_result = [], []
        
        self._init_graph()


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_state)
            '''placeholder'''
            self.label = tf.placeholder(tf.float32, shape=[None, self.n_class], name="label")
            self.static_index = tf.placeholder(tf.int32, shape=[None, len(self.onehot_feature)], name="static_index")
            if len(self.vector_feature) != 0:
                self.dynamic_index = tf.placeholder(tf.int32, shape=[None, len(self.vector_feature), self.dynamic_max_len], name="dynamic_index")
                self.dynamic_len = tf.placeholder(tf.int32, shape=[None, len(self.vector_feature)], name="dynamic_len")
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()
            
            # static
            self.static_fm_embs = tf.nn.embedding_lookup(self.weights["static_ffm_embeddings"],
                                                          self.static_index) # None * static_feature_size * [k * F]
            # dynamic
            if len(self.vector_feature) != 0:
                self.dynamic_fm_embs = tf.nn.embedding_lookup(self.weights["dynamic_ffm_embeddings"],
                                                              self.dynamic_index) # None * [dynamic_feature_size * max_len] * [k * F]
                self.fm_mask = tf.sequence_mask(tf.reshape(self.dynamic_len,[-1]), maxlen=self.dynamic_max_len) # [None * dynamic_feature] * max_len
                self.fm_mask = tf.expand_dims(self.fm_mask, axis=-1) # [None * dynamic_feature] * max_len * 1
                self.fm_mask = tf.concat([self.fm_mask for i in range(self.embedding_size)], axis = -1) # [None * dynamic_feature] * max_len * [k * F]
                self.dynamic_fm_embs = tf.reshape(self.dynamic_fm_embs,[-1, self.dynamic_max_len, self.embedding_size]) # [None * dynamic_feature] * max_len * [k * F]
                self.dynamic_fm_embs = tf.multiply(self.dynamic_fm_embs, tf.to_float(self.fm_mask)) # [None * dynamic_feature] * max_len * [k * F]
                self.dynamic_fm_embs = tf.reshape(tf.reduce_sum(self.dynamic_fm_embs, axis=1),[-1, self.field_sizes[1],
                                                                                                 self.embedding_size]) # None * dynamic_feature_size * [k * F]
                self.padding_lengths = tf.concat([tf.expand_dims(self.dynamic_len, axis=-1)
                                                  for i in range(self.embedding_size)],axis=-1) # None * dynamic_feature_size * [k * F]
                self.dynamic_fm_embs = tf.div(self.dynamic_fm_embs, tf.to_float(self.padding_lengths)) # None * dynamic_feature_size * [k * F]
                # concat
                self.fm_embs = tf.concat([self.static_fm_embs, self.dynamic_fm_embs], axis=1)
            else:
                self.fm_embs = self.static_fm_embs
            
            # 矩阵乘
            self.fm_embs_out = tf.matmul(self.fm_embs, tf.transpose(self.fm_embs,[0,2,1]))
            self.fm_embs_out = tf.reshape(self.fm_embs_out, 
                                          [-1, self.total_field_size * self.total_field_size]) # None * [F * (F-1) / 2 * k]
    
            self.y_deep = self.fm_embs_out #tf.reshape(self.ffm_embs_col,[-1, self.total_field_size * self.total_field_size * self.embedding_size])
    
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.matmul(self.y_deep, self.weights["layer_%d" % i])
                #self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer
            self.y_deep = tf.layers.dense(inputs=self.y_deep, units=self.deep_layers[-1], activation=self.deep_layers_activation)
            
            self.out = tf.add(tf.matmul(self.y_deep, self.weights["concat_projection"]), self.weights["concat_bias"])
            self.out = tf.reshape(self.out, shape=[-1, self.n_class])
            self.label = tf.reshape(self.label, shape=[-1, self.n_class])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose:
                print("#params: %d" % total_parameters)


    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        return tf.Session(config=config)


    def _initialize_weights(self):
        weights = dict()
        # FFM embeddings
        input_size = self.embedding_size #  * self.total_field_size
        glorot = 0.0001#np.sqrt(2.0 / (input_size * self.total_feature_sizes))
        weights["static_ffm_embeddings"] = tf.Variable(
            tf.random_normal([self.total_feature_sizes[0], input_size], 0.0, glorot),
            name="static_ffm_embeddings")  # static_feature_size * [K * F]
        if len(self.vector_feature) != 0:
            weights["dynamic_ffm_embeddings"] = tf.Variable(
                tf.random_normal([self.total_feature_sizes[1], input_size], 0.0, glorot),
                name="dynamic_ffm_embeddings")  # dynamic_feature_size * [K * F]
        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.total_field_size * self.total_field_size 
        glorot = np.sqrt(2.0 / (input_size))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]

        # final concat projection layer
        input_size = self.deep_layers[-1]
        if self.extern_lr_size:
            input_size += self.extern_lr_feature_size
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, self.n_class)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(-3.5), dtype=np.float32)

        return weights


    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z
    
    
    def get_embeddings(self, dic):
        emb = self.sess.run(
                tf.nn.embedding_lookup(
                        self.weights["static_ffm_embeddings"], 
                        np.array(range(self.total_feature_sizes[0]))))
        for col in self.onehot_feature:
            try:
                for key in dic[col]:
                    dic[col][key] = emb[dic[col][key]]
            except:
                pass
        if len(self.vector_feature) != 0:
            emb = self.sess.run(
                    tf.nn.embedding_lookup(
                            self.weights["dynamic_ffm_embeddings"], 
                            np.array(range(self.total_feature_sizes[1]))))
            for col in self.vector_feature:
                for key in dic[col]:
                    try:
                        dic[col][key] = emb[dic[col][key]]
                    except:
                        pass
        
        return dic


class Model_FFM(BaseEstimator, TransformerMixin):
    def __init__(self, field_sizes, total_feature_sizes,
                 onehot_feature=[], 
                 vector_feature=[],
                 n_class=2,
                 dynamic_max_len=10, extern_lr_size=0, extern_lr_feature_size=0,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[256, 128], dropout_deep=[1.0, 1.0],
                 val_batch_size=128,
                 deep_layers_activation=tf.nn.relu,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=1, batch_norm_decay=0.995,
                 random_state=465,
                 loss_type="logloss", 
                 l2_reg=0.0,
                 verbose=False):
        self.field_sizes = field_sizes
        self.total_field_size = sum(field_sizes)
        self.total_feature_sizes = total_feature_sizes
        self.embedding_size = embedding_size
        self.extern_lr_size = extern_lr_size
        self.dynamic_max_len = dynamic_max_len
        self.extern_lr_feature_size = extern_lr_feature_size
        self.onehot_feature = onehot_feature
        self.vector_feature = vector_feature
        self.n_class = n_class
        self.verbose = verbose
        

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers

        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation

        self.val_batch_size = val_batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.random_state = random_state
        self.loss_type = loss_type
        self.train_result, self.valid_result = [], []
        
        self._init_graph()


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_state)
            '''placeholder'''
            self.label = tf.placeholder(tf.float32, shape=[None, self.n_class], name="label")
            self.static_index = tf.placeholder(tf.int32, shape=[None, len(self.onehot_feature)], name="static_index")
            if len(self.vector_feature) != 0:
                self.dynamic_index = tf.placeholder(tf.int32, shape=[None, len(self.vector_feature), self.dynamic_max_len], name="dynamic_index")
                self.dynamic_len = tf.placeholder(tf.int32, shape=[None, len(self.vector_feature)], name="dynamic_len")
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()
            
            # static
            self.static_ffm_embs = tf.nn.embedding_lookup(self.weights["static_ffm_embeddings"],
                                                          self.static_index) # None * static_feature_size * [k * F]
            # dynamic
            if len(self.vector_feature) != 0:
                self.dynamic_ffm_embs = tf.nn.embedding_lookup(self.weights["dynamic_ffm_embeddings"],
                                                              self.dynamic_index) # None * [dynamic_feature_size * max_len] * [k * F]
                self.ffm_mask = tf.sequence_mask(tf.reshape(self.dynamic_len,[-1]), maxlen= self.dynamic_max_len) # [None * dynamic_feature] * max_len
                self.ffm_mask = tf.expand_dims(self.ffm_mask, axis=-1) # [None * dynamic_feature] * max_len * 1
                self.ffm_mask = tf.concat([self.ffm_mask for i in range(self.embedding_size * self.total_field_size)], axis = -1) # [None * dynamic_feature] * max_len * [k * F]
                self.dynamic_ffm_embs = tf.reshape(self.dynamic_ffm_embs,[-1, self.dynamic_max_len, self.embedding_size * self.total_field_size]) # [None * dynamic_feature] * max_len * [k * F]
                self.dynamic_ffm_embs = tf.multiply(self.dynamic_ffm_embs, tf.to_float(self.ffm_mask)) # [None * dynamic_feature] * max_len * [k * F]
                self.dynamic_ffm_embs = tf.reshape(tf.reduce_sum(self.dynamic_ffm_embs, axis=1),[-1, self.field_sizes[1],
                                                                                                 self.embedding_size * self.total_field_size]) # None * dynamic_feature_size * [k * F]
                self.padding_lengths = tf.concat([tf.expand_dims(self.dynamic_len, axis=-1)
                                                  for i in range(self.embedding_size * self.total_field_size)],axis=-1) # None * dynamic_feature_size * [k * F]
                self.dynamic_ffm_embs = tf.div(self.dynamic_ffm_embs, tf.to_float(self.padding_lengths)) # None * dynamic_feature_size * [k * F]
                # concat
                self.ffm_embs = tf.concat([self.static_ffm_embs, self.dynamic_ffm_embs], axis=1)
            else:
                self.ffm_embs = self.static_ffm_embs
            
            # 矩阵乘
            self.ffm_embs_col = tf.reshape(self.ffm_embs,
                                           [-1, self.total_field_size, self.total_field_size, self.embedding_size]) # None * F * F * k
            self.ffm_embs_row = tf.transpose(self.ffm_embs_col, [0, 2, 1, 3]) # None * F * F * k
            self.ffm_embs_out = tf.multiply(self.ffm_embs_col, self.ffm_embs_row) # None *F * F * k
            self.ones = tf.ones_like(self.ffm_embs_out)
            self.op = tf.contrib.linalg.LinearOperatorTriL(tf.transpose(self.ones,[0,3,1,2])) # None *k * F *F
            self.upper_tri_mask = tf.less(tf.transpose(self.op.to_dense(), [0,2,3,1]), self.ones) # None *F * F * k

            self.ffm_embs_out = tf.boolean_mask(self.ffm_embs_out, self.upper_tri_mask) # [None * F * (F-1) * k]
            self.ffm_embs_out = tf.reshape(self.ffm_embs_out, [-1, self.total_field_size * (self.total_field_size-1) // 2
                                                              * self.embedding_size]) # None * [F * (F-1) / 2 * k]
    
            self.y_deep = self.ffm_embs_out #tf.reshape(self.ffm_embs_col,[-1, self.total_field_size * self.total_field_size * self.embedding_size])
    
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.matmul(self.y_deep, self.weights["layer_%d" % i])
                #self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer
            self.y_deep = tf.layers.dense(inputs=self.y_deep, units=self.deep_layers[-1], activation=self.deep_layers_activation)
            
            self.out = tf.add(tf.matmul(self.y_deep, self.weights["concat_projection"]), self.weights["concat_bias"])
            self.out = tf.reshape(self.out, shape=[-1, self.n_class])
            self.label = tf.reshape(self.label, shape=[-1, self.n_class])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose:
                print("#params: %d" % total_parameters)


    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        return tf.Session(config=config)


    def _initialize_weights(self):
        weights = dict()
        # FFM embeddings
        input_size = self.embedding_size * self.total_field_size
        glorot = 0.0001#np.sqrt(2.0 / (input_size * self.total_feature_sizes))
        weights["static_ffm_embeddings"] = tf.Variable(
            tf.random_normal([self.total_feature_sizes[0], input_size], 0.0, glorot),
            name="static_ffm_embeddings")  # static_feature_size * [K * F]
        if len(self.vector_feature) != 0:
            weights["dynamic_ffm_embeddings"] = tf.Variable(
                tf.random_normal([self.total_feature_sizes[1], input_size], 0.0, glorot),
                name="dynamic_ffm_embeddings")  # dynamic_feature_size * [K * F]
        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.total_field_size * (self.total_field_size -1) // 2 * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]

        # final concat projection layer
        input_size = self.deep_layers[-1]
        if self.extern_lr_size:
            input_size += self.extern_lr_feature_size
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, self.n_class)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(-3.5), dtype=np.float32)

        return weights


    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z
    
    def get_embeddings(self, dic):
        emb = self.sess.run(
                tf.nn.embedding_lookup(
                        self.weights["static_ffm_embeddings"], 
                        np.array(range(self.total_feature_sizes[0]))))
        for col in self.onehot_feature:
            try:
                for key in dic[col]:
                    dic[col][key] = emb[dic[col][key]]
            except:
                pass
        if len(self.vector_feature) != 0:
            emb = self.sess.run(
                    tf.nn.embedding_lookup(
                            self.weights["dynamic_ffm_embeddings"], 
                            np.array(range(self.total_feature_sizes[1]))))
            for col in self.vector_feature:
                for key in dic[col]:
                    dic[col][key] = emb[dic[col][key]]
        
        return dic

def quick_merge(df1:pd.DataFrame, df2:pd.DataFrame, on:list, feat:list, fillna=-1):
    assert type(on) == list and type(feat) == list
    if len(on) == 1:
        d = {}
        for item in df2[on+feat].values:
            if item[0] not in d:
                d[item[0]] = item[1:]
        df1['temp'] = df1[on].values.tolist()
        for ix, col in enumerate(feat):
            df1[col] = df1['temp'].apply(
                    lambda x:d[x[0]][ix] if x[0] in d else fillna)
        del df1['temp']
    elif len(on) == 2:
        d = {}
        for item in df2[on+feat].values:
            if item[0] not in d:
                d[item[0]] = {}
            if item[1] not in d[item[0]]:
                d[item[0]][item[1]] = item[2:]
        df1['temp'] = df1[on].values.tolist()
        for ix, col in enumerate(feat):
            df1[col] = df1['temp'].apply(
                    lambda x:d[x[0]][x[1]][ix] if x[0] in d and x[1] in d[x[0]] else fillna)
        del df1['temp']
    elif len(on) == 3:
        d = {}
        for item in df2[on+feat].values:
            if item[0] not in d:
                d[item[0]] = {}
            if item[1] not in d[item[0]]:
                d[item[0]][item[1]] = {}
            if item[2] not in d[item[0]][item[1]]:
                d[item[0]][item[1]][item[2]] = item[3:]
        df1['temp'] = df1[on].values.tolist()
        for ix, col in enumerate(feat):
            df1[col] = df1['temp'].apply(
                    lambda x:d[x[0]][x[1]][x[2]][ix] if 
                    x[0] in d and 
                    x[1] in d[x[0]] and 
                    x[2] in d[x[0]][x[1]] 
                    else fillna)
        del df1['temp']
    else:
        pass
    return df1
    
# =============================================================================
# if __name__ == '__main__':
#     
#     nrows = 200000
#     data = pd.read_csv('./input/user_basic_info.csv', header=None, nrows=None)
#     data.columns = ['uid', 'gender', 'city', 'prodName', 'ramCapacity', 
#                     'ramLeftRation', 'romCapacity', 'romLeftRation', 'color',
#                     'fontSize', 'ct', 'carrier', 'os']
#     label = pd.read_csv('./input/age_train.csv', header=None, nrows=nrows)
#     label.columns = ['uid', 'age']
#     label_name = 'age'
#     data = quick_merge(label, data, on=['uid'], 
#                        feat=['gender', 'city', 'prodName', 
#                              'color', 'ct', 'carrier', 'os'])
#     model = NFM(data.fillna('-1'), label_name='age', 
#                 onehot_feature=['gender', 'city', 'prodName', 
#                                 'color', 'ct', 'carrier', 'os'],
#                 verbose=True,
#                 use_model='ffm')
#     data['city_emb'] = data['city'].fillna('-1').apply(
#             lambda x:model.embedding['city'][x])
#     print(data['city_emb'].head())
# =============================================================================
    