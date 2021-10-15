import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input

from sklearn.metrics import roc_auc_score

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from tensorflow.keras import initializers

from models.tf_commands import *
dist_Normal = tf.compat.v1.distributions.Normal

from os import path
import pandas as pd

import copy
import itertools

# dist_normal = tf.contrib.distributions.Normal(0.0, 1.0)

# Sampling-Free Variational Inference of Bayesian Neural Networks by Variance Backpropagation
# https://github.com/manuelhaussmann/vbp
class vb_layer(Layer):
    def __init__(self, units, dim_input, is_ReLUoutput=False,
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 **kwargs):
        super(vb_layer, self).__init__(**kwargs)
        self.units = units
        self.d_input = dim_input

        self.is_ReLUoutput = is_ReLUoutput

        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.prior_prec = 1.0

        return None

    def build(self, input_shape):
        self.w_mus = self.add_weight(\
            shape=(self.d_input, self.units), initializer=self.kernel_initializer, name='kernel', trainable=True)
        self.w_logsig2 = self.add_weight(\
            shape=(self.d_input, self.units), initializer=self.kernel_initializer, name='kernel', trainable=True)

        self.b = self.add_weight(\
            shape=(1, self.units), initializer=self.bias_initializer, name='bias', trainable=True)

        if self.is_ReLUoutput:
            self.gamma = self.add_weight(\
                shape=(1, self.units), initializer=self.bias_initializer, name='omega_ard', trainable=True)

        return None

    def call(self, in_means, in_vars):
        # mean
        out_means = mat_mul(in_means, self.w_mus) + self.b

        if (self.is_ReLUoutput == False):
            # variance
            m2s2_w = square(self.w_mus) + exp(self.w_logsig2)
            out_vars = mat_mul(in_vars, m2s2_w) + mat_mul(square(in_means), exp(self.w_logsig2))

        else:
            # mean
            pZ = sigmoid(out_means)
            factor_ = exp(0.5 * self.gamma) * (2 / np.sqrt(self.units))
            out_means = multiply(multiply(pZ, out_means), factor_)

            # variance
            if in_vars is None:
                term1 = 0
            else:
                m2s2_w = square(self.w_mus) + exp(self.w_logsig2)
                term1 = mat_mul(in_vars, m2s2_w)
            term2 = mat_mul(square(in_means), exp(self.w_logsig2))

            # Compute E[h]^2
            term3 = square(mat_mul(in_means, self.w_mus) + self.b)

            out_vars = multiply(pZ, term1 + term2) + multiply(multiply(pZ, (1 - pZ)), term3)
            out_vars = multiply(square(factor_), out_vars)

        return out_means, out_vars

    def calcuate_KL(self):
        w_logsig2 = tf.clip_by_value(self.w_logsig2, -11, 11)
        kl = 0.5 * reduce_sum(multiply(self.prior_prec, (square(self.w_mus) + exp(w_logsig2))) \
                              - w_logsig2 - log(self.prior_prec))
        return kl


class DGP_RF_Embeddings(Model):
    def __init__(self, fea_dims, num_RF):
        super(DGP_RF_Embeddings, self).__init__()

        # DGP on the top of image features
        input_means = Input(shape=(fea_dims[0],))
        for i in range(len(fea_dims)-1):
            # Omega layer
            if (i==0):
                inter_means, inter_vars = input_means, None

            inter_means, inter_vars = vb_layer(num_RF, fea_dims[i], is_ReLUoutput=True)(inter_means, inter_vars)

            # Weight layer
            inter_means, inter_vars = vb_layer(fea_dims[i + 1], num_RF)(inter_means, inter_vars)

        self.DGP_RF = Model(input_means, [inter_means, inter_vars])

        return None

    def __call__(self, X, X_idx):
        # learning embeddings with DGP-RF
        out_means, out_vars = self.DGP_RF(X)

        '''
        # reduced to the image level
        embedd_means = segment_mean(out_means, segment_ids=X_idx)

        Nis = tf.cast(segment_sum(tf.ones_like(X_idx), segment_ids=X_idx), dtype=tf.float32)
        embedd_vars = divide(segment_sum(out_vars, segment_ids=X_idx), vec_colwise(square(Nis)))
        '''

        mat_tmp1 = divide(1, out_vars)

        embedd_vars = divide(1, segment_sum(mat_tmp1, segment_ids=X_idx))
        embedd_means = segment_sum(multiply(mat_tmp1, out_means), segment_ids=X_idx)

        return multiply(embedd_vars, embedd_means), embedd_vars

    def cal_regul(self):
        mr_KLsum = 0.0
        for mn_i in range(1, len(self.DGP_RF.layers)):
            layer_ = self.DGP_RF.layers[mn_i]

            mr_KLsum += layer_.calcuate_KL()

        return mr_KLsum


def Prob_Triplet_loss(y_true, est_means, est_mvars, NmulPnN, alpha=0.5):
    idx_pos = tf.where(y_true == 1.0)
    idx_pos = vec_rowwise(idx_pos)

    idx_neg = vec_colwise(tf.where(y_true == 0.0))

    n_pos = tf.size(idx_pos)
    n_neg = tf.size(idx_neg)

    idx_pos_ex = vec_flat(K.repeat(idx_pos, n=n_neg))
    idx_neg_ex = vec_flat(K.repeat(idx_neg, n=n_pos))

    muA = tf.gather(est_means, [0], axis=0)
    muP = tf.gather(est_means, idx_pos_ex, axis=0)
    muN = tf.gather(est_means, idx_neg_ex, axis=0)

    varA = tf.gather(est_mvars, [0], axis=0)
    varP = tf.gather(est_mvars, idx_pos_ex, axis=0)
    varN = tf.gather(est_mvars, idx_neg_ex, axis=0)

    probs_ = calculate_lik_prob(muA, muP, muN, varA, varP, varN)
    loss_ = reduce_sum(log(probs_))

    const_ = NmulPnN/(alpha*tf.cast((n_pos*n_neg), dtype=np.float32))
    return -const_ * loss_

def calculate_lik_prob(muA, muP, muN, varA, varP, varN, margin=0.0):
    muA2 = square(muA)
    muP2 = square(muP)
    muN2 = square(muN)

    varP2 = square(varP)
    varN2 = square(varN)

    mu = reduce_sum(muP2 + varP - muN2 - varN - 2*multiply(muA, muP-muN), axis=1)

    T1 = varP2 + 2*multiply(muP2, varP) + 2*multiply(varA+muA2, varP+muP2) \
         - 2*multiply(muA2, muP2) - 4*multiply(muA, multiply(muP, varP))
    T2 = varN2 + 2*multiply(muN2, varN) + 2*multiply(varA+muA2, varN+muN2) \
         - 2*multiply(muA2, muN2) - 4*multiply(muA, multiply(muN, varN))
    T3 = 4*multiply(muP, multiply(muN, varA))
    sigma = sqrt(tf.maximum(0.0, reduce_sum(2*T1 + 2*T2 - 2*T3, axis=1)))

    probs_ = dist_Normal(loc=mu, scale=sigma).cdf(margin)
    return probs_

class DGP_RF:
    def __init__(self, data_X, data_Y, trn_index, setting, str_filepath=None):
        N_pos = np.sum(data_Y[trn_index]==1)
        self.NpNm = 0.5*((N_pos*(N_pos-1))*np.sum(data_Y[trn_index]==0))

        self.max_iter = setting.max_iter
        self.iter_print = setting.iter_print

        self.sub_Ni = setting.sub_Ni

        self.batch_size = setting.batch_size

        self.ker_type = setting.ker_type
        self.n_RF = setting.n_RF

        self.regul_const = tf.cast(setting.regul_const, dtype=np.float32)

        self.alpha = setting.alpha

        # (X, fea_dims, tau = 1.0, dropout=0.05):
        fea_dims_sub = [100] * setting.n_layers
        fea_dims = np.array([data_X.data_mat[0].shape[1]] + fea_dims_sub)

        #- data loader
        self.data_X = data_X

        self.trn_index = trn_index
        self.Ytrn = data_Y[trn_index]

        self.Y = np.reshape(data_Y, [-1])
        self.pos_idx = np.intersect1d(np.argwhere(self.Y == 1.0), trn_index)
        self.neg_idx = np.intersect1d(np.argwhere(self.Y == 0.0), trn_index)

        # - define the model
        # __init__(self, fea_dims, num_RF, nMCsamples, num_Att=4, dim_Att=32, ker_type='rbf', flag_layer_norm=False):
        self.model = DGP_RF_Embeddings(fea_dims, self.n_RF)

        # optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

        self.mat_trn_est = None

        #- train the model
        if str_filepath is None:
            self.model_fit()
        else:
            if path.exists(str_filepath + '.index'):
                self.model.load_weights(str_filepath)
                print('load the trained model: ' + str_filepath)
            else:
                self.model_fit()
                self.model.save_weights(str_filepath)
        return None

    # Optimization process.
    def run_optimization(self, X, X_idx, Y, regul_const=1e-2):
        X = tf.convert_to_tensor(X, np.float32)
        X_idx = tf.convert_to_tensor(X_idx, np.int32)

        Y = tf.convert_to_tensor(Y, np.float32)

        #  Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            # Forward pass.
            est_means, est_vars = self.model(X, X_idx)

            # Compute loss.
            loss = Prob_Triplet_loss(Y, est_means, est_vars, self.NpNm, self.alpha)
            reg_ = self.model.cal_regul()

            obj = loss + regul_const*reg_

        # Variables to update, i.e. trainable variables.
        trainable_variables = self.model.trainable_variables

        # Compute gradients.
        gradients = g.gradient(obj, trainable_variables)

        # Update W and b following gradients.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return obj

    def model_fit(self):
        progbar = tf.keras.utils.Progbar(self.max_iter)

        n_pos = np.int_(np.maximum(np.rint(self.batch_size / 2), 1))
        n_neg = n_pos  # self.batch_size - n_pos

        iters_Pos = len(self.pos_idx)

        for epoch in range(0, self.max_iter):  #
            if (epoch < 10):
                eta = 1.0
            else:
                eta = 1.0

            obj = 0.0
            for iter in range(iters_Pos):
                anc_idx = self.pos_idx[iter]

                pos_idx = np.random.choice(np.setdiff1d(self.pos_idx, anc_idx), \
                                           np.minimum(n_pos, len(self.pos_idx) - 1), replace=False)
                pos_idx = np.concatenate(([anc_idx], pos_idx))

                neg_idx = np.random.choice(self.neg_idx, np.minimum(n_neg, len(self.neg_idx)), replace=False)

                # load np data matrics
                index_vec = np.concatenate((pos_idx, neg_idx))
                set_indices = self.mark_subImgs(self.data_X, index_vec, sub_Ni=self.sub_Ni)

                X, X_idx = self.gen_input_fromList(self.data_X, index_vec, set_indices[0])
                Y = self.Y[index_vec]
                Y[0] = -1  # for the ancher node

                obj += self.run_optimization(X, X_idx, Y, eta * self.regul_const)

            progbar.update(epoch+1) # This will update the progress bar graph.

            print('trn Obj: (%d: %.3f)' % (epoch, obj/iters_Pos))

            if self.iter_print & (epoch == (self.max_iter - 1)):  # epoch > 0) & (((epoch % 100) == 0) |
                out_ests = self.predict(self.trn_index, sub_Ni=self.sub_Ni, rep_num=1, flag_trndata=True)
                auc_val = roc_auc_score(self.Ytrn, out_ests)
                print(' trnAUC = (%.3f)' % auc_val)
        return None

    def mark_subImgs(self, data_X, index_vec, sub_Ni, rep_num=1, flag_AllIns=False):
        # calculate the instnace weights
        Nis = np.hstack([data_X.Nis[idx] for idx in index_vec])

        # random sampling
        set_indices = []
        for mn_rep in range(rep_num):
            set_indices_sub = []
            for Ni in Nis:
                if (flag_AllIns == False):
                    idx_selected = np.sort(np.random.choice( \
                        np.arange(Ni), size=np.minimum(Ni, sub_Ni), replace=False))
                else:
                    idx_selected = np.array(range(Ni))

                set_indices_sub.append(idx_selected)

            set_indices.append(set_indices_sub)

        return set_indices

    def gen_input_fromList(self, data_X, index_vec, set_indices):
        Nis = []
        for mn_i, idx in enumerate(index_vec, 0):
            idx_selected = set_indices[mn_i]
            Xsub = data_X.data_mat[idx][idx_selected]
            Nis.append(len(idx_selected))

            if mn_i == 0:
                X = Xsub
            else:
                X = np.concatenate((X, Xsub), axis=0)

        X_idx = [cnt * np.ones((Ni, 1), dtype=np.int32) for cnt, Ni in enumerate(Nis, 0)]
        X_idx = np.reshape(np.vstack(X_idx), [-1])

        return X, X_idx

    def select_sub_percnt(self, Xvec, pert=5):
        X_sorted = tf.sort(Xvec, direction='ASCENDING')

        N_ = Xvec.shape[-1]
        n_rem = np.int_(np.floor(N_*pert/100))

        idx_selected = range(n_rem+1, N_-n_rem)
        return tf.gather(X_sorted, idx_selected)

    def predict(self, tst_index, data_set_=None, sub_Ni=None, rep_num=1, flag_trndata=False):
        vec_colwise_np = lambda x: np.reshape(x, [-1, 1])
        vec_rowwise_np = lambda x: np.reshape(x, [1, -1])
        vec_flatten_np = lambda x: np.reshape(x, [-1])

        if data_set_ is None:
            # data_filenames = self.data_filenames
            data_set_ = self.data_X

        if sub_Ni is None:
            sub_Ni = self.sub_Ni

        # calcuate embeddings
        means_trn, vars_trn = self.model_eval\
            (self.trn_index, data_set_=self.data_X, sub_Ni=sub_Ni, rep_num=rep_num)

        # for the testing data
        if flag_trndata:
            means_tst, vars_tst = means_trn, vars_trn
        else:
            means_tst, vars_tst = self.model_eval(tst_index, data_set_=data_set_, sub_Ni=sub_Ni, rep_num=rep_num)

        # embeddings from the test data
        idx_pos = np.reshape(self.Ytrn == 1.0, [-1])
        idx_neg = np.reshape(self.Ytrn == 0.0, [-1])

        N_pos = np.sum(idx_pos)
        N_neg = np.sum(idx_neg)

        means_trn_pos = tf.boolean_mask(means_trn, idx_pos, axis=0)
        means_trn_neg = tf.boolean_mask(means_trn, idx_neg, axis=0)

        vars_trn_pos = tf.boolean_mask(vars_trn, idx_pos, axis=0)
        vars_trn_neg = tf.boolean_mask(vars_trn, idx_neg, axis=0)

        #
        idx_pos_ex = vec_flatten_np(np.tile(vec_rowwise_np(range(N_pos)), reps=N_neg))
        idx_neg_ex = np.repeat(vec_colwise_np(range(N_neg)), repeats=N_pos)

        #
        Y_probs = np.zeros((len(tst_index), 1))

        if flag_trndata:
            # for training data points
            for mn_i in range(len(tst_index)):
                muA = tf.expand_dims(tf.gather(means_tst, mn_i, axis=0), axis=0)
                varA = tf.expand_dims(tf.gather(vars_tst, mn_i, axis=0), axis=0)

                if self.Ytrn[mn_i] == 1.0:
                    idx = np.sum(self.Ytrn[range(mn_i + 1)] == 1)
                    idx_selected = ~(idx_pos_ex == (idx - 1))
                else:
                    idx = np.sum(self.Ytrn[range(mn_i + 1)] == 0)
                    idx_selected = ~(idx_neg_ex == (idx - 1))

                # prediction
                idx_pos_ex_new = tf.boolean_mask(idx_pos_ex, idx_selected)
                idx_neg_ex_new = tf.boolean_mask(idx_neg_ex, idx_selected)

                muP = tf.gather(means_trn_pos, idx_pos_ex_new, axis=0)
                muN = tf.gather(means_trn_neg, idx_neg_ex_new, axis=0)

                varP = tf.gather(vars_trn_pos, idx_pos_ex_new, axis=0)
                varN = tf.gather(vars_trn_neg, idx_pos_ex_new, axis=0)

                prob_sub = calculate_lik_prob(muA, muP, muN, varA, varP, varN)
                Y_probs[mn_i] = K.eval(reduce_mean(prob_sub))
        else:
            # for test data points
            muP = tf.gather(means_trn_pos, idx_pos_ex, axis=0)
            muN = tf.gather(means_trn_neg, idx_neg_ex, axis=0)

            varP = tf.gather(vars_trn_pos, idx_pos_ex, axis=0)
            varN = tf.gather(vars_trn_neg, idx_pos_ex, axis=0)

            for mn_i in range(len(tst_index)):
                muA = tf.expand_dims(tf.gather(means_tst, mn_i, axis=0), axis=0)
                varA = tf.expand_dims(tf.gather(vars_tst, mn_i, axis=0), axis=0)

                prob_pos = calculate_lik_prob(muA, muP, muN, varA, varP, varN)
                Y_probs[mn_i] = K.eval(reduce_mean(prob_pos))

        return Y_probs

    def model_eval(self, tst_index, data_set_, sub_Ni, rep_num, batch_size=5):
        Ntst = len(tst_index)

        for mn_i in range(np.int_(np.ceil(Ntst/batch_size))):
            if Ntst== 1:
                index_vec = tst_index
            else:
                sub_idx = range(mn_i * batch_size, np.minimum((mn_i + 1) * batch_size, Ntst))
                index_vec = tst_index[np.hstack(sub_idx)]

            set_indices = self.mark_subImgs(data_set_, index_vec, sub_Ni=sub_Ni, rep_num=rep_num)

            for mn_sub in range(rep_num):
                X, X_idx = self.gen_input_fromList(data_set_, index_vec, set_indices[mn_sub])
                out_means_sub, out_vars_sub = self.model(X, X_idx)

                out_means_sub = tf.expand_dims(out_means_sub, axis=-1)
                out_vars_sub = tf.expand_dims(out_vars_sub, axis=-1)

                if mn_sub == 0:
                    out_means, out_vars = out_means_sub, out_vars_sub
                else:
                    out_means = tf.concat((out_means, out_means_sub), axis=2)
                    out_vars = tf.concat((out_vars, out_vars_sub), axis=2)

            if mn_i == 0:
                means_set, vars_set = out_means, out_vars
            else:
                means_set = tf.concat((means_set, out_means), axis=0)
                vars_set = tf.concat((vars_set, out_vars), axis=0)

        return means_set, vars_set


