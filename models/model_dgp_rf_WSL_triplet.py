import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from sklearn.metrics import roc_auc_score

from settings import dropout_prob

from models.dgp_layers import Omega_layer, Linear
from models.tf_commands import *

from models.platt_prob_calibration import plattFit
from models.platt_prob_calibration import sigmoid as sigmoid_platt

from os import path
import pandas as pd

import copy
import itertools

dropout_1mp = 1 - dropout_prob

class DGP_RF_Embeddings(Model):
    def __init__(self, fea_dims, num_RF, nMCsamples, \
                 num_Att=4, dim_Att=32, ker_type='rbf', flag_layer_norm=False):
        super(DGP_RF_Embeddings, self).__init__()

        self.nAtt = num_Att
        self.dAtt = dim_Att
        self.nMCsamples = nMCsamples

        if ker_type == 'rbf':
            n_factor = 2
        else:
            n_factor = 1

        if len(fea_dims) > 2:
            flag_layer_norm = True

        # DGP on the top of image features
        input_feature = Input(shape=(fea_dims[0],))
        for i in range(len(fea_dims)-1):
            # Omega layer
            if i==0:
                inter = Omega_layer\
                    (num_RF, fea_dims[i], ker_type, flag_in=True, nMCsamples=self.nMCsamples,\
                     flag_layer_norm=flag_layer_norm)(input_feature)
            else:
                inter = Omega_layer\
                    (num_RF, fea_dims[i], ker_type, flag_layer_norm=flag_layer_norm)(output)

            # Weigth layer
            output = Linear(fea_dims[i + 1], n_factor*num_RF, activation=None)(inter)
        self.DGP_RF = Model(input_feature, output)

        # for Socring of each instance
        # https://stackoverflow.com/questions/47940040/how-to-set-input-shape-of-a-trained-model-in-keras
        input_feature = Input(batch_shape=self.DGP_RF.output.shape)
        output = Linear(num_Att, fea_dims[-1], activation=None)(input_feature)

        self.Seed_Vecs = Model(input_feature, output)

        # For Embbeding aggregation
        input_feature = Input(batch_shape=self.DGP_RF.output.shape)

        output = Linear(num_Att * dim_Att, fea_dims[-1])(input_feature)
        self.MHa_Transform = Model(input_feature, output)

        return None

    def __call__(self, X, X_idx=None):
        # learning embeddings with DGP-RF
        embeddings = self.DGP_RF(X)

        # Scoring: a single-layer DGP
        relevances_scores = transpose(self.Seed_Vecs(embeddings), [1, 0, 2])

        relevances_probs = relevances_scores/sqrt(embeddings.shape[-1])
        relevances_probs = exp(relevances_probs
                               - tf.gather(log(segment_sum(exp(relevances_probs), segment_ids=X_idx)), X_idx, axis=0))

        if X_idx is None:
            return relevances_probs

        # Embedding: multi-head attention
        # probs_new = tf.repeat(probs, self.dAtt*np.ones(self.nAtt, dtype=np.int32), axis=2)
        embeddings_new = transpose(self.MHa_Transform(embeddings), [1, 0, 2])
        probs_new = K.repeat_elements(relevances_probs, self.dAtt, axis=2)

        embedds = segment_sum(multiply(embeddings_new, probs_new), segment_ids=X_idx)
        return embedds

    def cal_regul(self):
        # DGPs
        reg_MCDrop, reg_MCDrop_none = cal_layers_regul_sq(self.DGP_RF.layers)

        # for the Score layer (a single DGP)
        reg_MCDrop_score, reg_MCDrop_none_score = cal_layers_regul_sq(self.Seed_Vecs.layers)

        reg_MCDrop += reg_MCDrop_score
        reg_MCDrop_none += reg_MCDrop_none_score

        reg_MCDrop_score, reg_MCDrop_none_score = cal_layers_regul_sq(self.MHa_Transform.layers)

        reg_MCDrop += reg_MCDrop_score
        reg_MCDrop_none += reg_MCDrop_none_score

        return (dropout_1mp * reg_MCDrop) + reg_MCDrop_none

def cal_layers_regul_sq(layers_):
    reg_MCDrop = 0.0
    reg_MCDrop_none = 0.0
    for mn_i in range(len(layers_)):
        layer_ = layers_[mn_i]

        num_weights = len(layer_.weights)
        if num_weights == 0:
            continue

        for mn_sub in range(num_weights):
            if mn_sub == 0:
                reg_MCDrop += reduce_sum(square(layer_.weights[mn_sub]))
            else:
                reg_MCDrop_none += reduce_sum(square(layer_.weights[mn_sub]))
    return reg_MCDrop, reg_MCDrop_none


def gen_pairwise_idx(ref_index):
    tmp_list = list(itertools.combinations(ref_index, 2))
    tmp_index = np.vstack(list(itertools.chain(*tmp_list)))

    return np.reshape(tmp_index[::2], [-1]), np.reshape(tmp_index[1::2], [-1])

def approx_alpha_lik(y_true, y_dist_agg, NmulPnN, alpha, nMCsamples):
    idx_pos = tf.where(y_true == 1.0)

    pairl_idx_left, pairl_idx_right = gen_pairwise_idx(K.eval(idx_pos))
    n_pos_comb = pairl_idx_left.shape[0]

    dist_pos = reduce_sum(square(tf.gather(y_dist_agg, pairl_idx_left, axis=0)
                                 - tf.gather(y_dist_agg, pairl_idx_right, axis=0)), axis=2)

    idx_pos = vec_colwise(np.intersect1d(K.eval(idx_pos), pairl_idx_left))
    idx_neg = vec_rowwise(tf.where(y_true == 0.0))

    n_pos = tf.size(idx_pos)
    n_neg = tf.size(idx_neg)

    idx_pos_ex = vec_flat(K.repeat(idx_pos, n=n_neg))
    idx_neg_ex = vec_flat(K.repeat(idx_neg, n=n_pos))

    dist_neg = tf.reshape(reduce_sum(square(tf.gather(y_dist_agg, idx_pos_ex, axis=0)
                                            - tf.gather(y_dist_agg, idx_neg_ex, axis=0)), axis=2), [n_pos, n_neg, nMCsamples])

    mat_diff = tf.gather(dist_neg, pairl_idx_left, axis=0) - tf.expand_dims(dist_pos, axis=1)

    loss_ = reduce_sum(reduce_sum(logsumexp(-alpha * logistic_loss(mat_diff), axis=2), axis=1))
    const_ = NmulPnN/(alpha*tf.cast(n_pos_comb*n_neg, dtype=np.float32))

    return -const_ *loss_


def approx_alpha_lik_anchor(y_true, y_dist_agg, NmulPnN, alpha, nMCsamples):
    output_res = reduce_sum(
        square(y_dist_agg - tf.expand_dims(tf.gather(y_dist_agg, 0, axis=0), axis=0)), axis=2)

    idx_pos = tf.where(y_true == 1.0)
    idx_pos = vec_rowwise(idx_pos)

    idx_neg = vec_colwise(tf.where(y_true == 0.0))

    n_pos = tf.size(idx_pos)
    n_neg = tf.size(idx_neg)

    idx_pos_ex = vec_flat(K.repeat(idx_pos, n=n_neg))
    idx_neg_ex = vec_flat(K.repeat(idx_neg, n=n_pos))

    mat_diff = tf.gather(output_res, idx_neg_ex, axis=0) \
               - tf.gather(output_res, idx_pos_ex, axis=0)

    loss_ = reduce_sum(logsumexp(-alpha * logistic_loss(mat_diff), axis=1))
    const_ = NmulPnN/(alpha*tf.cast((n_pos*n_neg), dtype=np.float32))

    return -const_ * loss_


class DGP_RF:
    def __init__(self, data_X, data_Y, trn_index, setting, str_filepath=None):
        N_pos = np.sum(data_Y[trn_index]==1)
        self.NpNm = 0.5*((N_pos*(N_pos-1))*np.sum(data_Y[trn_index]==0))

        self.max_iter = setting.max_iter
        self.iter_print = setting.iter_print

        self.sub_Ni = setting.sub_Ni
        self.selection_mode = setting.selection_mode

        self.batch_size = setting.batch_size

        self.ker_type = setting.ker_type
        self.n_RF = setting.n_RF

        self.regul_const = tf.cast(setting.regul_const, dtype=np.float32)

        self.nMCsamples = setting.nMCsamples
        self.alpha = setting.alpha

        self.num_Att = setting.num_Att
        self.dim_Att = setting.dim_Att

        self.batch_anchor = setting.batch_anchor
        if self.batch_anchor:
            self.loss_fun = approx_alpha_lik_anchor
        else:
            self.loss_fun = approx_alpha_lik

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
        self.model = DGP_RF_Embeddings(fea_dims, self.n_RF, self.nMCsamples, \
                                       setting.num_Att, setting.dim_Att, self.ker_type)

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
            Y_pred = self.model(X, X_idx)

            # Compute loss.
            loss = self.loss_fun(Y, Y_pred, self.NpNm, self.alpha, self.nMCsamples)
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

        total_pos = len(self.pos_idx)

        if self.batch_anchor:
            iters_Pos = len(self.pos_idx)
        else:
            iters_Pos = np.int_(np.ceil(total_pos/n_pos))

        for epoch in range(0, self.max_iter):  #
            if (epoch < 10):
                eta = 1.0
            else:
                eta = 1.0

            if ~self.batch_anchor:
                random_pos_index = copy.deepcopy(self.pos_idx)
                np.random.shuffle(random_pos_index)

            obj = 0.0
            for iter in range(iters_Pos):
                if self.batch_anchor:
                    anc_idx = self.pos_idx[iter]

                    pos_idx = np.random.choice(np.setdiff1d(self.pos_idx, anc_idx), \
                                               np.minimum(n_pos, len(self.pos_idx) - 1), replace=False)
                    pos_idx = np.concatenate(([anc_idx], pos_idx))
                else:
                    pos_idx = random_pos_index[range(iter * n_pos, np.minimum((iter + 1) * n_pos, total_pos))]
                    if len(pos_idx) < 2:
                        pos_idx = np.concatenate(
                            (pos_idx, np.random.choice(np.setdiff1d(self.pos_idx, pos_idx), 1, replace=False)))

                neg_idx = np.random.choice(self.neg_idx, np.minimum(n_neg, len(self.neg_idx)), replace=False)

                # load np data matrics
                index_vec = np.concatenate((pos_idx, neg_idx))
                set_indices = self.mark_subImgs\
                    (self.data_X, index_vec, sub_Ni=self.sub_Ni, selection_mode=self.selection_mode)

                X, X_idx = self.gen_input_fromList(self.data_X, index_vec, set_indices[0])
                Y = self.Y[index_vec]

                if self.batch_anchor:
                    Y[0] = -1  # for the ancher node

                obj += self.run_optimization(X, X_idx, Y, eta * self.regul_const)

            progbar.update(epoch+1) # This will update the progress bar graph.

            print('trn Obj: (%d: %.3f)' % (epoch, obj/iters_Pos))

            if self.iter_print & (epoch == (self.max_iter - 1)):  # epoch > 0) & (((epoch % 100) == 0) |
                out_ests = self.predict(self.trn_index, sub_Ni=self.sub_Ni, rep_num=1, flag_trndata=True)
                auc_val = roc_auc_score(self.Ytrn, out_ests)
                print(' trnAUC = (%.3f)' % auc_val)
        return None

    def mark_subImgs(self, data_X, index_vec, sub_Ni, \
                     selection_mode='random', rep_num=1, flag_AllIns=False):
        # calculate the instnace weights
        Nis = np.hstack([data_X.Nis[idx] for idx in index_vec])

        # random sampling
        if selection_mode=='random':
            set_indices = []
            for mn_rep in range(rep_num):

                set_indices_sub = []
                for Ni in Nis:
                    idx_selected = np.random.choice(\
                        np.arange(Ni), size=np.minimum(Ni, sub_Ni), replace=False)
                    set_indices_sub.append(np.sort(idx_selected))

                set_indices.append(set_indices_sub)

            return set_indices

        # draw tiles based on the scores
        batch_size = 1000
        X = np.vstack([data_X.data_mat[idx] for idx in index_vec])

        X_idx = [cnt * np.ones((Ni, 1), dtype=np.int32) for cnt, Ni in enumerate(Nis, 0)]
        X_idx = np.reshape(np.vstack(X_idx), [-1])

        Ncnt = X.shape[0]
        for mn_i in range(np.int_(np.ceil(Ncnt/batch_size))):
            sub_idx = range(mn_i * batch_size, np.minimum((mn_i + 1) * batch_size, Ncnt))
            ins_relevances_sub = self.model(X[sub_idx])

            if mn_i == 0:
                ins_relevances = ins_relevances_sub
            else:
                ins_relevances = tf.concat((ins_relevances, ins_relevances_sub), axis=1)

        for mn_i in range(self.nMCsamples):
            probs = ins_relevances[mn_i] - tf.gather(segment_max(ins_relevances[mn_i], segment_ids=X_idx), X_idx)
            probs = exp(probs - tf.gather(log(segment_sum(exp(probs), segment_ids=X_idx)), X_idx))

            probs = tf.expand_dims(probs, axis=0)

            if mn_i == 0:
                prob_relevances = probs
            else:
                prob_relevances = tf.concat((prob_relevances, probs), axis=0)

        ins_relevances = K.eval(reduce_mean(prob_relevances, axis=0))

        # select a subset
        range_score_dims = range(self.num_Att)
        pos_vec = np.cumsum(Nis)

        set_indices = []
        for mn_rep in range(rep_num):

            set_indices_sub = []
            for mn_i, Ni in enumerate(Nis, 0):
                if mn_i==0:
                    chk_start_pos = 0
                else:
                    chk_start_pos = pos_vec[mn_i-1]
                chk_last_pos = pos_vec[mn_i]

                cur_score = ins_relevances[chk_start_pos:chk_last_pos, ]

                Ni_sel = np.minimum(Ni, sub_Ni)
                if Ni_sel < self.num_Att:
                    print('%dth: too small number (%d) of tiles'% (mn_i, Ni))

                Ni_sub_each = np.int_(np.floor(Ni_sel/self.num_Att))
                Ni_sub_each_fst = Ni_sub_each + (Ni_sel - (Ni_sub_each*self.num_Att))

                if (flag_AllIns == False):
                    # select tiles with the largest weights
                    if selection_mode == 'max':
                        sorted_idx = np.argsort(-cur_score, axis=0)

                        for mn_sub in range_score_dims:
                            if mn_sub  == 0:
                                idx_selected = sorted_idx[range(Ni_sub_each_fst), 0]
                            else:
                                chk_idx = np.in1d(sorted_idx[:, mn_sub], idx_selected, assume_unique=True)
                                sub_idx = sorted_idx[~chk_idx, mn_sub]

                                idx_selected = np.concatenate((idx_selected, sub_idx[range(Ni_sub_each)]))
                    # select tiles with the probabilities
                    else:
                        for mn_sub in range_score_dims:
                            Ni_cur = Ni_sub_each_fst if mn_sub == 0 else Ni_sub_each

                            score_sub = cur_score[:, mn_sub]/np.sum(cur_score[:, mn_sub])
                            if np.sum(np.isnan(score_sub)) == 0:
                                idx_selected_sub = np.random.choice(\
                                    np.arange(Ni), size=Ni_cur, replace=False, p=score_sub)
                            else:
                                print(score_sub)
                                idx_selected_sub = np.random.choice(\
                                    np.arange(Ni), size=Ni_cur, replace=False)

                            if mn_sub == 0:
                                idx_selected = idx_selected_sub
                            else:
                                idx_selected = np.concatenate((idx_selected, idx_selected_sub))

                    idx_selected = np.sort(idx_selected)

                    Ndiff = Ni_sel - len(idx_selected)
                    if Ndiff > 0:
                        idx_selected = np.concatenate(
                            (idx_selected, np.random.choice(np.setdiff1d(range(Ni), idx_selected), size=Ndiff, replace=False)))
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

    def predict(self, tst_index, data_set_=None, \
                selection_mode=None, sub_Ni=None, rep_num=1, flag_trndata=False):
        vec_colwise_np = lambda x: np.reshape(x, [-1, 1])
        vec_rowwise_np = lambda x: np.reshape(x, [1, -1])
        vec_flatten_np = lambda x: np.reshape(x, [-1])

        if data_set_ is None:
            # data_filenames = self.data_filenames
            data_set_ = self.data_X

        if selection_mode is None:
            selection_mode =  self.selection_mode

        if sub_Ni is None:
            sub_Ni = self.sub_Ni

        # calcuate embeddings
        Ntrn = len(self.trn_index)

        flag_cal_again = True
        if self.mat_trn_est is not None:
                if self.mat_trn_est.shape[1] == (rep_num * self.nMCsamples):
                    flag_cal_again = False

        if flag_cal_again == True:
            mat_trn = self.model_eval(self.trn_index, data_set_=self.data_X, \
                                      sub_Ni=sub_Ni, selection_mode=selection_mode, rep_num=rep_num)
            self.mat_trn_est = mat_trn
        else:
            mat_trn = self.mat_trn_est

        # for the testing data
        mat_tst = self.model_eval\
            (tst_index, data_set_=data_set_, sub_Ni=sub_Ni, selection_mode=selection_mode, rep_num=rep_num)

        # embeddings from the test data
        idx_pos = np.reshape(self.Ytrn == 1.0, [-1])
        idx_neg = np.reshape(self.Ytrn == 0.0, [-1])

        N_pos = np.sum(idx_pos)
        N_neg = np.sum(idx_neg)

        mat_trn_pos = tf.boolean_mask(mat_trn, idx_pos, axis=0)
        mat_trn_neg = tf.boolean_mask(mat_trn, idx_neg, axis=0)

        #
        idx_pos_ex = vec_flatten_np(np.tile(vec_rowwise_np(range(N_pos)), reps=N_neg))
        idx_neg_ex = np.repeat(vec_colwise_np(range(N_neg)), repeats=N_pos)

        # for training data points
        Ytrn_probs = np.zeros((Ntrn, 1))
        for mn_i in range(Ntrn):
            tmp_anc = tf.expand_dims(tf.gather(mat_trn, mn_i, axis=0), axis=0)

            if self.Ytrn[mn_i] == 1.0:
                idx = np.sum(self.Ytrn[range(mn_i+1)] == 1)
                idx_selected = ~(idx_pos_ex == (idx-1))
            else:
                idx = np.sum(self.Ytrn[range(mn_i+1)] == 0)
                idx_selected = ~(idx_neg_ex == (idx-1))

            # prediction
            tmp_pos_dist = reduce_sum(square(mat_trn_pos - tmp_anc), axis=2)
            tmp_neg_dist = reduce_sum(square(mat_trn_neg - tmp_anc), axis=2)

            idx_pos_ex_new = tf.boolean_mask(idx_pos_ex, idx_selected)
            idx_neg_ex_new = tf.boolean_mask(idx_neg_ex, idx_selected)

            tmp_diff_dist = tf.gather(tmp_neg_dist, idx_neg_ex_new, axis=0) \
                            - tf.gather(tmp_pos_dist, idx_pos_ex_new, axis=0)

            prob_sub = self.select_sub_percnt(reduce_mean(exp(-logistic_loss(tmp_diff_dist)), axis=1))
            Ytrn_probs[mn_i] = K.eval(reduce_mean(prob_sub))

        if flag_trndata:
            return Ytrn_probs

        # for test data points
        Ytst_probs = np.zeros((len(tst_index), 1))
        for mn_i in range(len(tst_index)):
            tmp_anc = tf.expand_dims(tf.gather(mat_tst, mn_i, axis=0), axis=0)

            tmp_pos_dist = reduce_sum(square(mat_trn_pos - tmp_anc), axis=2)
            tmp_neg_dist = reduce_sum(square(mat_trn_neg - tmp_anc), axis=2)

            # prediction
            tmp_diff_dist = tf.gather(tmp_neg_dist, idx_neg_ex, axis=0) \
                            - tf.gather(tmp_pos_dist, idx_pos_ex, axis=0)

            prob_pos = self.select_sub_percnt(reduce_mean(exp(-logistic_loss(tmp_diff_dist)), axis=1))
            Ytst_probs[mn_i] = K.eval(reduce_mean(prob_pos))

        # Platt's method
        A, B = plattFit(Ytrn_probs, self.Ytrn)  # rescling-coefficients

        Ytrn_probs_adj = sigmoid_platt(Ytrn_probs, A, B)
        Ytst_probs_adj = sigmoid_platt(Ytst_probs, A, B)

        return Ytrn_probs, Ytrn_probs_adj, Ytst_probs, Ytst_probs_adj

    def model_eval(self, tst_index, data_set_, sub_Ni, selection_mode, rep_num, batch_size=5):
        Ntst = len(tst_index)

        for mn_i in range(np.int_(np.ceil(Ntst/batch_size))):
            if Ntst== 1:
                index_vec = tst_index
            else:
                sub_idx = range(mn_i * batch_size, np.minimum((mn_i + 1) * batch_size, Ntst))
                index_vec = tst_index[np.hstack(sub_idx)]

            set_indices = self.mark_subImgs \
                (data_set_, index_vec, sub_Ni=sub_Ni, selection_mode=selection_mode, rep_num=rep_num)

            for mn_sub in range(rep_num):
                X, X_idx = self.gen_input_fromList(data_set_, index_vec, set_indices[mn_sub])
                out_ = self.model(X, X_idx)

                if mn_sub == 0:
                    out_sub = out_
                else:
                    out_sub = tf.concat((out_sub, out_), axis=1)

            if mn_i == 0:
                outputs = out_sub
            else:
                outputs = tf.concat((outputs, out_sub), axis=0)

        return outputs

    def pred_tileWeights(self, data_set_=None, str_filenaem=None):
        col_names = [['pID'], ['tileID'], [str(idx) + 'th_weight' for idx in range(self.num_Att)], ['mean_weight']]
        col_names = list(itertools.chain(*col_names))

        for idx in range(len(data_set_.sample_ids)):
            scores = self.model(data_set_.data_mat[idx])

            probs = scores - tf.reduce_max(scores, axis=0, keepdims=True)
            probs_logexpsum = logsumexp(probs, axis=0, keepdims=True)
            probs = reduce_mean(exp(probs - probs_logexpsum), axis=0)

            tileID = data_set_.sample_ids_tiles[idx]
            pID = np.repeat(data_set_.sample_ids[idx], repeats=len(tileID))

            probs_mat = K.eval(tf.concat((probs, vec_colwise(reduce_mean(probs, axis=1))), axis=1))
            df_sub = pd.concat([pd.DataFrame(zip(pID, tileID)), pd.DataFrame(probs_mat)], axis=1)
            if idx == 0:
                df_Info = df_sub
            else:
                df_Info = pd.concat([df_Info, df_sub], axis=0)

        df_Info.columns = col_names
        df_Info.to_csv(str_filenaem, header=True, index=None, mode='w', sep='\t')

        return None
