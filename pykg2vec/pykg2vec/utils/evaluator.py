#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for evaluating the results
"""
import os
import timeit
import torch
import numpy as np
import pandas as pd
from pykg2vec.utils.logger import Logger
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

class MetricCalculator:
    '''
        MetricCalculator aims to
        1) address all the statistic tasks.
        2) provide interfaces for querying results.

        MetricCalculator is expected to be used by "evaluation_process".
    '''
    _logger = Logger().get_logger(__name__)

    def __init__(self, config):
        self.config = config

        self.hr_t = config.knowledge_graph.read_cache_data('hr_t')
        self.tr_h = config.knowledge_graph.read_cache_data('tr_h')

        # (f)mr  : (filtered) mean rank
        # (f)mrr : (filtered) mean reciprocal rank
        # (f)hit : (filtered) hit-k ratio
        self.mr = {}
        self.fmr = {}
        self.mrr = {}
        self.fmrr = {}
        self.hit = {}
        self.fhit = {}

        self.epoch = None

        self.reset()

    def reset(self):
        # temporarily used buffers and indexes.
        self.rank_head = []
        self.rank_tail = []
        self.f_rank_head = []
        self.f_rank_tail = []
        self.epoch = None
        self.start_time = timeit.default_timer()

    def append_result(self, result):
        predict_tail = result[0]
        predict_head = result[1]

        h, r, t = result[2], result[3], result[4]

        self.epoch = result[5]

        t_rank, f_t_rank = self.get_tail_rank(predict_tail, h, r, t)
        h_rank, f_h_rank = self.get_head_rank(predict_head, h, r, t)

        self.rank_head.append(h_rank)
        self.rank_tail.append(t_rank)
        self.f_rank_head.append(f_h_rank)
        self.f_rank_tail.append(f_t_rank)

    def get_tail_rank(self, tail_candidate, h, r, t):
        """Function to evaluate the tail rank.

           Args:
               id_replace_tail (list): List of the predicted tails for the given head, relation pair
               h (int): head id
               r (int): relation id
               t (int): tail id
               hr_t (dict): list of tails for the given hwS and relation pari.

            Returns:
                Tensors: Returns tail rank and filetered tail rank
        """
        trank = 0
        ftrank = 0

        for j in range(len(tail_candidate)):
            val = tail_candidate[-j - 1]
            if val != t:
                trank += 1
                ftrank += 1
                if val in self.hr_t[(h, r)]:
                    ftrank -= 1
            else:
                break

        return trank, ftrank

    def get_head_rank(self, head_candidate, h, r, t):
        """Function to evaluate the head rank.

           Args:
               head_candidate (list): List of the predicted head for the given tail, relation pair
               h (int): head id
               r (int): relation id
               t (int): tail id

            Returns:
                Tensors: Returns head  rank and filetered head rank
        """
        hrank = 0
        fhrank = 0

        for j in range(len(head_candidate)):
            val = head_candidate[-j - 1]
            if val != h:
                hrank += 1
                fhrank += 1
                if val in self.tr_h[(t, r)]:
                    fhrank -= 1
            else:
                break

        return hrank, fhrank

    def settle(self):
        head_ranks = np.asarray(self.rank_head, dtype=np.float32)+1
        tail_ranks = np.asarray(self.rank_tail, dtype=np.float32)+1
        head_franks = np.asarray(self.f_rank_head, dtype=np.float32)+1
        tail_franks = np.asarray(self.f_rank_tail, dtype=np.float32)+1

        ranks = np.concatenate((head_ranks, tail_ranks))
        franks = np.concatenate((head_franks, tail_franks))

        self.mr[self.epoch] = np.mean(ranks)
        self.mrr[self.epoch] = np.mean(np.reciprocal(ranks))
        self.fmr[self.epoch] = np.mean(franks)
        self.fmrr[self.epoch] = np.mean(np.reciprocal(franks))

        for hit in self.config.hits:
            self.hit[(self.epoch, hit)] = np.mean(ranks <= hit, dtype=np.float32)
            self.fhit[(self.epoch, hit)] = np.mean(franks <= hit, dtype=np.float32)

    def get_curr_scores(self):
        scores = {'mr': self.mr[self.epoch],
                  'fmr':self.fmr[self.epoch],
                  'mrr':self.mrr[self.epoch],
                  'fmrr':self.fmrr[self.epoch]}
        return scores


    def save_test_summary(self, model_name):
        """Function to save the test of the summary.

            Args:
                model_name (str): specify the name of the model.

        """
        files = os.listdir(str(self.config.path_result))
        l = len([f for f in files if model_name in f if 'Testing' in f])
        with open(str(self.config.path_result / (model_name + '_summary_' + str(l) + '.txt')), 'w') as fh:
            fh.write('----------------SUMMARY----------------\n')
            for key, val in self.config.__dict__.items():
                if 'gpu' in key:
                    continue
                if 'knowledge_graph' in key:
                    continue
                if not isinstance(val, str):
                    if isinstance(val, list):
                        v_tmp = '['
                        for i, v in enumerate(val):
                            if i == 0:
                                v_tmp += str(v)
                            else:
                                v_tmp += ',' + str(v)
                        v_tmp += ']'
                        val = v_tmp
                    else:
                        val = str(val)
                fh.write(key + ':' + val + '\n')
            fh.write('-----------------------------------------\n')
            fh.write("\n----------Metadata Info for Dataset:%s----------------" % self.config.knowledge_graph.dataset_name)
            fh.write("Total Training Triples   :%d\n"%self.config.tot_train_triples)
            fh.write("Total Testing Triples    :%d\n"%self.config.tot_test_triples)
            fh.write("Total validation Triples :%d\n"%self.config.tot_valid_triples)
            fh.write("Total Entities           :%d\n"%self.config.tot_entity)
            fh.write("Total Relations          :%d\n"%self.config.tot_relation)
            fh.write("---------------------------------------------")

        columns = ['Epoch', 'Mean Rank', 'Filtered Mean Rank', 'Mean Reciprocal Rank', 'Filtered Mean Reciprocal Rank']
        for hit in self.config.hits:
            columns += ['Hit-%d Ratio'%hit, 'Filtered Hit-%d Ratio'%hit]

        results = []
        for epoch, _ in self.mr.items():
            res_tmp = [epoch, self.mr[epoch], self.fmr[epoch], self.mrr[epoch], self.fmrr[epoch]]

            for hit in self.config.hits:
                res_tmp.append(self.hit[(epoch, hit)])
                res_tmp.append(self.fhit[(epoch, hit)])

            results.append(res_tmp)

        df = pd.DataFrame(results, columns=columns)

        with open(str(self.config.path_result / (model_name + '_Testing_results_' + str(l) + '.csv')), 'a') as fh:
            df.to_csv(fh)

    def display_summary(self):
        """Function to print the test summary."""
        stop_time = timeit.default_timer()
        test_results = []
        test_results.append('')
        test_results.append("------Test Results for %s: Epoch: %d --- time: %.2f------------" % (self.config.dataset_name, self.epoch, stop_time - self.start_time))
        test_results.append('--# of entities, # of relations: %d, %d'%(self.config.tot_entity, self.config.tot_relation))
        test_results.append('--mr,  filtered mr             : %.4f, %.4f'%(self.mr[self.epoch], self.fmr[self.epoch]))
        test_results.append('--mrr, filtered mrr            : %.4f, %.4f'%(self.mrr[self.epoch], self.fmrr[self.epoch]))
        for hit in self.config.hits:
            test_results.append('--hits%d                        : %.4f ' % (hit, (self.hit[(self.epoch, hit)])))
            test_results.append('--filtered hits%d               : %.4f ' % (hit, (self.fhit[(self.epoch, hit)])))
        test_results.append("---------------------------------------------------------")
        test_results.append('')
        self._logger.info("\n".join(test_results))

class StratMetricCalculator:
    '''
        MetricCalculator aims to
        1) address all the statistic tasks.
        2) provide interfaces for querying results.

        MetricCalculator is expected to be used by "evaluation_process".
    '''
    _logger = Logger().get_logger(__name__)

    def __init__(self, config):
        self.config = config

        self.hr_t = config.knowledge_graph.read_cache_data('hr_t')
        self.tr_h = config.knowledge_graph.read_cache_data('tr_h')

        # (f)mr  : (filtered) mean rank
        # (f)mrr : (filtered) mean reciprocal rank
        # (f)hit : (filtered) hit-k ratio
        self.mr = {}
        self.fmr = {}
        self.mrr = {}
        self.fmrr = {}
        self.hit = {}
        self.fhit = {}

        self.popularity_e = {}
        self.popularity_r = {}
        self.w_e = {}
        self.w_r = {}
        self.beta_e = config.__dict__.get('beta_e',-1)
        self.beta_r = config.__dict__.get('beta_r',0)

        self.epoch = None

        self.reset()

    def reset(self):
        # temporarily used buffers and indexes.
        self.rank_head = []
        self.rank_tail = []
        self.f_rank_head = []
        self.f_rank_tail = []
        self.hits_weights_head = []
        self.mrr_weights_head = []
        self.hits_weights_tail = []
        self.mrr_weights_tail = []
        self.epoch = None
        self.start_time = timeit.default_timer()

        for (h, r), t_dict in self.hr_t.items():
            for t in t_dict:
                self.popularity_e[h] = self.popularity_e.get(h,0) + 1
                self.popularity_e[t] = self.popularity_e.get(t,0) + 1
                self.popularity_r[r] = self.popularity_r.get(r,0) + 1

                self.w_e[h] = (1./self.popularity_e[h])**(self.beta_e)
                self.w_e[t] = (1./self.popularity_e[t])**(self.beta_e)
                self.w_r[r] = (1./self.popularity_r[r])**(self.beta_r)


    def append_result(self, result):
        predict_tail = result[0]
        predict_head = result[1]

        h, r, t = result[2], result[3], result[4]

        self.epoch = result[5]

        t_rank, f_t_rank = self.get_tail_rank(predict_tail, h, r, t)
        h_rank, f_h_rank = self.get_head_rank(predict_head, h, r, t)

        self.rank_head.append(h_rank)
        self.hits_weights_head.append(self.w_e[t]*1./(self.w_e[h]+self.w_e[t])*1./self.popularity_r[r])
        self.mrr_weights_head.append(self.w_e[t]*1./(self.w_e[h]+self.w_e[t])*1./self.popularity_r[r])
        self.rank_tail.append(t_rank)
        self.hits_weights_tail.append(self.w_e[h]*1./(self.w_e[h]+self.w_e[t])*1./self.popularity_r[r])
        self.mrr_weights_tail.append(self.w_e[h]*1./(self.w_e[h]+self.w_e[t])*1./self.popularity_r[r])
        self.f_rank_head.append(f_h_rank)
        self.f_rank_tail.append(f_t_rank)
        

    def get_tail_rank(self, tail_candidate, h, r, t):
        """Function to evaluate the tail rank.

           Args:
               id_replace_tail (list): List of the predicted tails for the given head, relation pair
               h (int): head id
               r (int): relation id
               t (int): tail id
               hr_t (dict): list of tails for the given hwS and relation pari.

            Returns:
                Tensors: Returns tail rank and filetered tail rank
        """
        trank = 0
        ftrank = 0

        for j in range(len(tail_candidate)):
            val = tail_candidate[-j - 1]
            if val != t:
                trank += 1
                ftrank += 1
                if val in self.hr_t[(h, r)]:
                    ftrank -= 1
            else:
                break

        return trank, ftrank

    def get_head_rank(self, head_candidate, h, r, t):
        """Function to evaluate the head rank.

           Args:
               head_candidate (list): List of the predicted head for the given tail, relation pair
               h (int): head id
               r (int): relation id
               t (int): tail id

            Returns:
                Tensors: Returns head  rank and filetered head rank
        """
        hrank = 0
        fhrank = 0

        for j in range(len(head_candidate)):
            val = head_candidate[-j - 1]
            if val != h:
                hrank += 1
                fhrank += 1
                if val in self.tr_h[(t, r)]:
                    fhrank -= 1
            else:
                break

        return hrank, fhrank

    def settle(self):
        head_ranks = np.asarray(self.rank_head, dtype=np.float32)+1
        tail_ranks = np.asarray(self.rank_tail, dtype=np.float32)+1
        head_franks = np.asarray(self.f_rank_head, dtype=np.float32)+1
        tail_franks = np.asarray(self.f_rank_tail, dtype=np.float32)+1

        ranks = np.concatenate((head_ranks, tail_ranks))
        franks = np.concatenate((head_franks, tail_franks))
        hits_weights = np.concatenate((self.hits_weights_head, self.hits_weights_tail))
        mrr_weights = np.concatenate((self.mrr_weights_head, self.mrr_weights_tail))


        self.mr[self.epoch] = np.sum(mrr_weights*ranks)
        self.mrr[self.epoch] = np.sum(mrr_weights*np.reciprocal(ranks))
        self.fmr[self.epoch] = np.sum(mrr_weights*franks)
        self.fmrr[self.epoch] = np.sum(mrr_weights*np.reciprocal(franks))

        for hit in self.config.hits:
            self.hit[(self.epoch, hit)] = np.sum(hits_weights*(1.*(ranks <= hit)), dtype=np.float32)
            self.fhit[(self.epoch, hit)] = np.sum(hits_weights*(1.*(franks <= hit)), dtype=np.float32)

    def get_curr_scores(self):
        scores = {'mr': self.mr[self.epoch],
                  'fmr':self.fmr[self.epoch],
                  'mrr':self.mrr[self.epoch],
                  'fmrr':self.fmrr[self.epoch]}
        return scores


    def save_test_summary(self, model_name):
        """Function to save the test of the summary.

            Args:
                model_name (str): specify the name of the model.

        """
        files = os.listdir(str(self.config.path_result))
        l = len([f for f in files if model_name in f if 'Testing' in f])
        with open(str(self.config.path_result / (model_name + '_summary_' + str(l) + '.txt')), 'w') as fh:
            fh.write('----------------SUMMARY----------------\n')
            for key, val in self.config.__dict__.items():
                if 'gpu' in key:
                    continue
                if 'knowledge_graph' in key:
                    continue
                if not isinstance(val, str):
                    if isinstance(val, list):
                        v_tmp = '['
                        for i, v in enumerate(val):
                            if i == 0:
                                v_tmp += str(v)
                            else:
                                v_tmp += ',' + str(v)
                        v_tmp += ']'
                        val = v_tmp
                    else:
                        val = str(val)
                fh.write(key + ':' + val + '\n')
            fh.write('-----------------------------------------\n')
            fh.write("\n----------Metadata Info for Dataset:%s----------------" % self.config.knowledge_graph.dataset_name)
            fh.write("Total Training Triples   :%d\n"%self.config.tot_train_triples)
            fh.write("Total Testing Triples    :%d\n"%self.config.tot_test_triples)
            fh.write("Total validation Triples :%d\n"%self.config.tot_valid_triples)
            fh.write("Total Entities           :%d\n"%self.config.tot_entity)
            fh.write("Total Relations          :%d\n"%self.config.tot_relation)
            fh.write("---------------------------------------------")

        columns = ['Epoch', 'Stratified Mean Rank', 'Stratified Filtered Mean Rank', 'Stratified Mean Reciprocal Rank', 'Stratified Filtered Mean Reciprocal Rank']
        for hit in self.config.hits:
            columns += ['Stratified Hit-%d Ratio'%hit, 'Stratified Filtered Hit-%d Ratio'%hit]

        results = []
        for epoch, _ in self.mr.items():
            res_tmp = [epoch, self.mr[epoch], self.fmr[epoch], self.mrr[epoch], self.fmrr[epoch]]

            for hit in self.config.hits:
                res_tmp.append(self.hit[(epoch, hit)])
                res_tmp.append(self.fhit[(epoch, hit)])

            results.append(res_tmp)

        df = pd.DataFrame(results, columns=columns)

        with open(str(self.config.path_result / (model_name + '_Stratified_Testing_results_' + str(l) + '.csv')), 'a') as fh:
            df.to_csv(fh)

    def display_summary(self):
        """Function to print the test summary."""
        stop_time = timeit.default_timer()
        test_results = []
        test_results.append('')
        test_results.append("------Test Results for %s: Epoch: %d --- time: %.2f------------" % (self.config.dataset_name, self.epoch, stop_time - self.start_time))
        test_results.append('--# of entities, # of relations: %d, %d'%(self.config.tot_entity, self.config.tot_relation))
        test_results.append('--strat-mr,  strat-filtered mr             : %.4f, %.4f'%(self.mr[self.epoch], self.fmr[self.epoch]))
        test_results.append('--strat-mrr, strat-filtered mrr            : %.4f, %.4f'%(self.mrr[self.epoch], self.fmrr[self.epoch]))
        for hit in self.config.hits:
            test_results.append('--strat-hits%d                        : %.4f ' % (hit, (self.hit[(self.epoch, hit)])))
            test_results.append('--strat-filtered hits%d               : %.4f ' % (hit, (self.fhit[(self.epoch, hit)])))
        test_results.append("---------------------------------------------------------")
        test_results.append('')
        self._logger.info("\n".join(test_results))



class Evaluator:
    """Class to perform evaluation of the model.

        Args:
            model (object): Model object
            tuning (bool): Flag to denoting tuning if True

        Examples:
            >>> from pykg2vec.utils.evaluator import Evaluator
            >>> evaluator = Evaluator(model=model, tuning=True)
            >>> evaluator.test_batch(Session(), 0)
            >>> acc = evaluator.output_queue.get()
            >>> evaluator.stop()
    """
    _logger = Logger().get_logger(__name__)

    def __init__(self, model, config, tuning=False, seed=None):
        self.model = model
        self.config = config
        self.tuning = tuning
        self.train_data = self.config.knowledge_graph.read_cache_data('triplets_train')
        self.test_data = self.config.knowledge_graph.read_cache_data('triplets_test')
        self.eval_data = self.config.knowledge_graph.read_cache_data('triplets_valid')
        self.metric_calculator = MetricCalculator(self.config)
        self.strat_metric_calculator = StratMetricCalculator(self.config)

        # random seed for negative samples
        if seed is not None:
            np.random.seed(seed)

    def test_tail_rank(self, h, r, topk=-1):
        if hasattr(self.model, 'predict_tail_rank'):
            rank = self.model.predict_tail_rank(torch.LongTensor([h]).to(self.config.device), torch.LongTensor([r]).to(self.config.device), topk=topk)
            return rank.squeeze(0)

        h_batch = torch.LongTensor([h]).repeat([self.config.tot_entity]).to(self.config.device)
        r_batch = torch.LongTensor([r]).repeat([self.config.tot_entity]).to(self.config.device)
        entity_array = torch.LongTensor(list(range(self.config.tot_entity))).to(self.config.device)

        preds = self.model.forward(h_batch, r_batch, entity_array)
        _, rank = torch.topk(preds, k=topk)
        return rank

    def test_head_rank(self, r, t, topk=-1):
        if hasattr(self.model, 'predict_head_rank'):
            rank = self.model.predict_head_rank(torch.LongTensor([t]).to(self.config.device), torch.LongTensor([r]).to(self.config.device), topk=topk)
            return rank.squeeze(0)

        entity_array = torch.LongTensor(list(range(self.config.tot_entity))).to(self.config.device)
        r_batch = torch.LongTensor([r]).repeat([self.config.tot_entity]).to(self.config.device)
        t_batch = torch.LongTensor([t]).repeat([self.config.tot_entity]).to(self.config.device)

        preds = self.model.forward(entity_array, r_batch, t_batch)
        _, rank = torch.topk(preds, k=topk)
        return rank

    def test_rel_rank(self, h, t, topk=-1):
        if hasattr(self.model, 'predict_rel_rank'):
            # TODO: This is not implemented for conve, convkb, proje_pointwise and tucker
            rank = self.model.predict_rel_rank(h.to(self.config.device), t.to(self.config.device), topk=topk)
            return rank.squeeze(0)

        h_batch = torch.LongTensor([h]).repeat([self.config.tot_relation]).to(self.config.device)
        rel_array = torch.LongTensor(list(range(self.config.tot_relation))).to(self.config.device)
        t_batch = torch.LongTensor([t]).repeat([self.config.tot_relation]).to(self.config.device)

        preds = self.model.forward(h_batch, rel_array, t_batch)
        _, rank = torch.topk(preds, k=topk)
        return rank

    def mini_test(self, epoch=None):
        if self.config.test_num == 0:
            tot_valid_to_test = len(self.eval_data)
        else:
            tot_valid_to_test = min(self.config.test_num, len(self.eval_data))
        if self.config.debug:
            tot_valid_to_test = 10

        self._logger.info("Mini-Testing on [%d/%d] Triples in the valid set." % (tot_valid_to_test, len(self.eval_data)))
        return self.test(self.eval_data, tot_valid_to_test, epoch=epoch)

    def full_test(self, epoch=None):
        tot_valid_to_test = len(self.test_data)
        if self.config.debug:
            tot_valid_to_test = 10

        self._logger.info("Full-Testing on [%d/%d] Triples in the test set." % (tot_valid_to_test, len(self.test_data)))
        return self.test(self.test_data, tot_valid_to_test, epoch=epoch)

    def test(self, data, num_of_test, epoch=None):
        self.metric_calculator.reset()

        progress_bar = tqdm(range(num_of_test))
        for i in progress_bar:
            h, r, t = data[i].h, data[i].r, data[i].t

            # generate head batch and predict heads.
            h_tensor = torch.LongTensor([h])
            r_tensor = torch.LongTensor([r])
            t_tensor = torch.LongTensor([t])

            hrank = self.test_head_rank(r_tensor, t_tensor, self.config.tot_entity)
            trank = self.test_tail_rank(h_tensor, r_tensor, self.config.tot_entity)

            result_data = [trank.cpu().numpy(), hrank.cpu().numpy(), h, r, t, epoch]

            self.metric_calculator.append_result(result_data)

        self.metric_calculator.settle()
        self.metric_calculator.display_summary()

        if self.metric_calculator.epoch >= self.config.epochs - 1:
            self.metric_calculator.save_test_summary(self.model.model_name)

        return self.metric_calculator.get_curr_scores()

    def strat_full_test(self, epoch=None, save_summary_by_default=False):
        tot_valid_to_test = len(self.test_data)
        if self.config.debug:
            tot_valid_to_test = 10

        self._logger.info("Full-Testing on [%d/%d] Triples in the test set." % (tot_valid_to_test, len(self.test_data)))
        return self.strat_test(self.test_data, tot_valid_to_test, epoch=epoch, save_summary_by_default=save_summary_by_default)

    def strat_test(self, data, num_of_test, epoch=None, save_summary_by_default=False):
        self.strat_metric_calculator.reset()

        progress_bar = tqdm(range(num_of_test))
        for i in progress_bar:
            h, r, t = data[i].h, data[i].r, data[i].t

            # generate head batch and predict heads.
            h_tensor = torch.LongTensor([h])
            r_tensor = torch.LongTensor([r])
            t_tensor = torch.LongTensor([t])

            hrank = self.test_head_rank(r_tensor, t_tensor, self.config.tot_entity)
            trank = self.test_tail_rank(h_tensor, r_tensor, self.config.tot_entity)

            result_data = [trank.cpu().numpy(), hrank.cpu().numpy(), h, r, t, epoch]

            self.strat_metric_calculator.append_result(result_data)

        self.strat_metric_calculator.settle()
        self.strat_metric_calculator.display_summary()

        if self.strat_metric_calculator.epoch >= self.config.epochs - 1:
            self.strat_metric_calculator.save_test_summary(self.model.model_name)

        return self.strat_metric_calculator.get_curr_scores()


    def get_triple_score(self, h, r, t):
        if self.model.model_name.lower() in ['conve','tucker','proje_pointwise']:
            # pred_tails = self.model(h, r, direction="tail")[:,t]  # (h, r) -> hr_t forward
            # pred_heads = self.model(t, r, direction="head")[:,h]  # (t, r) -> tr_h backward
            hr_t_preds = self.model(h, r, direction="tail")
            pred_tails = torch.gather(hr_t_preds,dim=1,index=t.unsqueeze(1).repeat([1,hr_t_preds.shape[1]]))[:,0]
            tr_h_preds = self.model(t, r, direction="head")
            pred_heads = torch.gather(tr_h_preds,dim=1,index=h.unsqueeze(1).repeat([1,tr_h_preds.shape[1]]))[:,0]
            preds = (pred_tails+pred_heads)/2.0
            if self.model.model_name.lower() in ['conve','tucker']:
                preds = -preds
        else:
            preds = self.model.forward(h, r, t)
            # ConvKB -> -ve?

        return preds


    def get_pred_acc(self, pos_data, neg_data, neg_rate=1, batch_size=-1):
        """Function to get the triple classification accuracy.

           Args:
                h_pos, r_pos, t_pos: positive head, rel, tail
                h_neg, r_neg, t_neg: negative head, rel, tail; expects negative samples for a given (h_pos,r_pos,t_pos) to be subsequent in order
                neg_rate: number of negative samples used

            Returns:
                Tensors: triple classification accuracy
        """

        h_pos, r_pos, t_pos = [], [], []
        for i in range(len(pos_data)):
            h, r, t = pos_data[i].h, pos_data[i].r, pos_data[i].t
            h_pos.append(h)
            r_pos.append(r)
            t_pos.append(t)

        h_neg, r_neg, t_neg = [], [], []
        for i in range(len(neg_data)):
            h, r, t = neg_data[i].h, neg_data[i].r, neg_data[i].t
            h_neg.append(h)
            r_neg.append(r)
            t_neg.append(t)

        h_pos = torch.LongTensor(h_pos)
        r_pos = torch.LongTensor(r_pos)
        t_pos = torch.LongTensor(t_pos)
        if self.config.device=='cuda':
            h_pos = h_pos.cuda()
            r_pos = r_pos.cuda()
            t_pos = t_pos.cuda()

        h_neg = torch.LongTensor(h_neg)
        r_neg = torch.LongTensor(r_neg)
        t_neg = torch.LongTensor(t_neg)
        if self.config.device=='cuda':
            h_neg = h_neg.cuda()
            r_neg = r_neg.cuda()
            t_neg = t_neg.cuda()
        # TODO: implement batches if needed

        if batch_size>0:
            all_correct_preds = torch.empty((0,), dtype=float)
            num_batches = int(np.ceil(h_pos.shape[0]*1./batch_size))
            print('Computing triple score and classification accuracy...')
            for batch in tqdm(range(num_batches)):
                start_idx = batch*batch_size 
                end_idx = min((batch+1)*batch_size, h_pos.shape[0])
                h_pos_batch = h_pos[start_idx:end_idx]
                r_pos_batch = r_pos[start_idx:end_idx]
                t_pos_batch = t_pos[start_idx:end_idx]
                h_neg_batch = h_neg[start_idx:end_idx]
                r_neg_batch = r_neg[start_idx:end_idx]
                t_neg_batch = t_neg[start_idx:end_idx]

                pos_preds = self.get_triple_score(h_pos_batch, r_pos_batch, t_pos_batch)
                neg_preds = self.get_triple_score(h_neg_batch, r_neg_batch, t_neg_batch)
                neg_preds = neg_preds.reshape((h_pos_batch.shape[0], neg_rate))
                preds = torch.cat((pos_preds.reshape((pos_preds.shape[0],1)), neg_preds), dim=1)
                min_idx = torch.argmin(preds, dim=1)
                correct_preds = torch.where(min_idx==0, 1, 0).float()
                all_correct_preds = torch.cat((all_correct_preds, correct_preds), dim=0)
            acc = torch.mean( all_correct_preds )            
        else:
            pos_preds = self.get_triple_score(h_pos, r_pos, t_pos)
            neg_preds = self.get_triple_score(h_neg, r_neg, t_neg)
            neg_preds = neg_preds.reshape((h_pos.shape[0], neg_rate))
            all_preds = torch.cat((pos_preds.reshape((pos_preds.shape[0],1)), neg_preds), dim=1)
            min_idx = torch.argmin(all_preds, dim=1)
            acc = torch.mean( torch.where(min_idx==0, 1, 0).float() )

        return acc


    def get_pred_auc(self, pos_data, neg_data, neg_rate=1, batch_size=-1):
        """Function to get the triple classification AUC.

           Args:
                h_pos, r_pos, t_pos: positive head, rel, tail
                h_neg, r_neg, t_neg: negative head, rel, tail; expects negative samples for a given (h_pos,r_pos,t_pos) to be subsequent in order
                neg_rate: number of negative samples used

            Returns:
                Tensors: triple classification AUC
        """

        h_pos, r_pos, t_pos = [], [], []
        for i in range(len(pos_data)):
            h, r, t = pos_data[i].h, pos_data[i].r, pos_data[i].t
            h_pos.append(h)
            r_pos.append(r)
            t_pos.append(t)

        h_neg, r_neg, t_neg = [], [], []
        for i in range(len(neg_data)):
            h, r, t = neg_data[i].h, neg_data[i].r, neg_data[i].t
            h_neg.append(h)
            r_neg.append(r)
            t_neg.append(t)

        h_pos = torch.LongTensor(h_pos)
        r_pos = torch.LongTensor(r_pos)
        t_pos = torch.LongTensor(t_pos)
        if self.config.device=='cuda':
            h_pos = h_pos.cuda()
            r_pos = r_pos.cuda()
            t_pos = t_pos.cuda()

        h_neg = torch.LongTensor(h_neg)
        r_neg = torch.LongTensor(r_neg)
        t_neg = torch.LongTensor(t_neg)
        if self.config.device=='cuda':
            h_neg = h_neg.cuda()
            r_neg = r_neg.cuda()
            t_neg = t_neg.cuda()

        if batch_size>0:
            num_batches = int(np.ceil(h_pos.shape[0]*1./batch_size))
            print('Computing triple score and classification AUC...')
            all_pred_logits = torch.empty((0,), dtype=float)
            all_labels = torch.empty((0,), dtype=int)
            for batch in tqdm(range(num_batches)):
                start_idx = batch*batch_size 
                end_idx = min((batch+1)*batch_size, h_pos.shape[0])
                h_pos_batch = h_pos[start_idx:end_idx]
                r_pos_batch = r_pos[start_idx:end_idx]
                t_pos_batch = t_pos[start_idx:end_idx]
                h_neg_batch = h_neg[start_idx:end_idx]
                r_neg_batch = r_neg[start_idx:end_idx]
                t_neg_batch = t_neg[start_idx:end_idx]

                pos_preds = self.get_triple_score(h_pos_batch, r_pos_batch, t_pos_batch)
                labels_pos = torch.ones((pos_preds.shape[0],), dtype=int)
                neg_preds = self.get_triple_score(h_neg_batch, r_neg_batch, t_neg_batch)
                labels_neg = torch.ones((neg_preds.shape[0],), dtype=int)*0
                preds = torch.cat((pos_preds, neg_preds), dim=0)
                labels = torch.cat((labels_pos, labels_neg), dim=0)

                all_pred_logits = torch.cat((all_pred_logits,preds), dim=0)
                all_labels = torch.cat((all_labels,labels), dim=0)            
            auc = roc_auc_score(all_labels.detach().numpy(), all_pred_logits.detach().numpy())
        else:
            pos_preds = self.get_triple_score(h_pos, r_pos, t_pos)
            labels_pos = torch.ones((pos_preds.shape[0],), dtype=int)
            neg_preds = self.get_triple_score(h_neg, r_neg, t_neg)
            labels_neg = torch.ones((neg_preds.shape[0],), dtype=int)*0
            all_pred_logits = torch.cat((pos_preds, neg_preds), dim=0)
            all_labels = torch.cat((labels_pos, labels_neg), dim=0)
            auc = roc_auc_score(all_labels.detach().numpy(), all_pred_logits.detach().numpy())

        return auc

    def get_pred_rocacc(self, pos_data, neg_data, neg_rate=1, batch_size=-1):
        """Function to get the triple classification Accuracy.

           Args:
                h_pos, r_pos, t_pos: positive head, rel, tail
                h_neg, r_neg, t_neg: negative head, rel, tail; expects negative samples for a given (h_pos,r_pos,t_pos) to be subsequent in order
                neg_rate: number of negative samples used

            Returns:
                Tensors: triple classification Accuracy
        """

        h_pos, r_pos, t_pos = [], [], []
        for i in range(len(pos_data)):
            h, r, t = pos_data[i].h, pos_data[i].r, pos_data[i].t
            h_pos.append(h)
            r_pos.append(r)
            t_pos.append(t)

        h_neg, r_neg, t_neg = [], [], []
        for i in range(len(neg_data)):
            h, r, t = neg_data[i].h, neg_data[i].r, neg_data[i].t
            h_neg.append(h)
            r_neg.append(r)
            t_neg.append(t)

        h_pos = torch.LongTensor(h_pos)
        r_pos = torch.LongTensor(r_pos)
        t_pos = torch.LongTensor(t_pos)
        if self.config.device=='cuda':
            h_pos = h_pos.cuda()
            r_pos = r_pos.cuda()
            t_pos = t_pos.cuda()

        h_neg = torch.LongTensor(h_neg)
        r_neg = torch.LongTensor(r_neg)
        t_neg = torch.LongTensor(t_neg)
        if self.config.device=='cuda':
            h_neg = h_neg.cuda()
            r_neg = r_neg.cuda()
            t_neg = t_neg.cuda()

        if batch_size>0:
            num_batches = int(np.ceil(h_pos.shape[0]*1./batch_size))
            print('Computing triple score and classification AUC...')
            all_pred_logits = torch.empty((0,), dtype=float)
            all_labels = torch.empty((0,), dtype=int)
            for batch in tqdm(range(num_batches)):
                start_idx = batch*batch_size 
                end_idx = min((batch+1)*batch_size, h_pos.shape[0])
                h_pos_batch = h_pos[start_idx:end_idx]
                r_pos_batch = r_pos[start_idx:end_idx]
                t_pos_batch = t_pos[start_idx:end_idx]
                h_neg_batch = h_neg[start_idx:end_idx]
                r_neg_batch = r_neg[start_idx:end_idx]
                t_neg_batch = t_neg[start_idx:end_idx]

                pos_preds = self.get_triple_score(h_pos_batch, r_pos_batch, t_pos_batch)
                labels_pos = torch.ones((pos_preds.shape[0],), dtype=int)
                neg_preds = self.get_triple_score(h_neg_batch, r_neg_batch, t_neg_batch)
                labels_neg = torch.ones((neg_preds.shape[0],), dtype=int)*0
                preds = torch.cat((pos_preds, neg_preds), dim=0)
                labels = torch.cat((labels_pos, labels_neg), dim=0)

                all_pred_logits = torch.cat((all_pred_logits,preds), dim=0)
                all_labels = torch.cat((all_labels,labels), dim=0)  
            all_labels = all_labels.detach().numpy()
            all_pred_logits = all_pred_logits.detach().numpy()
            false_pos_rate, true_pos_rate, thresh = roc_curve(all_labels, all_pred_logits)        
        else:
            pos_preds = self.get_triple_score(h_pos, r_pos, t_pos)
            labels_pos = torch.ones((pos_preds.shape[0],), dtype=int)
            neg_preds = self.get_triple_score(h_neg, r_neg, t_neg)
            labels_neg = torch.ones((neg_preds.shape[0],), dtype=int)*0
            all_pred_logits = torch.cat((pos_preds, neg_preds), dim=0)
            all_labels = torch.cat((labels_pos, labels_neg), dim=0)
            all_labels = all_labels.detach().numpy()
            all_pred_logits = all_pred_logits.detach().numpy()
            false_pos_rate, true_pos_rate, thresh = roc_curve(all_labels, all_pred_logits)   

        true_neg_rate = 1-false_pos_rate
        optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - true_neg_rate), thresh)), key=lambda i: i[0], reverse=False)[0][1]
        # Get the accuracy
        roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in all_pred_logits]
        acc = np.mean( np.where(roc_predictions==all_labels, 1, 0) )

        return acc


    # Sample negative relations
    # sampling: uniform, bern
    def sample_negative_rels(self, sampling='unif', neg_rate=1.0, pos_data_len=-1):
        class Triple:
            def __init__(self, h, r, t):
                self.h = h
                self.r = r
                self.t = t

        
        positive_triples = {}
        h_batch, r_batch, t_batch = [], [], []
        for i in range(len(self.train_data)):
            h, r, t = self.train_data[i].h, self.train_data[i].r, self.train_data[i].t
            h_batch.append(h)
            r_batch.append(r)
            t_batch.append(t)
            positive_triples['{},{}->{}'.format(h,r,t)] = 1
        for i in range(len(self.test_data)):
            h, r, t = self.test_data[i].h, self.test_data[i].r, self.test_data[i].t
            h_batch.append(h)
            r_batch.append(r)
            t_batch.append(t)
            positive_triples['{},{}->{}'.format(h,r,t)] = 1
        for i in range(len(self.eval_data)):
            h, r, t = self.eval_data[i].h, self.eval_data[i].r, self.eval_data[i].t
            h_batch.append(h)
            r_batch.append(r)
            t_batch.append(t)
            positive_triples['{},{}->{}'.format(h,r,t)] = 1
        
        neg_samples = []
        if sampling=='unif':
            for h,r,t in zip(h_batch,r_batch,t_batch):
                neg_rate_int = np.ceil(neg_rate)
                if neg_rate_int%2==1:
                    neg_rate_int += 1 
                neg_matches_count = neg_rate_int
                while neg_matches_count>0:
                    hr = np.random.randint(self.config.tot_entity)
                    tr = np.random.randint(self.config.tot_entity)
                    triple_h = '{},{}->{}'.format(hr,r,t)
                    triple_t = '{},{}->{}'.format(h,r,tr)

                    if positive_triples.get(triple_h,0)==0 and positive_triples.get(triple_t,0)==0:
                        neg_samples.append(Triple(hr,r,t))
                        # neg_samples[0].append(hr)
                        # neg_samples[1].append(r)
                        # neg_samples[2].append(t)

                        neg_samples.append(Triple(h,r,tr))
                        # neg_samples[0].append(h)
                        # neg_samples[1].append(r)
                        # neg_samples[2].append(tr)

                        neg_matches_count -= 2

            neg_sample_indices = np.arange(len(neg_samples))
            np.random.shuffle(neg_sample_indices)
            if pos_data_len<=-1:
                num_neg_samples = int(neg_rate*(len(positive_triples)))
            else:
                num_neg_samples = int(neg_rate*pos_data_len)
            ret_neg_samples = np.array(neg_samples)[neg_sample_indices][:num_neg_samples]
        else:
            raise NotImplementedError

        return ret_neg_samples.tolist()

    # Sample negative relations
    # sampling: uniform, bern
    def sample_negative_rels_by_corrupting_triples(self, data, sampling='unif', neg_rate=1.0, pos_data_len=-1):
        class Triple:
            def __init__(self, h, r, t):
                self.h = h
                self.r = r
                self.t = t

        
        # positive_triples = {}
        h_batch, r_batch, t_batch = [], [], []
        for i in range(len(data)):
            h, r, t = data[i].h, data[i].r, data[i].t
            h_batch.append(h)
            r_batch.append(r)
            t_batch.append(t)
            # positive_triples['{},{}->{}'.format(h,r,t)] = 1
        
        
        neg_samples = []
        if sampling=='unif':
            for h,r,t in zip(h_batch,r_batch,t_batch):
                neg_rate_int = np.ceil(neg_rate)
                neg_matches_count = neg_rate_int
                cur_neg_samples = []
                while neg_matches_count>0:
                    # filtered corruption to get negative samples
                    hr = np.random.randint(self.config.tot_entity)
                    while hr in self.tr_h[(t, r)]:
                        hr = np.random.randint(self.config.tot_entity)
                    tr = np.random.randint(self.config.tot_entity)
                    while tr in self.hr_t[(h, r)]:
                        tr = np.random.randint(self.config.tot_entity)
                    # triple_h = '{},{}->{}'.format(hr,r,t)
                    # triple_t = '{},{}->{}'.format(h,r,tr)

                    # if positive_triples.get(triple_h,0)==0 and positive_triples.get(triple_t,0)==0:
                    cur_neg_samples.append(Triple(hr,r,t))
                    cur_neg_samples.append(Triple(h,r,tr))

                    neg_matches_count -= 2

                cur_neg_samples = np.random.choice(cur_neg_samples, neg_rate_int)
                neg_samples.extend(cur_neg_samples)

            ret_neg_samples = np.array(neg_samples)
        else:
            raise NotImplementedError

        return ret_neg_samples.tolist()

    def get_graph_with_edge_scores(self, data=None, test_batch_size=1000):
        self.model.eval()
        if data==None:
            data = self.test_data
        num_batches = int(np.ceil(len(data)/test_batch_size))

        all_preds = []
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx*test_batch_size
            end_idx = min(len(data),start_idx+test_batch_size)
            data_batch = data[start_idx:end_idx]
            h_batch, r_batch, t_batch = [], [], []
            for i in range(len(data_batch)):
                h, r, t = data[i].h, data[i].r, data[i].t
                h_batch.append(h)
                r_batch.append(r)
                t_batch.append(t)

            h_tensor = torch.LongTensor(h_batch)
            r_tensor = torch.LongTensor(r_batch)
            t_tensor = torch.LongTensor(t_batch)
            if self.config.device=='cuda':
                h_tensor = h_tensor.cuda()
                r_tensor = r_tensor.cuda()
                t_tensor = t_tensor.cuda()

            preds = self.get_triple_score(h_tensor,r_tensor,t_tensor)
            all_preds.extend(preds.cpu().detach().numpy())

        return all_preds