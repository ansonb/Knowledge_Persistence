from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.common import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer
from pykg2vec.utils.evaluator import Evaluator
import sys
from scipy import sparse
import numpy as np
import os

import make_preds_graph
from argparse import ArgumentParser
import subprocess
import pandas
import traceback

parser = ArgumentParser()


parser.add_argument('-mn', '--model_name', default='TransE', type=str)
parser.add_argument('-d', '--dataset', default='fb15k_237', type=str)
parser.add_argument('-fe', '--final_epoch', required=True, type=int, help="final epoch for which training was run for this model and dataset pair")
parser.add_argument('-mfol', '--saved_models_folder', default='', type=str, help='folder to read the saved model files from')
parser.add_argument('-b', '--batch_size', default=1000, type=int, help='batch size to evaluate on')
parser.add_argument('-dsp', '--dataset_path', default='', type=str, help='dataset folder')
parser.add_argument('-ifol', '--in_folder', default='/projects/Graphs_TDA_ML/KG_TDA/results2', type=str, help='root folder containing training results')
parser.add_argument('-ofol', '--out_folder', default='', type=str, help='output folder to write the graphs')
parser.add_argument('-ns', '--neg_samples', action='store_true', help='Whether to use negative samples')
parser.add_argument('-ds', '--data_split', default='train', type=str, help='Whether to use train/test split for making graphs')
parser.add_argument('-seed', '--seed', default=42, type=int, help='Random seed for sampling') # TODO: handle None
parser.add_argument('-hpf', '--hparams_file', default=None, type=str, help='Hyperparameter file location')
parser.add_argument('-saved_model_not_necessary', '--saved_model_not_necessary', default=False, type=bool, help='Continue graph creation even without trained model')




def main():
    args = parser.parse_args()
    root_folder = args.in_folder
    try:
        if args.final_epoch==-1:
            results_df = pandas.read_csv("{0}/summary/{1}/{2}/{3}_Testing_results_0.csv".format(
                    root_folder, args.model_name, args.dataset, args.model_name.lower()), encoding='utf-8')
        else:
            results_df = pandas.read_csv("{0}/summary/{1}/{2}/{3}_{4}_Testing_results_0.csv".format(
                    root_folder, args.model_name, args.dataset, args.model_name.lower(), 
                    args.final_epoch), encoding='utf-8')

        pd_epoch_intervals = list(results_df['Epoch'])
    except Exception as e:
        print('== Exception: ',e)
        if args.saved_model_not_necessary:
            pd_epoch_intervals = [0]
        else:
            traceback.print_exc()
            raise

    

    for epoch in pd_epoch_intervals:
        ld_file_loc = os.path.join(args.saved_models_folder,args.model_name.lower(),str(epoch))
        if os.path.exists(ld_file_loc):
            if args.out_folder=='':
                result_folder = '{}/summary/{}/{}/{}'.format(root_folder,args.model_name,args.dataset,'Graphs_neg')
            else:
                result_folder = args.out_folder
            if not args.neg_samples:
                cmd = "python make_preds_graph.py -ds='{}' -dsp='{}' -b={} -mn='{}' -r='{}' -exp=True -es=False -device='cuda' -ld='{}' -seed={} -hpf='{}'".format(args.dataset,args.dataset_path,args.batch_size,args.model_name,result_folder,ld_file_loc,args.seed,args.hparams_file)
            else:
                if args.data_split=='train':
                    cmd = "python make_preds_graph_with_negsamples.py -ds='{}' -dsp='{}' -b={} -mn='{}' -r='{}' -exp=True -es=False -device='cuda' -ld='{}' -seed={} -hpf='{}'".format(args.dataset,args.dataset_path,args.batch_size,args.model_name,result_folder,ld_file_loc,args.seed,args.hparams_file)
                else:
                    cmd = "python make_preds_graph_with_negsamples_test.py -ds='{}' -dsp='{}' -b={} -mn='{}' -r='{}' -exp=True -es=False -device='cuda' -ld='{}' -seed={} -hpf='{}'".format(args.dataset,args.dataset_path,args.batch_size,args.model_name,result_folder,ld_file_loc,args.seed,args.hparams_file)
            cmd_output = subprocess.check_output(cmd, shell=True)
            print(cmd_output)
        elif args.saved_model_not_necessary:
            if args.out_folder=='':
                result_folder = '{}/summary/{}/{}/{}'.format(root_folder,args.model_name,args.dataset,'Graphs_neg')
            else:
                result_folder = args.out_folder
            if not args.neg_samples:
                cmd = "python make_preds_graph.py -ds='{}' -dsp='{}' -b={} -mn='{}' -r='{}' -exp=True -es=False -device='cuda' -seed={} -hpf='{}'".format(args.dataset,args.dataset_path,args.batch_size,args.model_name,result_folder,args.seed,args.hparams_file)
            else:
                if args.data_split=='train':
                    cmd = "python make_preds_graph_with_negsamples.py -ds='{}' -dsp='{}' -b={} -mn='{}' -r='{}' -exp=True -es=False -device='cuda' -seed={} -hpf='{}'".format(args.dataset,args.dataset_path,args.batch_size,args.model_name,result_folder,args.seed,args.hparams_file)
                else:
                    cmd = "python make_preds_graph_with_negsamples_test.py -ds='{}' -dsp='{}' -b={} -mn='{}' -r='{}' -exp=True -es=False -device='cuda' -seed={} -hpf='{}'".format(args.dataset,args.dataset_path,args.batch_size,args.model_name,result_folder,args.seed,args.hparams_file)
            cmd_output = subprocess.check_output(cmd, shell=True)
            print(cmd_output)

if __name__ == "__main__":
    main()