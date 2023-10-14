from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.common import Importer, KGEArgParser, HyperparameterLoader
from pykg2vec.utils.trainer import Trainer
from pykg2vec.utils.evaluator import Evaluator
import sys
from scipy import sparse
import numpy as np
import os

def get_sparse_graph(preds, data, num_entities, num_relations):	
	adj_tensor = np.ones((num_entities, num_entities, num_relations))*np.finfo(np.float).max
	for i in range(len(data)):
		h, r, t = data[i].h, data[i].r, data[i].t
		adj_tensor[h,t,r] = preds[i]

	adj_mat = np.min(adj_tensor, axis=-1)

	thresh = np.max(preds)
	[I, J] = np.meshgrid(np.arange(num_entities), np.arange(num_entities))
	I = I[adj_mat <= thresh]
	J = J[adj_mat <= thresh]
	V = adj_mat[adj_mat <= thresh]
	N = num_entities
	pred_graph = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

	min_pred = np.min(preds)
	Vg = np.ones(V.shape)*min_pred
	N = num_entities
	original_graph = sparse.coo_matrix((Vg, (I, J)), shape=(N, N)).tocsr()

	return pred_graph, original_graph

def get_sparse_graph_spoptim(preds, data, num_entities, num_relations):	
	adj_tensor = {}
	thresh = -1e8
	for i in range(len(data)):
		h, r, t = data[i].h, data[i].r, data[i].t
		key_ = '{},{}'.format(h,t)
		adj_tensor[key_] = adj_tensor.get(key_, [])
		adj_tensor[key_].append(preds[i])
		if preds[i]>thresh:
			thresh = preds[i]

	I, J, V = [], [], []
	for ht_pair, pred in adj_tensor.items():
		h, t = [int(i) for i in ht_pair.split(',')]
		I.append(h)
		J.append(t)
		V.append(np.min(pred))
	N = num_entities
	pred_graph = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

	min_pred = np.min(preds)
	Vg = np.ones((len(V)))*min_pred
	N = num_entities
	original_graph = sparse.coo_matrix((Vg, (I, J)), shape=(N, N)).tocsr()

	return pred_graph, original_graph

def main():
	# getting the customized configurations from the command-line arguments.
	args = KGEArgParser().get_args(sys.argv[1:])

	# Preparing data and cache the data for later usage
	knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, custom_dataset_path=args.dataset_path)
	knowledge_graph.prepare_data()

	# Extracting the corresponding model config and definition from Importer().
	config_def, model_def = Importer().import_model_config(args.model_name.lower())
	if args.hp_abs_file is None:
		config = config_def(args)
	else:
		args.exp = False # to avoid loading default params
		config = config_def(args)
		# print('== hp_abs_file: ',args.hp_abs_file)
		params = HyperparameterLoader(args).load_hyperparameter(args.dataset_name, args.model_name)
		for key, value in params.items():
			config.__dict__[key] = value # copy all the setting from the hparam file.
	model = model_def(**config.__dict__)

	# Create, Compile and evaluate the model
	trainer = Trainer(model, config)
	trainer.build_model()
	# trainer.train_model()
	evaluator = Evaluator(trainer.model, config, seed=args.seed)

	neg_data = evaluator.sample_negative_rels(sampling='unif',neg_rate=1.0,pos_data_len=len(evaluator.train_data))

	preds_pos = evaluator.get_graph_with_edge_scores(data=evaluator.train_data, test_batch_size=args.batch_size)
	preds_neg = evaluator.get_graph_with_edge_scores(data=neg_data, test_batch_size=args.batch_size)

	pred_graph_pos, original_graph = get_sparse_graph_spoptim(preds_pos, evaluator.train_data, config.tot_entity, config.tot_relation)
	pred_graph_neg, _ = get_sparse_graph_spoptim(preds_neg, neg_data, config.tot_entity, config.tot_relation)

	# out_folder = os.path.join(args.result,'Graphs_neg')
	out_folder = args.result
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)
	try:
		epoch = int(config.load_from_data.split('/')[-1])
		out_file_preds_pos = os.path.join(out_folder,'preds_pos_{}.npz'.format(epoch))
		out_file_preds_neg = os.path.join(out_folder,'preds_neg_{}.npz'.format(epoch))
		out_file_graph = os.path.join(out_folder,'org_{}.npz'.format(epoch))
	except Exception as e:
	    out_file_preds_pos = os.path.join(out_folder,'preds_pos.npz')
	    out_file_preds_neg = os.path.join(out_folder,'preds_neg.npz')
	    out_file_graph = os.path.join(out_folder,'org.npz')
	
	sparse.save_npz(out_file_preds_pos, pred_graph_pos)
	sparse.save_npz(out_file_preds_neg, pred_graph_neg)
	sparse.save_npz(out_file_graph, original_graph)

	del knowledge_graph
	del model
	del evaluator

if __name__ == "__main__":
    main()