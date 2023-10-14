from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.common import Importer, KGEArgParser
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
	config = config_def(args)
	model = model_def(**config.__dict__)

	# Create, Compile and evaluate the model
	trainer = Trainer(model, config)
	trainer.build_model()
	# trainer.train_model()
	evaluator = Evaluator(trainer.model, config)
	preds = evaluator.get_graph_with_edge_scores(data=evaluator.train_data, test_batch_size=args.batch_size)
	pred_graph, original_graph = get_sparse_graph_spoptim(preds, evaluator.train_data, config.tot_entity, config.tot_relation)

	out_folder = os.path.join(args.result,'Graphs')
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)
	try:
		epoch = int(config.load_from_data.split('/')[-1])
		out_file_preds = os.path.join(out_folder,'preds_{}.npz'.format(epoch))
		sparse.save_npz(out_file_preds, pred_graph)
		out_file_graph = os.path.join(out_folder,'org_{}.npz'.format(epoch))
	except Exception as e:
	    out_file_preds = os.path.join(out_folder,'preds.npz')
	    sparse.save_npz(out_file_preds, pred_graph)
	    out_file_graph = os.path.join(out_folder,'org.npz')

	sparse.save_npz(out_file_graph, original_graph)

if __name__ == "__main__":
    main()