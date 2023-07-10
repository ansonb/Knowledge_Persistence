from ripser import ripser
import scipy
from argparse import ArgumentParser
import pickle
import numpy as np
import time
import sliced
import persim
import os
import gudhi
import ot
import time

start_time = time.time()

np.random.seed(0)


parser = ArgumentParser()

parser.add_argument('-pf', '--pos_preds_file', default='', type=str, help='pos preds graph file to read')
parser.add_argument('-npf', '--neg_preds_file', default='', type=str, help='neg preds graph file to read')
parser.add_argument('-org_f', '--org_file', default='', type=str, help='original graph file to read')
parser.add_argument('-of', '--out_file', default='./result_neg.pkl', type=str, help='file to write the result to')
parser.add_argument('-md', '--max_dim', default=1, type=int, help='maximum dimension of homology to compute')
parser.add_argument('-r', '--range', default='norm', type=str, help='Method to use to normalise the PDs')
parser.add_argument('--save_epoch_as_key', action="store_true", help='Whether to save epoch as a key in the result; used for finding correlation between all epochs')
parser.add_argument('-pdw', '--pd_weights_swd', action='store_true', help='whether to assign weights for finding wassertein distance')
parser.add_argument('-mtr', '--metric', default='sliced_wass', type=str, help='Metric to compute; Options are [sliced_wass, wass]')

args = parser.parse_args()

input_file_preds_pos = args.pos_preds_file
input_file_preds_neg = args.neg_preds_file
input_file_org = args.org_file
out_file = args.out_file
max_dim = args.max_dim
metric_to_compute = args.metric
RANGE = args.range

def remove_inf_from_pd(pd):
	if len(pd.shape)==2 and pd.shape[0]>0:
		non_inf_vals = np.where(pd[:, 1] < 1e6)[0]
		desired_ = pd[non_inf_vals]
	else:
		desired_ = pd
	return desired_

def obtain_sliced_wasserstein_dist(pd_a, pd_b, a=None, b=None):
	if len(pd_a)==0 or len(pd_b)==0:
		d_ = 0
	else:
		d_ = sliced.sliced_wasserstein_distance(pd_a, pd_b, a=a, b=b)
	return d_

D_preds_pos = scipy.sparse.load_npz(input_file_preds_pos)
D_preds_neg = scipy.sparse.load_npz(input_file_preds_neg)
D_org = scipy.sparse.load_npz(input_file_org)

non_zero_indices_pos = D_org.nonzero()
non_zero_indices_neg = D_preds_neg.nonzero()
min_pos = np.min(D_org[non_zero_indices_pos])
min_neg = np.min(D_preds_neg[non_zero_indices_neg])
if (min_pos<0 or min_neg<0) and RANGE not in ['sigmoid','abs']:
	min_ = min(min_pos,min_neg)
	D_org[non_zero_indices_pos] = D_org[non_zero_indices_pos] + 2*abs(min_)
	D_preds_pos[non_zero_indices_pos] = D_preds_pos[non_zero_indices_pos] + 2*abs(min_)
	D_preds_neg[non_zero_indices_neg] = D_preds_neg[non_zero_indices_neg] + 2*abs(min_)


if RANGE=='norm':
	non_zero_indices_pos = D_org.nonzero()
	min_pos = np.min(D_preds_pos[non_zero_indices_pos])
	max_pos = np.max(D_preds_pos[non_zero_indices_pos])
	min_neg = np.min(D_preds_neg[non_zero_indices_neg])
	max_neg = np.max(D_preds_neg[non_zero_indices_neg])

	D_preds_pos[non_zero_indices_pos] = D_preds_pos[non_zero_indices_pos] - min_pos
	D_preds_pos[non_zero_indices_pos] = D_preds_pos[non_zero_indices_pos]*2./(max_pos-min_pos) + min_pos

	D_preds_neg[non_zero_indices_neg] = D_preds_neg[non_zero_indices_neg] - min_neg
	D_preds_neg[non_zero_indices_neg] = D_preds_neg[non_zero_indices_neg]*2./(max_neg-min_neg) + min_neg
elif RANGE=='norm_v2':
	non_zero_indices_pos = D_org.nonzero()
	min_pos = np.min(D_preds_pos[non_zero_indices_pos])
	max_pos = np.max(D_preds_pos[non_zero_indices_pos])
	min_neg = np.min(D_preds_neg[non_zero_indices_neg])
	max_neg = np.max(D_preds_neg[non_zero_indices_neg])

	D_preds_pos[non_zero_indices_pos] = D_preds_pos[non_zero_indices_pos] - min_pos
	D_preds_pos[non_zero_indices_pos] = D_preds_pos[non_zero_indices_pos]*2./(max_pos-min_pos)

	D_preds_neg[non_zero_indices_neg] = D_preds_neg[non_zero_indices_neg] - min_neg
	D_preds_neg[non_zero_indices_neg] = D_preds_neg[non_zero_indices_neg]*2./(max_neg-min_neg)
elif RANGE=='norm_v3':
	epsilon = 1e-8
	non_zero_indices_pos = D_org.nonzero()
	min_pos = np.min(D_preds_pos[non_zero_indices_pos])
	max_pos = np.max(D_preds_pos[non_zero_indices_pos])
	min_neg = np.min(D_preds_neg[non_zero_indices_neg])
	max_neg = np.max(D_preds_neg[non_zero_indices_neg])

	max_ = max(max_pos,max_neg)
	min_ = min(min_pos,min_neg)

	D_preds_pos[non_zero_indices_pos] = D_preds_pos[non_zero_indices_pos] - min_
	D_preds_pos[non_zero_indices_pos] = D_preds_pos[non_zero_indices_pos]*2./(max_-min_ +epsilon)

	D_preds_neg[non_zero_indices_neg] = D_preds_neg[non_zero_indices_neg] - min_
	D_preds_neg[non_zero_indices_neg] = D_preds_neg[non_zero_indices_neg]*2./(max_-min_ +epsilon)
elif RANGE=='norm_v4':
	epsilon = 1e-8
	non_zero_indices_pos = D_org.nonzero()
	min_pos = np.min(D_preds_pos[non_zero_indices_pos])
	max_pos = np.max(D_preds_pos[non_zero_indices_pos])
	min_neg = np.min(D_preds_neg[non_zero_indices_neg])
	max_neg = np.max(D_preds_neg[non_zero_indices_neg])

	max_ = max(max_pos,max_neg)
	min_ = min(min_pos,min_neg)

	D_preds_pos[non_zero_indices_pos] = D_preds_pos[non_zero_indices_pos] - min_

	D_preds_neg[non_zero_indices_neg] = D_preds_neg[non_zero_indices_neg] - min_
elif RANGE=='abs':
	D_preds_pos = np.abs(D_preds_pos)
	D_preds_neg = np.abs(D_preds_neg)
elif RANGE=='sigmoid':
	row = D_preds_pos.tocoo().row
	col = D_preds_pos.tocoo().col
	data = D_preds_pos.tocoo().data
	data = scipy.special.expit(data)
	D_preds_pos = scipy.sparse.coo_matrix((data,(row,col))).tocsr()

	row = D_preds_neg.tocoo().row
	col = D_preds_neg.tocoo().col
	data = D_preds_neg.tocoo().data
	data = scipy.special.expit(data)
	D_preds_neg = scipy.sparse.coo_matrix((data,(row,col))).tocsr()
else:
	pass


def get_superlevel_graph(graph):
	row = graph.tocoo().row
	col = graph.tocoo().col
	data = graph.tocoo().data
	# data = scipy.special.expit(data)

	data_unique = np.unique(data)
	data_unique.sort()
	def mapper_fn(x): 
		index = np.argwhere(data_unique==x)[0][0]; 
		return data_unique[len(data_unique)-(index+1)]
	data_superlevel = []
	for d in data:
		data_superlevel.append(mapper_fn(d))

	mat_dim = max(max(row),max(col))+1
	graph_superlevel = scipy.sparse.coo_matrix((data_superlevel,(row,col)), shape=(mat_dim,mat_dim)).tocsr()
	return graph_superlevel

D_preds_pos_superlevel = get_superlevel_graph(D_preds_pos)
D_preds_neg_superlevel = get_superlevel_graph(D_preds_neg)

results_preds_pos = ripser(D_preds_pos, distance_matrix=True, maxdim=args.max_dim)
dgms_preds_pos_sublevel = [remove_inf_from_pd(dgm) for dgm in results_preds_pos['dgms']]
results_preds_pos_superlevel = ripser(D_preds_pos_superlevel, distance_matrix=True, maxdim=args.max_dim)
dgms_preds_pos_superlevel = [remove_inf_from_pd(dgm) for dgm in results_preds_pos_superlevel['dgms']]
 

results_preds_neg = ripser(D_preds_neg, distance_matrix=True, maxdim=args.max_dim)
dgms_preds_neg_sublevel = [remove_inf_from_pd(dgm) for dgm in results_preds_neg['dgms']]
results_preds_neg_superlevel = ripser(D_preds_neg_superlevel, distance_matrix=True, maxdim=args.max_dim)
dgms_preds_neg_superlevel = [remove_inf_from_pd(dgm) for dgm in results_preds_neg_superlevel['dgms']]

dgms_preds_pos = [[] for _ in range(args.max_dim+1)]
for i in range(args.max_dim+1):
	dgms_preds_pos[i].extend(dgms_preds_pos_sublevel[i])
	dgms_preds_pos[i].extend(dgms_preds_pos_superlevel[i])
dgms_preds_neg = [[] for _ in range(args.max_dim+1)]
for i in range(args.max_dim+1):
	dgms_preds_neg[i].extend(dgms_preds_neg_sublevel[i])
	dgms_preds_neg[i].extend(dgms_preds_neg_superlevel[i])

dists = []
for i in range(max_dim+1):
	# start_time = time.time()
	if args.pd_weights_swd:
		a = [d-b for b,d in dgms_preds_neg[i]]
		b = [d-b for b,d in dgms_preds_pos[i]]
	else:
		a = None
		b = None
	if metric_to_compute=='sliced_wass':
		dist = obtain_sliced_wasserstein_dist(dgms_preds_neg[i],dgms_preds_pos[i], a=a, b=b)
	elif metric_to_compute=='wass':
		dist = persim.wasserstein(dgms_preds_neg[i],dgms_preds_pos[i])
	else:
		raise NotImplementedError
	dists.append(dist)

if os.path.exists(out_file):
	with open(out_file, 'rb') as f:
		data = pickle.load(f)
else:
	data = {}
if args.save_epoch_as_key:
	epoch_num = input_file_preds_pos.split('/')[-1].split('.')[0].split('_')[-1]
	method = input_file_preds_pos.split('/')[-4] + '_' + epoch_num
else:
	method = input_file_preds_pos.split('/')[-4]
data[method] = {
	'graphs_path_pos': input_file_preds_pos,
	'graphs_path_neg': input_file_preds_neg,
	'distances': dists
}

end_time = time.time()

print('Time taken to compute distances: ',end_time-start_time)

out_folder = '/'.join(out_file.split('/')[:-1])
os.makedirs(out_folder, exist_ok=True)
with open(out_file, 'wb') as f:
	pickle.dump(data,f)
