from torch_geometric.datasets import Planetoid
import torch
from torch_geometric.utils import to_dense_adj
from graph_coarsening.coarsening_utils import *
from graph_coarsening.graph_utils import zero_diag
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull
from pyunlocbox import functions, solvers
import torch.nn.functional as F

def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

def extract_components(H):

        if H.A.shape[0] != H.A.shape[1]:
            H.logger.error('Inconsistent shape to extract components. '
                           'Square matrix required.')
            return None

        if H.is_directed():
            raise NotImplementedError('Directed graphs not supported yet.')

        graphs = []
        visited = np.zeros(H.A.shape[0], dtype=bool)

        while not visited.all():
            stack = set([np.nonzero(~visited)[0][0]])
            comp = []

            while len(stack):
                v = stack.pop()
                if not visited[v]:
                    comp.append(v)
                    visited[v] = True

                    stack.update(set([idx for idx in H.A[v, :].nonzero()[1]
                                      if not visited[idx]]))

            comp = sorted(comp)
            G = H.subgraph(comp)
            G.info = {'orig_idx': comp}
            graphs.append(G)

        return graphs

def coarsening(dataset, coarsening_ratio, coarsening_method, configs=None):
    if dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=dataset)
    elif dataset == 'Physics':
        dataset = Coauthor(root='./dataset', name=dataset)
    else:
        dataset = Planetoid(root='./dataset', name=dataset)
    data = dataset[0]
    G = gsp.graphs.Graph(W=to_dense_adj(data.edge_index)[0])
    components = extract_components(G)
    print('the number of subgraphs is', len(components))
    candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
    number = 0
    C_list=[]
    Gc_list=[]
    while number < len(candidate):
        H = candidate[number]
        if len(H.info['orig_idx']) > 10:
            W0=None
            if configs.features_mode is not None:
                keep = H.info['orig_idx']
                H_features = data.x[keep]
                Wmix = make_weights(W0=torch.Tensor(H.W.todense()),configs=configs,feats=H_features)
                if not configs.on_Wmix:
                    W0 = H.W
                H = gsp.graphs.Graph(Wmix)
            C, Gc, Call, Gall = coarsen(H, r=coarsening_ratio, method=coarsening_method, W0=W0)
            C_list.append(C)
            Gc_list.append(Gc)
        number += 1
    return data.x.shape[1], len(set(np.array(data.y))), candidate, C_list, Gc_list

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def splits(data, num_classes, exp):
    if exp!='fixed':
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if exp == 'random':
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        else:
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data


def load_data(dataset, candidate, C_list, Gc_list, exp):
    if dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=dataset)
    elif dataset == 'Physics':
        dataset = Coauthor(root='./dataset', name=dataset)
    else:
        dataset = Planetoid(root='./dataset', name=dataset)
    n_classes = len(set(np.array(dataset[0].y)))
    data = splits(dataset[0], n_classes, exp)
    train_mask = data.train_mask
    val_mask = data.val_mask
    labels = data.y
    features = data.x

    coarsen_node = 0
    number = 0
    coarsen_row = None
    coarsen_col = None
    coarsen_features = torch.Tensor([])
    coarsen_train_labels = torch.Tensor([])
    coarsen_train_mask = torch.Tensor([]).bool()
    coarsen_val_labels = torch.Tensor([])
    coarsen_val_mask = torch.Tensor([]).bool()

    while number < len(candidate):
        H = candidate[number]
        keep = H.info['orig_idx']
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]
        if len(H.info['orig_idx']) > 10 and torch.sum(H_train_mask)+torch.sum(H_val_mask) > 0:
            train_labels = one_hot(H_labels, n_classes)
            train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
            val_labels = one_hot(H_labels, n_classes)
            val_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])

            C = C_list[number]
            Gc = Gc_list[number]

            new_train_mask = torch.BoolTensor(np.sum(C.dot(train_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(train_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_train_mask[mix_mask > 1] = False

            new_val_mask = torch.BoolTensor(np.sum(C.dot(val_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(val_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_val_mask[mix_mask > 1] = False

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, new_val_mask], dim=0)

            if coarsen_row is None:
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += Gc.W.shape[0]

        elif torch.sum(H_train_mask)+torch.sum(H_val_mask)>0:

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, H_labels.float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, H_labels.float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, H_val_mask], dim=0)

            if coarsen_row is None:
                raise Exception('The graph does not need coarsening.')
            else:
                current_row = H.W.tocoo().row + coarsen_node
                current_col = H.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += H.W.shape[0]
        number += 1

    print('the size of coarsen graph features:', coarsen_features.shape)

    coarsen_edge = torch.LongTensor([coarsen_row, coarsen_col])
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()

    return data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge

############################################################## Feature Learning SITARAM #########################################################################
# %%
def make_weights(W0, configs, feats, norm_weights=False):
    prnt_feats_report(X=feats)
    if configs.features_mode == "RBF":
        W_feats = calc_rbf(X=feats,sigma=1.0,norm_ord=1)
    elif configs.features_mode == "CosSim":
        W_feats = calc_cos_sim(X=feats)
    elif configs.features_mode == "Smooth_Sigz":
        W_feats = learn_frm_ss(W0=W0.numpy(),X=feats.numpy(),configs=configs)
    else:
        raise NotImplementedError("Feature Learning Method Not Implemented Yet :(")
    # print(W0,W_feats,end="\n\n")
    W_mix = combine_weights(W1 = W0, W2 = W_feats, alpha = configs.mix_alpha, thresh = configs.mix_threshold)
    if norm_weights:
        W_mix = F.normalize(W_mix, p=1)
    # print(W_mix)
    # print("")
    prnt_report(w0=W0,wm=W_mix)
    return W_mix

# %%
def calc_cos_sim(X):
    N = X.shape[0]
    x_norm = X.norm(dim=1)[:, None]
    x_norm = X / torch.where(x_norm==0,torch.Tensor([1]),x_norm)
    res = torch.mm(x_norm, x_norm.transpose(0,1))
    res = (1+res) - (2*torch.eye(N))
    # print((torch.isnan(x_norm)).sum().item(),(torch.isnan(X)).sum().item(),(torch.isnan(res)).sum().item())
    return res

# %%
def calc_rbf(X,sigma,norm_ord):
    if norm_ord is not None:
        X = F.normalize(X, p=norm_ord, dim=-1)
    return torch.exp(-torch.cdist(X,X)**2/(2*sigma*sigma))-torch.eye(X.shape[0])

# %%
def combine_weights(W1, W2, alpha, thresh, mode="Weights", norm_before_combining=True):
    print("")
    # print(W1,W2)
    # print(W2.max().item())
    # print(abs(W2.max().item()))
    # print(W2,W2/abs(W2.max().item()))
    if norm_before_combining:
        W_mix = (alpha*(W1/abs(W1.max().item()))) + ((1-alpha)*(W2/abs(W2.max().item())))
    else:
        W_mix = (alpha*(W1)) + ((1-alpha)*W2)
    W_mix = zero_diag(W_mix)
    W_mix = (W_mix+W_mix.permute(1,0))/2
    # print(W_mix,end="\n\n")
    if mode == "Weights":
        return torch.where(W_mix>=thresh,W_mix,torch.Tensor([0]))
    elif mode == "Adjacency":
        return torch.where(W_mix>=thresh,torch.Tensor([1]),torch.Tensor([0]))
    else:
        raise NotImplementedError("Weight Matrices Convex Combination Thresholding Mode not Available SitaRam _/\_")

# %%
def prnt_report(w0,wm):
    pos_dect=False
    init_edges = (w0==1).sum().item()
    if init_edges==0:
        init_edges = (w0>0).sum().item()
        pos_dect=True
    init_nulls = (w0.shape[0]**2)-init_edges
    final_edges = (wm>0).sum().item()
    if pos_dect:
        final_undropped_edges = (wm[torch.where(w0>0)]>0).sum().item()
    else:
        final_undropped_edges = (wm[torch.where(w0==1)]>0).sum().item()
    final_created_edges = final_edges - final_undropped_edges
    assert(final_created_edges>=0)
    # print(pos_dect)
    print("Edges Undropped SitaRam _/\_ (in %): "+str((final_undropped_edges/init_edges)*100))
    print("Edges Created SitaRam _/\_ (in %): "+str((final_created_edges/init_nulls)*100))
    print("Number of edges initially SitaRam _/\_: "+str(init_edges))
    print("Number of edges finally SitaRam _/\_: "+str(final_edges))
    print("")

# %%
def learn_frm_ss(X,W0,configs):
    # print(W0.shape,X.shape)
    w0=vectorize_sym(A=W0)
    x_norm = np.linalg.norm(X,ord=2,axis=-1,keepdims=True)
    feats = X/np.where(x_norm==0,1,x_norm)
    return torch.Tensor(log_degree_barrier(X=X, dist_type='sqeuclidean', alpha=configs.alpha_ss, beta=configs.beta_ss, w0=w0, maxit=configs.maxiter_ss, verbosity='ALL'))
    
# %%
def vectorize_sym(A,to_assert_sr=True,expand=False):
    if to_assert_sr:
        assert np.sum(A!=A.T)==0, "Matrix to be vectorized not Symmetric SITARAM _/\_"
    m = A.shape[0]
    length = (m*(m-1))//2
    vec = np.zeros(length)
    start_idx=0
    for i,row in enumerate(A):
        vec_len = m-1-i
        vec[start_idx:start_idx+vec_len] = row[i+1:]
        start_idx += vec_len
    if expand:
        return np.expand_dims(vec,axis=1)
    return vec

# %%
def log_degree_barrier(X, dist_type='sqeuclidean', alpha=1, beta=1, step=0.5,
                       w0=None, maxit=1000, rtol=1e-5, retall=False,
                       verbosity='NONE'):
    r"""
    Learn graph by imposing a log barrier on the degrees
    This is done by solving
    :math:`\tilde{W} = \underset{W \in \mathcal{W}_m}{\text{arg}\min} \,
    \|W \odot Z\|_{1,1} - \alpha 1^{T} \log{W1} + \beta \| W \|_{F}^{2}`,
    where :math:`Z` is a pairwise distance matrix, and :math:`\mathcal{W}_m`
    is the set of valid symmetric weighted adjacency matrices.
    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of N variable observations in an M-dimensional
        space. The learned graph will have N nodes.
    dist_type : string
        Type of pairwise distance between variables. See
        :func:`spatial.distance.pdist` for the possible options.
    alpha : float, optional
        Regularization parameter acting on the log barrier
    beta : float, optional
        Regularization parameter controlling the density of the graph
    step : float, optional
        A number between 0 and 1 defining a stepsize value in the admissible
        stepsize interval (see [Komodakis & Pesquet, 2015], Algorithm 6)
    w0 : array_like, optional
        Initialization of the edge weights. Must be an N(N-1)/2-dimensional
        vector.
    maxit : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Stopping criterion. Relative tolerance between successive updates.
    retall : boolean
        Return solution and problem details. See output of
        :func:`pyunlocbox.solvers.solve`.
    verbosity : {'NONE', 'LOW', 'HIGH', 'ALL'}, optional
        Level of verbosity of the solver. See :func:`pyunlocbox.solvers.solve`.
    Returns
    -------
    W : array_like
        Learned weighted adjacency matrix
    problem : dict, optional
        Information about the solution of the optimization. Only returned if
        retall == True.
    Notes
    -----
    This is the solver proposed in [Kalofolias, 2016] :cite:`kalofolias2016`.
    Examples
    --------
    """

    # Parse X
    N = X.shape[0]
    z = sp.spatial.distance.pdist(X, dist_type)  # Pairwise distances

    # Parse stepsize
    if (step <= 0) or (step > 1):
        raise ValueError("step must be a number between 0 and 1.")

    # Parse initial weights
    w0 = np.zeros(z.shape) if w0 is None else w0
    if (w0.shape != z.shape):
        raise ValueError("w0 must be of dimension N(N-1)/2.")

    # Get primal-dual linear map
    K_sr, Kt_sr = weight2degmap(N,array=True)
    K = lambda w: K_sr.dot(w)
    Kt = lambda d: Kt_sr.dot(d)
    norm_K = np.sqrt(2 * (N - 1))

    # Assemble functions in the objective
    f1 = functions.func()
    f1._eval = lambda w: 2 * np.dot(w, z)
    f1._prox = lambda w, gamma: np.maximum(0, w - (2 * gamma * z))

    f2 = functions.func()
    f2._eval = lambda w: - alpha * np.sum(np.log(np.maximum(
        np.finfo(np.float64).eps, K(w))))
    f2._prox = lambda d, gamma: np.maximum(
        0, 0.5 * (d + np.sqrt(d**2 + (4 * alpha * gamma))))

    f3 = functions.func()
    f3._eval = lambda w: beta * np.sum(w**2)
    f3._grad = lambda w: 2 * beta * w
    lipg = 2 * beta

    # Rescale stepsize
    stepsize = step / (1 + lipg + norm_K)

    # Solve problem
    solver = solvers.mlfbf(L=K, Lt=Kt, step=stepsize)
    problem = solvers.solve([f1, f2, f3], x0=w0, solver=solver, maxit=maxit,
                            rtol=rtol, verbosity=verbosity)

    # Transform weight matrix from vector form to matrix form
    W = sp.spatial.distance.squareform(problem['sol'])

    if retall:
        return W, problem
    else:
        return W

# %%
def weight2degmap(N, array=False):
    r"""
    Generate linear operator K such that W @ 1 = K @ vec(W).
    Parameters
    ----------
    N : int
        Number of nodes on the graph
    Returns
    -------
    K : function
        Operator such that K(w) is the vector of node degrees
    Kt : function
        Adjoint operator mapping from degree space to edge weight space
    array : boolean, optional
        Indicates if the maps are returned as array (True) or callable (False).
    Examples
    --------
    Notes
    -----
    Used in :func:`learn_graph.log_degree_barrier method`.
    """
    Ne = int(N * (N - 1) / 2)  # Number of edges
    row_idx1 = np.zeros((Ne, ))
    row_idx2 = np.zeros((Ne, ))
    count = 0
    for i in np.arange(1, N):
        row_idx1[count: (count + (N - i))] = i - 1
        row_idx2[count: (count + (N - i))] = np.arange(i, N)
        count = count + N - i
    row_idx = np.concatenate((row_idx1, row_idx2))
    col_idx = np.concatenate((np.arange(0, Ne), np.arange(0, Ne)))
    vals = np.ones(len(row_idx))
    K = sp.sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(N, Ne))
    if array:
        return K, K.transpose()
    else:
        return lambda w: K.dot(w), lambda d: K.transpose().dot(d)

# %%
def prnt_feats_report(X):
    zeros = (X==0).sum().item()
    print("Sparsity in given component's feature matrix % SITARAM _/\_ => "+str(100*(zeros/(X.shape[0]*X.shape[1]))),end="\n\n")
