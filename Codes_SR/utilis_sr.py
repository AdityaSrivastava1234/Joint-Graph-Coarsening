import pygsp as gsp
import torch
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ******************************* For making plots SitaRam ***********************
# %%
def plot_train_val_curve(train_loss_list,test_loss_list,no_of_epochs,nm,path,is_percent=True,mark_fifty=False):
    if test_loss_list!=[]:
        plt.plot(range(no_of_epochs),train_loss_list,label = 'Train '+nm)
        if len(test_loss_list)==2:
            plt.plot(range(no_of_epochs),test_loss_list[0],label = 'Val '+nm)
            plt.plot(range(no_of_epochs),test_loss_list[1],label = 'Test '+nm)
        else:
            plt.plot(range(no_of_epochs),test_loss_list,label = 'Val '+nm)
        plt.title('Train-Val '+nm+' Curve')
        plt.legend()
    else:
        plt.plot(range(no_of_epochs),train_loss_list)
        plt.title(nm+' Curve')
    plt.grid()
    if is_percent:
        plt.axis([-0.7,no_of_epochs+1,-0.5,100+0.5])
        if mark_fifty or 'AUC_Score' in nm:
            plt.hlines(y = 50, xmin = -0.7, xmax = no_of_epochs+1, linestyles='dotted' ,colors='r')
    plt.xlabel('No. of epochs --->')
    plt.ylabel(nm+' ---> ')
    plt.savefig(path+'//'+nm+'_SR.png')
    plt.clf()

# ************************** For Fetching Optimizer(s) ***************************
# %%
def make_opt_sr(model,lr,wd,opt_name='Adam_SR',b1=0.9,b2=0.999,params_lst=None):
    if params_lst!=None:
        trainable_params=params_lst
    else:
        trainable_params=model.parameters()
    if opt_name=='Adam_SR':
        optimizer = torch.optim.Adam(trainable_params, 
                    lr=lr,
                    weight_decay=wd,
                    betas=(b1,b2)
                    )
    elif opt_name=='RMSprop_SR':
        optimizer = torch.optim.RMSprop(trainable_params, 
                    lr=lr,
                    weight_decay=wd,
                    )
    elif opt_name=='Adagrad_SR':
        optimizer = torch.optim.Adagrad(trainable_params, 
                    lr=lr,
                    weight_decay=wd,
                    )
    return optimizer

# ************************** Main Coercening Functions ***************************
# %%
def coarsening(configs):
    ds_nm_sr = configs['dataset'].split('_')[0]
    if configs['dataset'] == 'dblp_SR':
        dataset = CitationFull(root=configs['data_path'][:-1], name=ds_nm_sr)
    elif configs['dataset'] == 'Physics_SR':
        dataset = Coauthor(root=configs['data_path']+configs['dataset'], name=ds_nm_sr)
    else:
        dataset = Planetoid(root=configs['data_path'][:-1], name=ds_nm_sr)
    data = dataset[0]
    G = gsp.graphs.Graph(W=to_dense_adj(data.edge_index)[0])
    components = extract_components(G)
    print('Number of Subgraphs present = ', len(components))
    print("")
    candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
    number = 0
    C_list=[]
    Gc_list=[]
    while number < len(candidate):
        H = candidate[number]
        if len(H.info['orig_idx']) > 10:
            C, Gc, Call, Gall = coarsen(G=H, r=configs['coarsening_ratio'], method=configs['coarsening_method'], K=configs['K'])
            C_list.append(C)
            Gc_list.append(Gc)
        number += 1
    return data.x.shape[1], len(set(np.array(data.y))), candidate, C_list, Gc_list

# %%
def coarsen(
    G,
    K=10,
    r=0.5,
    max_levels=10,
    method="variation_neighborhood",
    algorithm="greedy",
    Uk=None,
    lk=None,
    max_level_r=0.99,
):
    """
    This function provides a common interface for coarsening algorithms that contract subgraphs

    Parameters
    ----------
    G : pygsp Graph
    K : int
        The size of the subspace we are interested in preserving.
    r : float between (0,1)
        The desired reduction defined as 1 - n/N.
    method : String
        ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron'] 
    
    Returns
    -------
    C : np.array of size n x N
        The coarsening matrix.
    Gc : pygsp Graph
        The smaller graph.
    Call : list of np.arrays
        Coarsening matrices for each level
    Gall : list of (n_levels+1) pygsp Graphs
        All graphs involved in the multilevel coarsening

    Example
    -------
    C, Gc, Call, Gall = coarsen(G, K=10, r=0.8)
    """
    r = np.clip(r, 0, 0.999)
    G0 = G
    N = G.N

    # current and target graph sizes
    n, n_target = N, np.ceil((1 - r) * N)

    C = sp.sparse.eye(N, format="csc")
    Gc = G

    Call, Gall = [], []
    Gall.append(G)

    for level in range(1, max_levels + 1):

        G = Gc

        # how much more we need to reduce the current graph
        r_cur = np.clip(1 - n_target / n, 0.0, max_level_r)

        if "variation" in method:

            if level == 1:
                if (Uk is not None) and (lk is not None) and (len(lk) >= K):
                    mask = lk < 1e-10
                    lk[mask] = 1
                    lsinv = lk ** (-0.5)
                    lsinv[mask] = 0
                    B = Uk[:, :K] @ np.diag(lsinv[:K])
                else:
                    offset = 2 * max(G.dw)
                    T = offset * sp.sparse.eye(G.N, format="csc") - G.L
                    lk, Uk = sp.sparse.linalg.eigsh(T, k=K, which="LM", tol=1e-5)
                    lk = (offset - lk)[::-1]
                    Uk = Uk[:, ::-1]
                    mask = lk < 1e-10
                    lk[mask] = 1
                    lsinv = lk ** (-0.5)
                    lsinv[mask] = 0
                    B = Uk @ np.diag(lsinv)
                A = B
            else:
                B = iC.dot(B)
                d, V = np.linalg.eig(B.T @ (G.L).dot(B))
                mask = d == 0
                d[mask] = 1
                dinvsqrt = d ** (-1 / 2)
                dinvsqrt[mask] = 0
                A = B @ np.diag(dinvsqrt) @ V

            if method == "variation_edges":
                coarsening_list = contract_variation_edges(
                    G, K=K, A=A, r=r_cur, algorithm=algorithm
                )
            else:
                # coarsening_list = contract_variation_linear(
                #     G, K=K, A=A, r=r_cur, mode=method
                # )
                print("Not implemented yet! :(")

        else:
            # weights = get_proximity_measure(G, method, K=K)

            # if algorithm == "optimal":
            #     # the edge-weight should be light at proximal edges
            #     weights = -weights
            #     if "rss" not in method:
            #         weights -= min(weights)
            #     coarsening_list = matching_optimal(G, weights=weights, r=r_cur)

            # elif algorithm == "greedy":
            #     coarsening_list = matching_greedy(G, weights=weights, r=r_cur)
            print("Not implemented yet! :(")

        iC = get_coarsening_matrix(G, coarsening_list)

        if iC.shape[1] - iC.shape[0] <= 2:
            break  # avoid too many levels for so few nodes

        C = iC.dot(C)
        Call.append(iC)

        Wc = zero_diag(coarsen_matrix(G.W, iC))  # coarsen and remove self-loops
        Wc = (Wc + Wc.T) / 2  # this is only needed to avoid pygsp complaining for tiny errors

        if not hasattr(G, "coords"):
            Gc = gsp.graphs.Graph(Wc)
        else:
            Gc = gsp.graphs.Graph(Wc, coords=coarsen_vector(G.coords, iC))
        Gall.append(Gc)

        n = Gc.N

        if n <= n_target:
            break

    return C, Gc, Call, Gall

# ****************************************** Coarcening Related Utilities *******************************************
# %%
def matching_greedy(G, weights, r=0.4):
    """
    Generates a matching greedily by selecting at each iteration the edge
    with the largest weight and then removing all adjacent edges from the
    candidate set.

    Parameters
    ----------
    G : pygsp graph
    weights : np.array(M)
        a weight for each edge
    r : float
        The desired dimensionality reduction (r = 1 - n/N)

    Notes:
    * The complexity of this is O(M)
    * Depending on G, the algorithm might fail to return ratios>0.3
    """

    N = G.N

    # the edge set
    edges = np.array(G.get_edge_list()[0:2])
    M = edges.shape[1]

    idx = np.argsort(-weights)
    # idx = np.argsort(weights)[::-1]
    edges = edges[:, idx]

    # the candidate edge set
    candidate_edges = edges.T.tolist()

    # the matching edge set (this is a list of arrays)
    matching = []

    # which vertices have been selected
    marked = np.zeros(N, dtype=np.bool)

    n, n_target = N, (1 - r) * N
    while len(candidate_edges) > 0:

        # pop a candidate edge
        [i, j] = candidate_edges.pop(0)

        # check if marked
        if any(marked[[i, j]]):
            continue

        marked[[i, j]] = True
        n -= 1

        # add it to the matching
        matching.append(np.array([i, j]))

        # termination condition
        if n <= n_target:
            break

    return np.array(matching)

# %%
def contract_variation_edges(G, A=None, K=10, r=0.5, algorithm="greedy"):
    """
    Sequential contraction with local variation and edge-based families.
    This is a specialized implementation for the edge-based family, that works
    slightly faster than the contract_variation() function, which works for
    any family.

    See contract_variation() for documentation.
    """
    N, deg, M = G.N, G.dw, G.Ne
    ones = np.ones(2)
    Pibot = np.eye(2) - np.outer(ones, ones) / 2

    # cost function for the edge
    def subgraph_cost(G, A, edge):
        edge, w = edge[:2].astype(np.int), edge[2]
        deg_new = 2 * deg[edge] - w
        L = np.array([[deg_new[0], -w], [-w, deg_new[1]]])
        B = Pibot @ A[edge, :]
        return np.linalg.norm(B.T @ L @ B)

    # cost function for the edge
    def subgraph_cost_old(G, A, edge):
        w = G.W[edge[0], edge[1]]
        deg_new = 2 * deg[edge] - w
        L = np.array([[deg_new[0], -w], [-w, deg_new[1]]])
        B = Pibot @ A[edge, :]
        return np.linalg.norm(B.T @ L @ B)

    # edges = np.array(G.get_edge_list()[0:2])
    edges = np.array(G.get_edge_list())
    weights = np.array([subgraph_cost(G, A, edges[:, e]) for e in range(M)])
    # weights = np.zeros(M)
    # for e in range(M):
    #    weights[e] = subgraph_cost_old(G, A, edges[:,e])

    if algorithm == "optimal":
        # identify the minimum weight matching
        # coarsening_list = matching_optimal(G, weights=weights, r=r)
        print("Not implemented yet! :(")

    elif algorithm == "greedy":
        # find a heavy weight matching
        coarsening_list = matching_greedy(G, weights=-weights, r=r)

    return coarsening_list

# %%
def get_coarsening_matrix(G, partitioning):
    """
    This function should be called in order to build the coarsening matrix C.

    Parameters
    ----------
    G : the graph to be coarsened
    partitioning : a list of subgraphs to be contracted

    Returns
    -------
    C : the new coarsening matrix

    Example
    -------
    C = contract(gsp.graphs.sensor(20),[0,1]) ??
    """

    # C = np.eye(G.N)
    C = sp.sparse.eye(G.N, format="lil")

    rows_to_delete = []
    for subgraph in partitioning:

        nc = len(subgraph)

        # add v_j's to v_i's row
        C[subgraph[0], subgraph] = 1 / np.sqrt(nc)  # np.ones((1,nc))/np.sqrt(nc)

        rows_to_delete.extend(subgraph[1:])

    # delete vertices
    # C = np.delete(C,rows_to_delete,0)

    C.rows = np.delete(C.rows, rows_to_delete)
    C.data = np.delete(C.data, rows_to_delete)
    C._shape = (G.N - len(rows_to_delete), G.N)

    C = sp.sparse.csc_matrix(C)

    # check that this is a projection matrix
    # assert sp.sparse.linalg.norm( ((C.T).dot(C))**2 - ((C.T).dot(C)) , ord='fro') < 1e-5

    return C

# %%
def coarsen_matrix(W, C):
    # Pinv = C.T; #Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return (Pinv.T).dot(W.dot(Pinv))

# %%
def coarsen_vector(x, C):
    return (C.power(2)).dot(x)

# %%
def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

# ********************************* Graph Processing utilities ***************************
# %%
def zero_diag(A):
    if sp.sparse.issparse(A):
        return A - sp.sparse.dia_matrix((A.diagonal()[sp.newaxis, :], [0]), shape=(A.shape[0], A.shape[1]))
    else:
        D = A.diagonal()
        return A - np.diag(D)

# %%
def extract_components(H):

    if H.A.shape[0] != H.A.shape[1]:
        H.logger.error('Inconsistent shape to extract components. ''Square matrix required.')
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