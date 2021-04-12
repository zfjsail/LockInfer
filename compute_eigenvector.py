from os.path import join
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds


def load_edges(fpath):
    edges = []
    with open(fpath) as rf:
        for i, line in enumerate(rf):
            cur_edge = line.strip().split(",")
            cur_edge = [float(x) for x in cur_edge]
            edges.append(cur_edge)
    edges = np.array(edges)
    return edges


def write_mat(wfpath, mat):
    with open(wfpath, "w") as wf:
        for i in range(mat.shape[0]):
            cur_row = mat[i]
            for v in cur_row:
                wf.write("%.5f," % v)
            wf.write("\n")


def compute_eigenvector(graphfilename,ufilename,vfilename):
    k = 5
    B = load_edges(graphfilename)
    B = B + np.ones_like(B)
    B = B.astype(int)
    i = B[:, 0]
    j = B[:, 1]
    s = np.ones_like(i)
    m = max(i)
    n = max(j)
    sparse_mat = coo_matrix((s, (i, j)), shape=(m+1, n+1))
    sparse_mat = sparse_mat.asfptype()
    u, s, v = svds(sparse_mat, k)

    for ii in range(k):
        if sum(u[:, ii]) < 0:
            u[:, ii] = -u[:, ii]
        if sum(v[:, ii]) < 0:
            v[:, ii] = -v[:, ii]
    write_mat(ufilename, u)
    write_mat(vfilename, v)

compute_eigenvector('graph_random_powerlaw','u_random_powerlaw','v_random_powerlaw')
compute_eigenvector('graph_two_blocks','u_two_blocks','v_two_blocks')
compute_eigenvector('graph_two_blocks_camou','u_two_blocks_camou','v_two_blocks_camou')
compute_eigenvector('graph_staircase','u_staircase','v_staircase')
