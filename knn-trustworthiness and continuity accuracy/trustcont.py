import distance
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt 
def trustworthiness(orig, proj, ks):
    """Calculate a trustworthiness values for dataset.
    orig
      matrix containing the data in the original space
    proj
      matrix containing the data in the projected space
    ks range indicating neighbourhood(s) for which
      trustworthiness is calculated.
    Return list of trustworthiness values
    """

    dd_orig = distance.distance_matrix(orig)
    dd_proj = distance.distance_matrix(proj)
    nn_orig = dd_orig.argsort()
    nn_proj = dd_proj.argsort()

    ranks_orig = distance.rank_matrix(dd_orig)

    trust = []
    for k in ks:
        print(k)
        moved = []
        for i in range(orig.shape[0]):
            moved.append(moved_in(nn_orig, nn_proj, i, k))

        trust.append(trustcont_sum(moved, ranks_orig, k))

    return trust

def continuity(orig, proj, ks):
    """Calculate a continuity values for dataset
    orig
      matrix containing the data in the original space
    proj
      matrix containing the data in the projected space
    ks range indicating neighbourhood(s) for which continuity
      is calculated.
    Return a list of continuity values
    """

    dd_orig = distance.distance_matrix(orig)
    dd_proj = distance.distance_matrix(proj)
    nn_orig = dd_orig.argsort()
    nn_proj = dd_proj.argsort()

    ranks_proj = distance.rank_matrix(dd_proj)

    cont = []
    for k in ks:
        print(k)
        moved = []
        for i in range(orig.shape[0]):
            moved.append(moved_out(nn_orig, nn_proj, i, k))

        cont.append(trustcont_sum(moved, ranks_proj, k))

    return cont

def moved_out(nn_orig, nn_proj, i, k):
    """Determine points that were neighbours in the original space,
    but are not neighbours in the projection space.
    nn_orig
      neighbourhood matrix for original data
    nn_proj
      neighbourhood matrix for projection data
    i
      index of the point considered
    k
      size of the neighbourhood considered
    Return a list of indices for 'moved out' values 
    """

    oo = list(nn_orig[i, 1:k+1])
    pp = list(nn_proj[i, 1:k+1])

    for j in pp:
        if (j in pp) and (j in oo):
            oo.remove(j)

    return oo

def moved_in(nn_orig, nn_proj, i, k):
    """Determine points that are neighbours in the projection space,
    but were not neighbours in the original space.
    nn_orig
      neighbourhood matrix for original data
    nn_proj
      neighbourhood matrix for projection data
    i
      index of the point considered
    k
      size of the neighbourhood considered
    Return a list of indices for points which are 'moved in' to point i
    """

    pp = list(nn_proj[i, 1:k+1])
    oo = list(nn_orig[i, 1:k+1])

    for j in oo:
        if (j in oo) and (j in pp):
            pp.remove(j)

    return pp


def scaling_term(k, n):
    """Term that scales measure between zero and one
    k  size of the neighbourhood
    n  number of datapoints
    """
    if k < (n / 2.0):
        return 2.0 / ((n*k)*(2*n - 3*k - 1))
    else:
        return 2.0 / (n * (n - k) * (n - k - 1))


def trustcont_sum(moved, ranks, k):
    """Calculate sum used in trustworthiness or continuity calculation.
    moved
       List of lists of indices for those datapoints that have either
       moved away in (Continuity) or moved in (Trustworthiness)
       projection
    ranks
       Rank matrix of data set. For trustworthiness, ranking is in the
       original space, for continuity, ranking is in the projected
       space.
    k
       size of the neighbournood
    """

    n = ranks.shape[0]
    s = 0

    # todo: weavefy this for speed
    for i in range(n):
        for j in moved[i]:
            s = s + (ranks[i, j] - k)

    a = scaling_term(k, n)

    return 1 - a * s

if __name__ == "__main__":
    #trimap = joblib.load("trimapY.pkl")
    #print(trimap.shape)
    #trimap_data = joblib.load("trimapdata.pkl")
    #print(trimap_data.shape)
##    tsne_x = joblib.load("xtsne.pkl")
##    tsne_y = joblib.load("ytsne.pkl")
    tsne_data = joblib.load("tsnedata.pkl")
    pca = joblib.load("PCA.pkl")
##    tsne_label = joblib.load("tsnelabel.pkl")

##    tsne_x = tsne_x.values
##    tsne_y = tsne_y.values
##
##    tsne = np.concatenate((np.reshape(tsne_x,(1000,1)),np.reshape(tsne_y,(1000,1))), axis = 1)
    ks = [1,2,3,4,5,6,7,8,9]

    pca_continuity = continuity(tsne_data,pca,ks)
    joblib.dump(pca_continuity,'pca_continuity.pkl')
    pca_trustworthiness = trustworthiness(tsne_data,pca,ks)
    joblib.dump(pca_trustworthiness,'pca_trustworthiness.pkl')

    
