"""
To be implemented for gpu.
"""
from keras.backend.common import floatx
from keras.utils.generic_utils import get_from_module
from tqdm import tqdm

from .preprocessing import *


@time_it_millis
@clip_extremes
def avg_fusion(activations):
    return np.average(activations,axis=0)

@time_it_millis
def jrer_fusion(activations):
    return np.multiply(np.exp(jr_fusion(activations)),er_fusion(activations))

@clip_extremes
def jr_fusion(activations):
    jr = np.log(np.divide(activations, np.add(1.,np.multiply(-1.,activations))))
    jr = np.sum(jr, axis=0)
    return jr

@clip_extremes
def er_fusion(activations):
    _max = np.amax(activations, axis=0)
    _min = np.amin(activations, axis=0)
    er = np.divide(_max, 1-_min)
    return er

@clip_extremes
def rank_fusion(activations,lambda_=0.001,mu=10**-6,eps_=10**-6):
    step=500
    result = np.zeros((activations[0].shape),dtype=floatx())
    for class_ in tqdm(range(activations.shape[2])):
        window=0
        res=None
        while window<activations.shape[1]:
            #initialization
            step_=min(step,activations.shape[1]-window)
            T=np.zeros((activations.shape[0],step_, step_), dtype=floatx())

            E=np.zeros((activations.shape[0],step_, step_), dtype=floatx())

            Y=np.zeros((activations.shape[0],step_, step_), dtype=floatx())
            max_m=10**10
            rho=1.1
            C=np.zeros((activations.shape[0]), dtype=floatx())
            for i in range(step_):
                for j in range(step_):
                    for k in range(T.shape[0]):
                        T[k, i, j]=np.sign(activations[k, i+window, class_]-activations[k, j+window, class_])

            #repeat

            exept=False
            repeats=0
            C_=np.ones((activations.shape[0]), dtype=floatx())
            while True:
                repeats+=1
                U,k,V = np.linalg.svd(
                    np.multiply(1/((activations.shape[0]+1)*step_),np.sum(Y,axis=0)) +
                    np.multiply(1 /(activations.shape[0]+1), np.sum(T,axis=0),) -
                    np.multiply(1/(activations.shape[0]+1),np.sum(E,axis=0)), full_matrices=True)

                CC, T_ = rank_min_body(C, E, T, U, V, Y, k, lambda_, max_m, mu, rho, step_)
                # print(C,CC)
                if repeats>2000 :raise Exception('limit hit')
                elif CC==2:
                    if np.max(C)<eps_:
                        T__=T_
                        break
                    elif np.subtract(C, C_).max()<0:
                        T__=T_
                        C_[:]=C[:]
                elif CC<2:
                    T_=T__
                    break
            s = np.multiply(1/step_, np.matmul(T_,np.ones(step_)))
            # s = np.clip(s,eps(),1-eps())
            try: res = np.append(res,s,axis=0)
            except: res = s
            window+=step
        # print(res)
        # print(activations[0, :, class_])
        result[:,class_]=res[:]

    return result


def rank_min_body(C, E, T, U, V, Y, k, lambda_, max_m, mu, rho, step_):
    k = shrinkage_operator(1 / step_, k)
    T_ = np.matmul(U, np.matmul(np.diag(k), V))
    for k in range(E.shape[0]):
        E[k] = shrinkage_operator(lambda_ / mu, T[k] + np.multiply(1 / mu, Y[k]) - T_)
    for k in range(Y.shape[0]):
        Y[k] = Y[k] + np.multiply(mu, T[k] - T_ - E[k])
    mu = min(rho * mu, max_m)
    for k in range(Y.shape[0]):
        C[k] = np.max(np.linalg.norm(
            T[k] - T_ - E[k], ord=np.inf
        ))
    CC = np.linalg.matrix_rank(T_)
    return CC, T_


def shrinkage_operator(e, k):
    k[np.where((k > -e) & (k < e))] = 0
    k[np.where(k > e)] -= e
    k[np.where(k < -e)] += e
    return k


# aliases
avg = AVG = avg_fusion
jrer = JRER = jrer_fusion

def get(identifier):
    return get_from_module(identifier, globals(), 'fusions')