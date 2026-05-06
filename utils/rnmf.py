import numpy as np
from time import time
import scipy.io as sio
np.set_printoptions(precision=4,suppress=True,linewidth=1000)

def factorize(S,K,Max_Iter=500000,check_point=100,thresh=5e-5,alpha=1):
    N = S.shape[0]

    # init A
    A = np.random.rand(N, K)
    # init I1, I2
    I1 = np.ones((K,1))
    I2 = np.ones((N,1))

    # optimize ||S-AA'||^2+alpha*||AI1-I2||^2
    check1 = np.linalg.norm(A @ A.T, 2)
    check2 = np.linalg.norm(S - A @ A.T, 2)
    exit_flag = 0
    for iter in range(1,Max_Iter):
        top = 4 * S @ A + 2*alpha * I2 @ I1.T + 1e-3
        bot = 4 * A @ A.T @ A + 2 * alpha * A @ I1 @ I1.T + 1e-3
        A *= np.sqrt(top/bot)
        if iter % check_point == 0:
            if alpha>=3:
                alpha = alpha*(1.0/3)
            else:
                alpha = 1
            now_check1 = np.linalg.norm(A @ A.T, 2)
            now_check2 = np.linalg.norm(S - A @ A.T, 2)
            if abs(now_check1 - check1) <= thresh and abs(now_check2 - check2) <= thresh:
                exit_flag = 1
                break
            else:
                check1 = now_check1
                check2 = now_check2
    return A, iter, exit_flag

def assign_H(S,view_list,thresh=0.9):
    if sum(view_list)>0:
        [eig_value,_] = np.linalg.eig(S)
        K = np.sum(eig_value>=thresh)
        K = max(K,max(view_list))   # K should be at least the maximum number of views
        H = assign_kH(S,view_list,K)
        return H,K
    else:
        return np.zeros([0,0]),0


def assign_kH(S,view_list,K,sparsity_thresh=0.6):
    N = S.shape[0]
    A, iter, exit_flag = factorize(S, K)
    print('Iteration is : {}, exit flag is : {}'.format(iter, exit_flag))

    # assign H row-wise
    H = np.zeros([N,K])
    max_indices = np.argmax(A, axis=1)
    H[np.arange(H.shape[0]), max_indices] = 1

    return H

if __name__ == '__main__':
    S14 =  np.asarray([[1, 0 ,0 ,0.7, 0.4, 0.3, 0.8, 0.4 ,0.5 ,0.3, 0.6 ,0.5, 0.4 ,0.2],
       [0 ,1 ,0, 0.3, 0.4, 0.6, 0.2, 0.5, 0.3, 0.6, 0.1, 0.2, 0.4, 0.],
       [0, 0 ,1 ,0.4, 0.6, 0.3, 0.2 ,0.4, 0.1 ,0.3, 0.4, 0.1, 0.3, 0.3],
       [0.7, 0.2, 0.3, 1 ,0 ,0, 0.8, 0.3, 0.2, 0.4, 0.6, 0.1, 0.2, 0.5],
       [0.3, 0.2, 0.4, 0 ,1, 0, 0.4, 0.8, 0.3, 0.1, 0.2, 0.7, 0.3, 0.5],
       [0.3, 0.6, 0.1, 0, 0, 1, 0.3, 0.5, 0.4, 0.2, 0.3, 0.1, 0.4, 0.8],
       [0.6, 0.3, 0.2, 0.7, 0.3, 0.1, 1, 0, 0, 0, 0.6, 0.3, 0.2, 0.4],
       [0.2, 0.2, 0.4, 0.1, 0.7, 0.3, 0 ,1, 0, 0, 0.4, 0.8, 0.6, 0.4],
       [0.1, 0.4 ,0.3, 0.5, 0.6, 0.4, 0, 0, 1, 0, 0.4, 0.5, 0.8, 0.3],
       [0.3, 0.2 ,0.5, 0.4 ,0.3, 0.4, 0, 0, 0, 1, 0.4, 0.3, 0.1, 0.2],
       [0.6, 0.3, 0.1, 0.6, 0.4, 0.3, 0.8, 0.1, 0.5, 0.3 ,1, 0 ,0, 0],
       [0.3, 0.2, 0.1, 0.3, 0.7, 0.4, 0.6, 0.3, 0.2, 0.1, 0, 1, 0 ,0],
       [0.3, 0.1, 0.4, 0.3, 0.2, 0.3 ,0.6, 0.3, 0.8, 0.3, 0 ,0 ,1 ,0],
       [0.3, 0.7, 0.3, 0.5, 0.3 ,0.9, 0.2, 0.3, 0.1 ,0.6, 0 ,0, 0, 1]])
       
    #S14 = np.asarray([[1,0.5,0.1,0.8,0.4],
    #			[0.3,1,0.2,0.3,0.9],
    #			[0.1,0.2,1,0.1,0.2],
    #			[0.8,0.3,0.1,1,0.2],
    #			[0.4,0.9,0.2,0.2,1]])
    			
    #S14 = np.array([[1,0.1],[0.1,1]])
    S = (S14+S14.T)/2
    # sigma = 20
    # data = sio.loadmat('1741.mat')
    # D = data['array']
    # D  = D[0:18,0:18]
    # S = init_S(D,view_list)

    view_list = [3,3,4,4]
    #view_list = [2]
    start = time()
    # S = init_S(D, view_list)
    # H = assign_kH(S,view_list,k)
    # A = factorize(S,view_list,6,2)

    H = assign_H(S,view_list)
    end = time()
    print(end-start)
    k = H[0].shape[1]
    print (H[0])
    for j in range(k):
        print (np.where(H[0][:,j]==1)[0]+1)
