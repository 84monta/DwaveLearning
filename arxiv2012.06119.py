from dwave.system import EmbeddingComposite
from pyqubo import Array,Constraint,Placeholder,Sum
from pulp import LpProblem,lpSum,LpVariable
import random
import numpy as np
import copy
from numba import jit

#####2次ナップサック問題

#乱数シード定義
np.random.seed(1)
random.seed(1)

#サイズ定義
M = 12

#それぞれを選択したときの価値定義
v = []
for i in range(M):
    tmp_v = []
    for j in range(M):
        if  i <= j:
            #i=j, it value of it self
            tmp_v.append(random.randint(2,M*2))
        else:
            tmp_v.append(v[j][i])
    v.append(tmp_v)

#それぞれのウェイト
w = np.random.randint(M,M*2,M)

#MAXIM WEIGHT
WEIGHT_LIMIT = int(sum(w)*0.6/10)*10

#全件 再帰処理
def find_maximumval_by_recurse(selected_list,unselected_list,current_weight,current_val,best_list,best_val):
    last_flag = True

    for i in unselected_list:
        new_current_weight = current_weight + w[i]
        if new_current_weight < WEIGHT_LIMIT:
            last_flag = False
            new_current_val = current_val + sum(v[j][i] for j in selected_list) + v[i][i]
            new_selected_list = copy.copy(selected_list)
            new_selected_list.add(i)
            new_unselected_list = copy.copy(unselected_list)
            new_unselected_list.discard(i)
            ret_val,ret_list = find_maximumval_by_recurse(new_selected_list,new_unselected_list,new_current_weight,new_current_val,best_list,best_val)
            if ret_val > best_val:
                best_val = ret_val
                best_list = ret_list
    
    if last_flag:
        return current_val,selected_list
    else: 
        return best_val,best_list



#unselected_list = set(range(M))
#selected_list = set()
#tmp = []
ret = find_maximumval_by_recurse(set(),set(range(M)),0,0,[],0)
print(ret)

def find_maxval_by_sampling():
    x = Array.create('x',shape=(M),vartype="BINARY")

    #Objective f(x)
    H1 = Sum(0,M,lambda i: Sum(0,M,lambda j: x[i]*x[j]*v[i][j]))
