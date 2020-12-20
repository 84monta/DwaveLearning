# Implementation of https://arxiv.org/pdf/2012.06119.pdf
from dwave.system import EmbeddingComposite,DWaveSampler
from pyqubo import Array,Constraint,Placeholder
from pulp import LpProblem,lpSum,LpVariable
from itertools import product
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
        if new_current_weight <= WEIGHT_LIMIT:
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
#ret = find_maximumval_by_recurse(set(),set(range(M)),0,0,[],0)
#print(ret)

def find_maxval_by_sampling(sweep=100):
    x = Array.create('x',shape=(M),vartype="BINARY")
    l = Placeholder("lambda")
    z = Placeholder("z")
    r = Placeholder("rho")


    #Objective f(x)
    H1 = 150-sum(x[i]*x[j]*v[i][j] for i,j in product(range(M),range(M)))

    H2 = sum(w[i]*x[i] for i in range(M)) - WEIGHT_LIMIT  - z
    H3 = (sum(w[i]*x[i] for i in range(M)) - WEIGHT_LIMIT  - z)**2

    H = H1 + l*H2 + r/2.0*H3

    model = H.compile()

    val_lambda = 0
    val_z = 0
    val_rho = 0.3
    gamma = 0.5

    for i in range(sweep):
        placeholder_vals = {"lambda":val_lambda,"z":val_z,"rho":val_rho}
        Q,offset = model.to_qubo(feed_dict = placeholder_vals)

        sampler = EmbeddingComposite(DWaveSampler(solver="DW_2000Q_6"))
        responses = sampler.sample_qubo(Q,num_reads=5000)

        solutions = model.decode_sampleset(responses,feed_dict = placeholder_vals)
        x_feas = 1000
        x_cost = 1000

        #sampling and compute x_feas and x_cost
        for sol in solutions:
            energy = model.energy(sample=sol.sample,vartype="BINARY",feed_dict = placeholder_vals)
            sol_cost = energy + gamma*(1 if val_z > 0 else 0)
            if sol_cost < x_cost:
                x_cost = sol_cost
                x_cost_sample = sol.sample
            sol_weight = sum(w[i]*sol.sample[f"x[{i}]"] for i in range(M))
            if sol_cost < x_feas and sol_weight <= WEIGHT_LIMIT:
                x_feas = sol_cost
                x_feas_sample = sol.sample
        
        Gx_cost = sum(w[i]*x_cost_sample[f"x[{i}]"] for i in range(M))
        val_z = min(0,Gx_cost - WEIGHT_LIMIT)
        val_lambda = val_lambda + val_rho*(Gx_cost - WEIGHT_LIMIT - val_z)

        print(f"val_z: {val_z}, val_lambda: {val_lambda}, Gx_cost: {Gx_cost}")

    
    print(x_feas_sample)







find_maxval_by_sampling(10)
