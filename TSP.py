from dwave.system import EmbeddingComposite,DWaveSampler
from dwave.system.composites.embedding import FixedEmbeddingComposite
from dwave.system.samplers import dwave_sampler
from pyqubo import Array,Sum,Constraint,Placeholder
import random
from matplotlib import pyplot as plt 
import networkx as nx
from math import sqrt
from itertools import product,combinations


class TSP:

    def __init__(self,size=10,seed=1):
        self.size = size
        random.seed(seed)

        self.cities = {}
        for i in range(size):
            self.cities[i] = (random.uniform(0.0,1.0),random.uniform(0.0,1.0))
        
        self.define_model()
        self.embedding = None
    
    def dist(self,i,j):
        return sqrt((self.cities[i][0]-self.cities[j][0])**2 + (self.cities[i][1]-self.cities[j][1])**2) 
    
    def define_model(self):
        city_num = self.size

        #変数定義
        x = Array.create('x',shape=(city_num,city_num),vartype="BINARY")

        #距離のペナルティ
        H1 = 0
        for i,j in product(range(city_num),range(city_num)):
            for k in range(city_num):
                H1 += self.dist(j,k)*x[i][j]*x[(i+1)%city_num][k]

        #ある時刻には訪れる都市は1つだけ
        H2 = 0
        for i in range(city_num):
            H2 += Constraint((Sum(0,city_num,lambda j:x[i][j])-1)**2, label = f"Constraint one hot for time{i}")

        #ある都市に訪れるのは1度だけ
        H3 = 0
        for i in range(city_num):
            H3 += Constraint((Sum(0,city_num,lambda j:x[j][i])-1)**2, label = f"Constraint one hot for city {i}")
        
        H = Placeholder("H1")*H1 + Placeholder("H2")*(H2+H3)

        #モデル作成
        self.model = H.compile()
    
    def solve_QPU(self,h1=1.0,h2=1.0,num=100,sol="DW_2000Q_6",emb_param={"verbose":2,"max_no_improvement":6000,"timeout":600,"chainlength_patience":1000,"threads":15}):
        sampler = EmbeddingComposite(DWaveSampler(solver=sol))
        Q,offset = self.model.to_qubo(feed_dict={"H1":h1,"H2":h2})
        self.responses = sampler.sample_qubo(Q,num_reads=num,chain_strength=5.0,embedding_parameters=emb_param,postprocess="optimization")

        self.solutions = self.model.decode_dimod_response(self.responses,feed_dict={"H1":h1,"H2":h2})
        
        self.best_dist = 100000
        self.best_route = []

        for sol in self.solutions:
            if len(sol[1]) == 0:
                tmp_sol = [ sum(i*val for i,val in sol[0]['x'][j].items()) for j in range(self.size)]
                if self.route_len(tmp_sol) < self.best_dist:
                    self.best_dist = self.route_len(tmp_sol)
                    self.best_route = tmp_sol

    def solve_QPU_Fix(self,h1=1.0,h2=1.0,num=100,sol="DW_2000Q_6",emb_param={}):
        from minorminer import find_embedding

        dsampler = DWaveSampler(solver=sol)
        #sampler = EmbeddingComposite(DWaveSampler(solver=sol))
        bqm = self.model.to_dimod_bqm(feed_dict={"H1":h1,"H2":h2})

        if self.embedding == None:
            S = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
            T = dsampler.structure.edgelist
            self.embedding = find_embedding(S,T,verbose=2,max_no_improvement=6000,timeout=600,chainlength_patience=1000,threads=10)
        sampler = FixedEmbeddingComposite(dsampler,self.embedding)
        #self.responses = sampler.sample(bqm,num_reads=num,chain_strength=max(map(abs,list(bqm.quadratic.values())+list(bqm.linear.values()))),postprocess="optimization")
        self.responses = sampler.sample(bqm,num_reads=num,chain_strength=max(map(abs,list(bqm.quadratic.values())+list(bqm.linear.values()))))

        self.solutions = self.model.decode_dimod_response(self.responses,feed_dict={"H1":h1,"H2":h2})
        
        self.best_dist = 100000
        self.best_route = []

        for sol in self.solutions:
            if len(sol[1]) == 0:
                tmp_sol = [ sum(i*val for i,val in sol[0]['x'][j].items()) for j in range(self.size)]
                if self.route_len(tmp_sol) < self.best_dist:
                    self.best_dist = self.route_len(tmp_sol)
                    self.best_route = tmp_sol

    def route_len(self,route):
        return sum(self.dist(route[i],route[(i+1)%self.size]) for i in range(self.size))

    def show_map(self):
        G = nx.Graph()
                
        pos = {}
        for i in range(self.size):
            G.add_node(i)
            pos[i] = self.cities[i]

        if len(self.best_route) != 0:
            for i in range(self.size):
                G.add_edge(self.best_route[i],self.best_route[(i+1)%self.size])
        nx.draw_networkx(G,pos=self.cities)
        plt.show()

    def solve_exact(self):
        from pulp import LpProblem,LpVariable,lpSum,LpStatus,PULP_CBC_CMD

        n_city = self.size
        #Pulpで解く
        p = LpProblem("TSP")
        x = LpVariable.dicts(name="x",indexs=(range(n_city),range(n_city)),cat='Binary')

        p += lpSum(self.dist(i,j)*x[i][j] for i,j in product(range(n_city),range(n_city)))

        for i in range(n_city):
            p += lpSum(x[i][j] for j in range(n_city)) == 1

        for i in range(n_city):
            p += lpSum(x[j][i] for j in range(n_city)) == 1

        V = set(range(n_city))
        for s_len in range(1,n_city-1):
            for S_list in combinations(range(n_city),s_len):
                S = set(S_list)
                _V = V - S
                tmp = 0
                for i,j in product(S,_V):
                    tmp += x[i][j]
                p += tmp >= 1
        
        #solver = COIN_CMD(threads=15)
        solver = PULP_CBC_CMD(threads=15)#,mip=True)
        status = p.solve(solver)
        print(LpStatus[status])

        self.exact_route = [0]
        for i in range(n_city-1):
            for j in range(n_city):
                if x[self.exact_route[i]][j].value() == 1.0:
                    self.exact_route.append(j)
                    break

        self.exact_dist = self.route_len(self.exact_route)


    def optimize_balance(self,num=500,solv="DW_2000Q_6",trial=100):
        import optuna

        def objective(trial):
            h1 = trial.suggest_uniform("h1",0.0,1.0)
            h2 = trial.suggest_uniform("h2",0.0,10.0)

            self.solve_QPU_Fix(h1,h2,num,solv,emb_param=None)

            #if self.best_dist == self.exact_dist:
            #    print(self.responses.first)

            #else:
            #    #return Bat status value
            #    return 10000.0
            count = 0
            for idx,sol in enumerate(self.solutions):
                if len(sol[1]) == 0:
                    tmp_sol = [ sum(i*val for i,val in sol[0]['x'][j].items()) for j in range(self.size)]
                    #もし得られた解が厳密解と誤差10%以内であればOK
                    if self.route_len(tmp_sol) < self.exact_dist*1.1:
                        #decode結果とresponsesのIndexは一致するよね？
                        count +=self.responses.record[idx][2]
            return -count

            #最適ルートと回数だけチェックする
            #return self.best_dist*num/(num+1 - self.responses.first[2])

        study = optuna.create_study(study_name=f"tsp_fix_{self.size}_{num}_{solv}",storage='sqlite:///optuna_study.db',load_if_exists=True)
        study.optimize(objective,n_trials=trial)

if __name__ == "__main__":
    for i in range(5,14):
        p = TSP(i)
        p.solve_exact()
        if i < 9:
            p.optimize_balance(solv="DW_2000Q_6",trial=200)

        p = TSP(i)
        p.solve_exact()
        p.optimize_balance(solv="Advantage_system1.1",trial=200)
        #p = TSP(i)

    #p.solve_exact()
    #p.solve_QPU_Fix()
    #p.solve_QPU_Fix()
    #p.solve_QPU(h1=1.0,h2=2.0,num=1000)
    #p.show_map()