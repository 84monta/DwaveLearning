from dwave.system import EmbeddingComposite,DWaveSampler
from pyqubo import Array,Sum,Constraint,Placeholder
import random
from matplotlib import pyplot as plt 
import networkx as nx
from math import sqrt
from itertools import product


class TSP:

    def __init__(self,size=10,seed=1):
        self.size = size
        random.seed(seed)

        self.cities = {}
        for i in range(size):
            self.cities[i] = (random.uniform(0.0,1.0),random.uniform(0.0,1.0))
        
        self.define_model()
    
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
        
        H = Placeholder("H1")*H1 + Placeholder("H2")*H2 + Placeholder("H3")*H3

        #モデル作成
        self.model = H.compile()
    
    def solve_QPU(self,h1=1.0,h2=1.0,h3=1.0,num=100,sol="DW_2000Q_6"):

        sampler = EmbeddingComposite(DWaveSampler(solver=sol))
        Q,offset = self.model.to_qubo(feed_dict={"H1":h1,"H2":h2,"H3":h3})
        responses = sampler.sample_qubo(Q,num_reads=num,chain_strength=5.0,embedding_parameters={"verbose":2,"max_no_improvement":6000,"timeout":600,"chainlength_patience":1000,"threads":15})

        solutions = self.model.decode_dimod_response(responses,feed_dict={"H1":h1,"H2":h2,"H3":h3})
        
        self.best_dist = 100000
        self.best_route = []

        for sol in solutions:
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


if __name__ == "__main__":
    p = TSP(7)
    p.solve_QPU(h1=1.0,h2=2.0,h3=2.0,num=1000)
    p.show_map()