# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# 本プログラムは、D-WaveのQPU、HybridSolver、並びにSimulatedAnnealingSampler
# で動作し、それぞれのSamplerの違いを比較できるようにD-Waveのサンプルコードを
# 修正したものです。
# 元のコードは以下
#
#     https://github.com/dwave-examples/simple-ocean-programs/blob/master/Basic_Programs/general_program_qubo.py


# Import the functions and packages that are used
from dwave.system import EmbeddingComposite, DWaveSampler, LeapHybridSampler
from neal import SimulatedAnnealingSampler
from dimod import SampleSet

# Define the problem as a Python dictionary
Q = {('B','B'): 1, 
    ('K','K'): 1, 
    ('A','C'): 2, 
    ('A','K'): -2, 
    ('B','C'): -2}

###### D-WaveのQPU Samplerを用いる方法です。（Leap時間を消費します！）
# Define the sampler that will be used to run the problem
sampler = EmbeddingComposite(DWaveSampler())

# Run the problem on the sampler and print the results
sampleset = sampler.sample_qubo(Q, num_reads = 10)
print(sampleset)

###### Leap Hybrid samplerを用いる方法です。（Leap時間を消費します！）
sampler_hybrid = LeapHybridSampler()
sampleset_hybrid = sampler_hybrid.sample_qubo(Q,time_limit=3)
print(sampleset_hybrid)

###### Simulated Annealing samplerを用いる方法です。（Leap時間を消費しません）
sampler_neal = SimulatedAnnealingSampler()
sampleset_neal = sampler_neal.sample_qubo(Q, num_reads=10)
print(sampleset_neal.aggregate())