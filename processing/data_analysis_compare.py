"""
/* Copyright (C) 2023 Lea Trenkwalder (LeaMarion) - All Rights Reserved
 * You may use, distribute and modify this code under the terms of the GNU GENERAL PUBLIC LICENSE v3 license.
 *
 * You should have received access to a copy of the GNU GENERAL PUBLIC LICENSE v3 license with
 * this file.
 */
"""

import numpy as np
import os
import sys
import cirq
import itertools as it
import inspect
from tqdm import tqdm, trange
from multiprocessing import Pool, Process, Manager
from functools import partial
import time


sys.path.insert(1, "../processing")
import HSC_processing as hsc_proc

sys.path.insert(1, "../shared")
import HSC_utils as hsc_utils

sys.path.insert(1, "../DDQN")
import HSC_processing as hsc_proc

sys.path.insert(1, "../SA")
import HSC_utils as hsc_utils

sys.path.insert(1, "../MCTS")
import HSC_utils as hsc_utils
from HSC_SA_utils import *

from importlib import reload
reload(hsc_utils)
reload(hsc_proc)

def generate_csv(data_rl, data_sa, data_mcts, table_type):

    f = open("../DDQN/results/table_compare.csv", "+w")

    header_list = ['seed','naivelen','naivelenred','rl','mcts','sa','rlred', 'mctsred', 'sared']
    numbers = range(len(header_list))
    header_dict = dict(zip(numbers, header_list))
    print(header_dict)
    headers = ";".join(header_list) +';\n'
    #print(headers)
    f.write(headers)
    print(header_list)
    for seed in range(5):
        line_content = str(seed) + ';'
        line_content += str(data_rl[seed]["naivelen"]) + ';'
        line_content += str(data_rl[seed]["naivelenOG"]) + ';'
        if table_type == 'percent':
            line_content += str(int(np.round(data_rl[seed]['actionlena']/data_rl[seed]["naivelenOG"]*100,0))) + '\%;'
            line_content += str(int(np.round(data_mcts[seed][0]['minsoln']/data_rl[seed]["naivelenOG"]*100,0))) + '\%;'
            line_content += str(int(np.round(data_sa[seed][0.25]['minsoln']/data_rl[seed]["naivelenOG"]*100,0))) + '\%;'
            line_content += str(int(np.round(data_rl[seed]['actionlenatail']/data_rl[seed]["naivelenOG"]*100,0))) + '\%;'
            line_content += str(int(np.round(data_mcts[seed][0]['minsolnred']/data_rl[seed]["naivelenOG"]*100,0))) + '\%;'
            line_content += str(int(np.round(data_sa[seed][0.25]['minsolnred']/data_rl[seed]["naivelenOG"]*100,0))) + '\%;'
        else:
            line_content += str(data_rl[seed]['actionlena']) + ';'
            line_content += str(data_mcts[seed][0]['minsoln']) + ';'
            line_content += str(data_sa[seed][0.25]['minsoln']) + ';'
            line_content += str(data_rl[seed]['actionlenatail']) + ';'
            line_content += str(data_mcts[seed][0]['minsolnred']) + ';'
            line_content += str(data_sa[seed][0.25]['minsolnred']) + ';'
        print(line_content)
        f.write(line_content + '\n')
    f.close()


def calculate_singular_solution(qubit = 4, size = 8, x_type = 'action', set_seed_idx=4, exp = 'overlap'):
    """
    Args:

    """
    if x_type == 'gate':
        gate_factor = 2
    else:
        gate_factor = 1

    folder_name = str(qubit)+'q_'+str(size)+'t'+exp

    s_list, t_list, u_list, action_ops = hsc_utils.generate_ops(n_qubits=qubit, method="SA")
    #print(folder_name)
    s_tup = np.load('../DDQN/results/'+folder_name+'/START_STATES/state_'+str(set_seed_idx)+'.npy', allow_pickle=True)
    actions = mapping_gate(s_tup, action_ops, qubit, False, None)
    length = 0
    for action in actions:
        length += len(action)
    return length*gate_factor

def calculate_sequential_solution(qubit = 4, size = 8, x_type = 'action', set_seed_idx=4, exp = 'overlap', average_over=100):
    """
    Args:

    """
    if x_type == 'gate':
        gate_factor = 2
    else:
        gate_factor = 1

    folder_name = str(qubit)+'q_'+str(size)+'t'+exp

    s_list, t_list, u_list, action_ops = hsc_utils.generate_ops(n_qubits=qubit, method="SA")

    s_tup = np.load('../DDQN/results/'+folder_name+'/START_STATES/state_'+str(set_seed_idx)+'.npy', allow_pickle=True)
    actions = mapping_gate(s_tup, action_ops, qubit, False, None)

    seed = np.random.RandomState(set_seed_idx)
    avg_num_actions = 0
    avg = average_over
    for i in range(avg):
        num_actions = 0
        iters = seed.permutation(len(actions))
        for idx in iters[:-1]:
            num_actions+=2*len(actions[idx])
        idx = iters[-1]
        num_actions += len(actions[idx])
        avg_num_actions += num_actions
    avg_num_actions=avg_num_actions/avg
    return avg_num_actions*gate_factor


best_data_dict = {}

#load results DDQN
SEEDS = 5
qubit = 4
size = 8
seq_average_over = 100
exp = '_cutoff100'
CONFIG = str(qubit) + 'q_' + str(size) + 't' + exp
table_path = "../DDQN/results/" + CONFIG + "/min_num_gates_5000.npy"
data_rl = np.load(table_path, allow_pickle=True).item()
print(data_rl)
X_TYPE = "gate"
table_type = 'percent'

for idx in range(5):
    N_sin = calculate_singular_solution(qubit=qubit, size=size, x_type=X_TYPE, set_seed_idx=idx, exp=exp)
    data_rl[idx].update({"naivelenOG": N_sin})
for idx in range(5):
    N_seq = calculate_sequential_solution(qubit=qubit, size=size, x_type=X_TYPE, set_seed_idx=idx, exp=exp,
                                          average_over=seq_average_over)
    data_rl[idx].update({"naivelen": N_seq})


#load results SA
config = 'naive'
eval_threshold = 500000
table_path_naive = "../SA/SA_RESULTS/min_num_gates_" + str(qubit) + "_qubits_" + config + "_eval_"+str(eval_threshold)+".npy"
data_sa = np.load(table_path_naive, allow_pickle=True).item()
print(data_sa)


#load results MCTS
table_path = "../MCTS/MCTS_RESULTS/min_num_gates_" + str(qubit) + "_qubits.npy"
data_mcts = np.load(table_path, allow_pickle=True).item()
print(data_mcts)


generate_csv(data_rl, data_sa, data_mcts,table_type)



