"""
/* Copyright (C) 2023 Lea Trenkwalder (LeaMarion) - All Rights Reserved
 * You may use, distribute and modify this code under the terms of the GNU GENERAL PUBLIC LICENSE v3 license.
 *
 * You should have received access to a copy of the GNU GENERAL PUBLIC LICENSE v3 license with
 * this file.
 */
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib
import itertools as it

import argparse
sys.path.insert(1, "../shared")
sys.path.insert(1, "../SA")
print(sys.path)
from HSC_SA_utils import *


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--par_idx', type=int, default=0, help='index for the parameter choice')
    args = parser.parse_args(argv)
    return args

args = get_args(sys.argv[1:])
sys.path.insert(1, "../shared")
import HSC_env as hsc_env
import HSC_utils as hsc_utils


def calculate_singular_solution(qubit = 4, size = 8, x_type = 'action', set_seed_idx=4, exp = ''):
    """
    Args:

    """
    if x_type == 'gate':
        gate_factor = 2
    else:
        gate_factor = 1

    folder_name = 'MCTS_RESULTS/'+exp+'/START_STATES/'

    s_list, t_list, u_list, action_ops = hsc_utils.generate_ops(n_qubits=qubit, method="SA")
    #print(folder_name)
    s_tup = np.load(folder_name+'state_'+str(set_seed_idx)+'.npy', allow_pickle=True)
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

    folder_name = 'MCTS_RESULTS/'+exp+'/START_STATES/'

    s_list, t_list, u_list, action_ops = hsc_utils.generate_ops(n_qubits=qubit, method="SA")

    s_tup = np.load(folder_name+'state_'+str(set_seed_idx)+'.npy', allow_pickle=True)
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


def generate_csv(table_dict, config):

    f = open("../MCTS/MCTS_RESULTS/MCTS_table"+config+".csv", "+w")

    header_list = ['seed','naivelen','naivelenOG']
    AGENTS = 1
    for num in range(AGENTS):
        header_list.append('actionlen'+str(num))
        header_list.append('actionlenred'+str(num))
    print(header_list)
    header_list.append('eval')
    numbers = range(len(header_list))
    headers = ";".join(header_list) +';\n'
    f.write(headers)


    for seed in table_dict.keys():
        notdone = True
        line_content = str(seed)
        line_content += ';' + str(table_dict[seed]['naivelen']) + ';' + str(table_dict[seed]['naivelenOG'])
        for agent in range(AGENTS):
                percent_minsoln = int(np.round(table_dict[seed][agent]['minsoln']/table_dict[seed]['naivelenOG']*100,0))
                percent_minsolnred = int(np.round(table_dict[seed][agent]['minsolnred']/table_dict[seed]['naivelenOG']*100,0))
                eval = table_dict[seed]['eval']
                line_content += ';'+str(percent_minsoln)+'\%;'+str(percent_minsolnred)+'\%;'+str(eval)

        line_content +=';\n'
        print(line_content)


        f.write(line_content)
    f.close()



SIZE = 8  # Size of target gate set
N_QUBITS = 4  # Number of qubits
CFGIDX_LIST = ["mcts_{}".format(j) for j in [1]]  # List of parameter configurations
SEED_IDX = [0]  # List of seed indices determining the initial target gate set


# Select configuration from the lists above
par_idx = args.par_idx
print('IDX',par_idx)

par_list = list(it.product(CFGIDX_LIST,SEED_IDX))[par_idx]
print('list',par_list)
STRUCT_DICT = hsc_utils.load_json(par_list[0])

SEEDS = 5
AGENTS = 1


file_folder = "MCTS_RESULTS/"+CFGIDX_LIST[0]+"/EVALUATIONS/"
for SEED in range(SEEDS):
    tree_eval = 0
    naive_eval = 0
    total_eval = 0
    for AGENT in range(AGENTS):
        print(SEED, AGENT)

        file_name = "MCTS__size_{}__cfgidx_{}__seed_{}_agent_{}_nqubits_{}_evaluations".format(
                    SIZE, STRUCT_DICT["idx"], SEED, AGENT, N_QUBITS)


        evaluations = np.load(file_folder + file_name + '.npy', allow_pickle=True)
        dict = evaluations.item()
        print(dict)
        tree_eval += dict['interactions']
        naive_eval += dict['naive_evaluations']
        total_eval += tree_eval+naive_eval

    print(tree_eval, naive_eval, total_eval)


seq_average_over = 100
x_type = 'gate'
config = ''
table_path = "../MCTS/MCTS_RESULTS/min_num_gates_" + str(N_QUBITS) + "_qubits.npy"
data = np.load(table_path, allow_pickle=True).item()

# LOAD singluar solution and add it to the table:
N_sin_qubit = []
for idx in range(5):
    N_sin = calculate_singular_solution(qubit=N_QUBITS, size=SIZE, x_type=x_type, set_seed_idx=idx, exp=CFGIDX_LIST[0])
    N_sin_qubit.append(N_sin)
    AGENT = 0
    file_name = "MCTS__size_{}__cfgidx_{}__seed_{}_agent_{}_nqubits_{}_evaluations".format(
        SIZE, STRUCT_DICT["idx"], idx, AGENT, N_QUBITS)

    evaluations = np.load(file_folder + file_name + '.npy', allow_pickle=True)
    dict = evaluations.item()
    tree_eval = dict['interactions']
    naive_eval = dict['naive_evaluations']
    total_eval = tree_eval + naive_eval
    # print(N_sin)
    data[idx].update({"naivelenOG": N_sin})
    data[idx].update({"eval":total_eval})
for idx in range(5):
    N_seq = calculate_sequential_solution(qubit=N_QUBITS, size=SIZE, x_type=x_type, set_seed_idx=idx,
                                          average_over=seq_average_over, exp=CFGIDX_LIST[0])
    data[idx].update({"naivelen": N_seq})
print(data)
generate_csv(data, config)


