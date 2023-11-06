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
import argparse
import pathlib
#import plotly.graph_objects as go
from tabulate import tabulate

print(pathlib.Path(__file__).parent.resolve())
print(pathlib.Path().resolve())

sys.path.insert(1, "../processing")
import HSC_processing as hsc_proc

sys.path.insert(1, "../shared")
import HSC_utils as hsc_utils


from importlib import reload
reload(hsc_utils)
reload(hsc_proc)

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--par_idx', type=int, default=0, help='index for the parameter choice')
    parser.add_argument('--device', type=str, default='cpu', help='assigned device')
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    SIZE = 8  # Size of target gate set
    N_QUBITS = 4  # Number of qubits
    CFGIDX = str(N_QUBITS)+"q_"+str(SIZE)+"t_cutoff100"  # Cofiguration to load
    SEEDS = 5
    AGENTS = 1
    SEED_LIST = np.arange(SEEDS)  # List of seed indices determining the initial target gate set
    AGENT_LIST = np.arange(AGENTS)  # List of agents
    table_content = list()
    table_dict = {}
    letters = list(map(chr, range(97, 123)))
    naive_sol_list = [56,74,72,72,86]
    for idx in range(len(SEED_LIST)):
        table_list = [0] * 2 * AGENTS
        table_dict.update({idx: {}})
        for idx2 in range(len(AGENT_LIST)):
            dict_key = "actionlen"+letters[idx2]
            print(dict_key)


            SEED = idx
            AGENT = idx2

            # Generate full target, source and mapping gates
            s_list, t_list, u_list, _ = hsc_utils.generate_ops(
                n_qubits=N_QUBITS, method="DDQN")

            # Generate the random subset of target gates to load
            SEED_OP = list(np.random.RandomState(SEED).choice(s_list, size=SIZE))

            # Mapping operator names
            u_list_names = [u.__name__ for u in u_list]

            # Lists of single and two qubit mapping gates
            u_list_single = [u for u in u_list if len(inspect.signature(u).parameters) == 2]
            u_list_double = [u for u in u_list if len(inspect.signature(u).parameters) > 2]

            # Replace with data folder
            #DIR_FOLDER = "../DDQN/DDQN_RESULTS/ACTIONS/actionstate/"
            #DIR_FOLDER  = "../DDQN/DDQN_RESULTS/ACTIONS/processed/"
            DIR_FOLDER = "../DDQN/RESULTS/"+CFGIDX+"/ACTIONS/processed/"
            #OLD_DIR = "../DDQN/DDQN_RESULTS/ACTIONS/"
            OLD_DIR = "../DDQN/RESULTS/"+CFGIDX+"/ACTIONS/"
            FILE_NAME = "HSC_DDQN_model__size_{}__cfgidx_{}__agent_{}__seed_{}_nqubits_{}_torch".format(SIZE, CFGIDX, AGENT, SEED, N_QUBITS)



            actions = []
            first_solns = []
            successful_agent = False

            #print(DIR_FOLDER+FILE_NAME +"_action_processed.npy")

            actions = np.load(OLD_DIR+FILE_NAME +
                              "_action.npy", allow_pickle=True)
            actions = [actions[-1][-500:-1]]

            processed_actions = np.load(DIR_FOLDER+FILE_NAME +
                                        "_action_processed.npy", allow_pickle=True)

            #print(processed_actions[5000])
            #print(actions[0])
            action_lengths = []
            for action in actions[0]:
                action_lengths.append(len(action))

            #print(action_lengths)
            query = 0
            for element in processed_actions:
                #print(element['sequence'])
                if element['sequence'] < 500:
                    query+= 0.5 *(element['full']/2)*(element['full']/2+1)
            #print('query',query)

            #print(action_lengths)
            full_reduced_list = []
            for element in processed_actions:
                if element['sequence'] < 5000:
                    full_reduced_list.append(element['inter reduced'])

            #print(action_lengths)
            full_list = []
            for element in processed_actions:
                #print(element['full'])
                if element['sequence'] < 5000:
                    full_list.append(element['full'])

            print("Seed: {}, Agent: {}".format(SEED, AGENT))
            #table_list[AGENT] = min(full_list)
            #table_list[AGENT+3] = min(full_reduced_list)
            print(np.round(min(full_list)/naive_sol_list[idx]*100,0), np.round(min(full_reduced_list)/naive_sol_list[idx]*100,0))
            table_dict[idx].update({dict_key:min(full_list)})
            table_dict[idx].update({dict_key+"tail": min(full_reduced_list)})

            #
            # for i in range(len(actions[0])):
            #     if len(actions[0][i]) == min(action_lengths):
            #         print(len(actions[0][i]))
            #         print(processed_actions[i])
            #         break
            #
            # for i in range(len(actions[0])):
            #     if processed_actions[i]['full reduced'] == min(full_reduced_list):
            #         print(len(actions[0][i]))
            #         print(processed_actions[i])
            #         break
            #
            # for i in range(len(actions[0])):
            #     if processed_actions[i]['full'] == min(full_list):
            #         print(len(actions[0][i]))
            #         print(processed_actions[i])
            #         break
            #
            # print('ALL MINIMAL')
            # for i in range(len(actions[0])):
            #     if processed_actions[i]['full reduced'] == min(full_reduced_list) and processed_actions[i]['full'] == min(full_list):
            #         print(len(actions[0][i]))
            #         print(processed_actions[i])
            #         break

        #print(table_list)
        #table_content.append(table_list)
    #table_content = np.array(table_content)
    #table_content = table_content.T
    print(table_dict)
    table_path = "../DDQN/results/"+CFGIDX+"/min_num_gates_5000.npy"
    print(table_path)
    np.save(table_path,table_dict)







