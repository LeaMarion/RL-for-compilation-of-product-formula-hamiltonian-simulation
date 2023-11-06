"""
/* Copyright (C) 2023 Eleanor Scerri (Elliescerri) - All Rights Reserved
 * You may use, distribute and modify this code under the terms of the GNU GENERAL PUBLIC LICENSE v3 license.
 *
 * You should have received access to a copy of the GNU GENERAL PUBLIC LICENSE v3 license with
 * this file.
 */
"""

import numpy as np
import os
import sys
import itertools as it
import argparse
import cirq

from tqdm import tqdm
from time import time
from datetime import datetime

import MCTSagent as mcts

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--par_idx', type=int, default=0, help='index for the parameter choice')
    args = parser.parse_args(argv)
    return args

args = get_args(sys.argv[1:])
sys.path.insert(1, "../shared")
import HSC_env as hsc_env
import HSC_utils as hsc_utils

from importlib import reload
reload(hsc_utils)

# Select configuration from the lists above
par_idx = args.par_idx

SEEDS = [0]  # List of seed indices determining the initial target gate set
CFGIDX_LIST = ["mcts_{}".format(j) for j in [1]]  # List of parameter configurations
par_list = list(it.product(CFGIDX_LIST))[par_idx]
print('list',len(par_list),par_list)
STRUCT_DICT = hsc_utils.load_json(par_list[0])

SIZE = STRUCT_DICT["size"]  # Size of target gate set
N_QUBITS = STRUCT_DICT["qubits"] # Number of qubits



# Run MCTS
print("Started")
agents = 1
start_time = datetime.now()

for cfg in CFGIDX_LIST:
    STRUCT_DICT = hsc_utils.load_json(cfg)
    # Generate full target, source and mapping gates. Target gates are PADded in HSC_env.reset()
    s_list, t_list, u_list, _ = hsc_utils.generate_ops(n_qubits=N_QUBITS, method="MCTS")
    for seed in SEEDS:
        for agent in range(agents):
            hsc_utils.check_folders(os.getcwd(), ["MCTS_RESULTS"])
            mcts_test = mcts.MCTS(s_list=s_list, u_list=u_list, t_list=t_list, n_qubits=N_QUBITS, size=SIZE, struct_dict=STRUCT_DICT, seed_idx = seed, agent_idx = agent)
            mcts_test.run_mcts()
            end_time = datetime.now()
            print('The runtime is: ', end_time - start_time)
            sys.stdout.flush()
