"""
/* Copyright (C) 2023 Eleanor Scerri (Elliescerri) modified by Lea Trenkwalder (LeaMarion) - All Rights Reserved
 * You may use, distribute and modify this code under the terms of the GNU GENERAL PUBLIC LICENSE v3 license.
 *
 * You should have received access to a copy of the GNU GENERAL PUBLIC LICENSE v3 license with
 * this file.
 */
"""

import os
import sys
import pickle
import numpy as np
import openfermion as of
import cirq
import itertools as it
import sys
from sys import argv
import random
import torch


#change to HSC_agent_gpu to use tensorflow
import HSC_agent_gpu_torch as hsc_agent

import argparse


def get_args(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--par_idx', type=int, default=0, help='index for the parameter choice')
	parser.add_argument('--config_file', type=str, default='DDQN_test', help='name of the config file')
	parser.add_argument('--state', type=int, default=0, help='integer')
	parser.add_argument('--device', type=str, default='cpu', help='assigned device')
	parser.add_argument('--train_mode', type=str, default=None, help='training mode')
	parser.add_argument('--num_states', type=int, default=1, help='number of states')
	parser.add_argument('--num_agents', type=int, default=50, help='number of states')


	args = parser.parse_args(argv)
	return args


args = get_args(sys.argv[1:])
sys.path.insert(1, "../shared")
import HSC_utils as hsc_utils

from importlib import reload
reload(hsc_agent)
reload(hsc_utils)


CFG = args.config_file
num_agents = args.num_agents
num_states = args.num_states


CFGIDX_LIST = [CFG]
AGENT_IDX = np.arange(0,num_agents)  # List of agents
SEED_IDX = np.arange(num_states)  # List of seed indices determining the initial target gate set
LOAD_SEED_IDX = np.arange(1)  # List of set seed indices to load for TL
LOAD_EP = [0]  # List of episodes to load for TL
SOURCE_MODEL = [""]



# Select configuration from the lists above
par_idx = args.par_idx
TRAIN_MODE = args.train_mode

if TRAIN_MODE == None:
	print(len(list(it.product(CFGIDX_LIST,SEED_IDX,AGENT_IDX,LOAD_SEED_IDX,LOAD_EP,SOURCE_MODEL))),list(it.product(CFGIDX_LIST,SEED_IDX,AGENT_IDX,LOAD_SEED_IDX,LOAD_EP,SOURCE_MODEL)))
	par_list = list(it.product(CFGIDX_LIST,SEED_IDX,AGENT_IDX,LOAD_SEED_IDX,LOAD_EP,SOURCE_MODEL))[par_idx]
elif TRAIN_MODE == 'random_start':
	par_list = list(it.product(CFGIDX_LIST, AGENT_IDX, LOAD_SEED_IDX, LOAD_EP, SOURCE_MODEL))[par_idx]
	par_list = list(par_list)
	par_list.insert(1,par_list[1])
	par_list = tuple(par_list)

print(par_list)
sys.stdout.flush()
STRUCT_DICT = hsc_utils.load_json(par_list[0])

#-----------QUBIT NUMBER------------------------

N_QUBITS = STRUCT_DICT["n_qubits"]
#-----------TARGET SIZE-------------------------
SIZE = STRUCT_DICT["size"]
# Bool for PADding target gate set
PAD = (STRUCT_DICT['pad']!=-1)

if TRAIN_MODE == 'random_start':
	NUM_TEST_EPISODES = STRUCT_DICT["num_test_episodes"]
	print(NUM_TEST_EPISODES)
	MAX_NUM_STATES = STRUCT_DICT["max_num_states"]
	print(MAX_NUM_STATES)
	TEST_INTERVAL = STRUCT_DICT["test_interval"]
	print(TEST_INTERVAL)

# Get random seeds
TORCH_SEED = torch.seed()
NUMPY_SEED = np.random.get_state()
RANDOM_SEED = random.getstate()
torch.set_num_threads(1)
device = args.device

# Generate full target, source and mapping gates. Target gates are PADded in HSC_env.reset()
s_list, t_list, u_list, _ = hsc_utils.generate_ops(n_qubits=N_QUBITS, method="DDQN", pad=PAD)

seed_folder = "RESULTS/"+CFG+"/RAND_SEED/"
hsc_utils.check_folders(os.getcwd(), [seed_folder])

seed_dict_name = seed_folder+"HSC_DDQN_model__size_{}__cfgidx_{}__agent_{}__seed_{}_nqubits_{}_torch".format(
	SIZE, par_list[0], par_list[2], par_list[1], N_QUBITS)


# Save random seeds
with open("{}.p".format(seed_dict_name), "wb") as fp:
    pickle.dump({"torch":TORCH_SEED, "numpy":NUMPY_SEED, "random":RANDOM_SEED}, fp, protocol=pickle.HIGHEST_PROTOCOL)

# with open(seed_dict_name+".p", 'rb') as f:
#     x = pickle.load(f)


agent = hsc_agent.DQN_agent(s_list=s_list,
							u_list=u_list,
							t_list=t_list,
							n_qubits=N_QUBITS+PAD,
							size=SIZE,
							idx=par_list[0],
							set_seed_idx=par_list[1],
							agent_idx=par_list[2],
							load_set_seed_idx=par_list[3],
							load_ep=par_list[4],
							load_model = par_list[5],
							device = device)


if train_mode == None:
	agent.train(with_test_episode=True)

elif TRAIN_MODE == 'random_start':
	agent.non_det_random_start_train(with_specific_test_episodes=True, num_test_episodes = NUM_TEST_EPISODES, max_num_states= MAX_NUM_STATES, test_interval=TEST_INTERVAL)

elif train_mode == 'specific_state':
	agent.specific_state_train(state_num=args.state)

else:
	raise NotImplementedError("The training mode "+train_mode+" is not implemented yet")




