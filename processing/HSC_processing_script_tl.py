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

from importlib import reload
reload(hsc_utils)
reload(hsc_proc)


def reduce_actions(action_ind, action_seq, action_markers):

	# Reshape if still in old obsolete format, which was fixed on 05/01/2021
	if len(np.shape(action_seq)) > 1:
		action_seq = np.reshape(action_seq, len(action_seq))

	if np.any(action_markers):
		check = False
	else:
		check = True

	action_markers, action_length_full = hsc_proc.action_length(
		action_seq, N_QUBITS, u_list_single, u_list_double, SEED_OP, t_list, action_markers, check, False, False)[0:2]

	action_length_reduced_tail = hsc_proc.action_length(
		action_seq, N_QUBITS, u_list_single, u_list_double, SEED_OP, t_list, action_markers, False, True, False)[1]

	action_length_reduced_full = hsc_proc.action_length(
		action_seq, N_QUBITS, u_list_single, u_list_double, SEED_OP, t_list, action_markers, False, True, True)[1]

	return {"sequence": action_ind, "full": action_length_full, "inter reduced": action_length_reduced_tail, "full reduced": action_length_reduced_full}


CFGIDX_LIST = ["536_100s_loadpad"]
SIZE = 8
N_QUBITS = 5

AGENT_LIST = np.arange(3)
LOAD_EP_LIST = [500, 1000, 5000, 10000, 50000]
LOAD_SEED_LIST = np.arange(4)

par_idx = int(sys.argv[1])

par_list = list(it.product(CFGIDX_LIST, AGENT_LIST,
						   LOAD_SEED_LIST, LOAD_EP_LIST))[par_idx]

CFGIDX = par_list[0]
AGENT = par_list[2]
LOAD_SEED = par_list[2]
LOAD_EP = par_list[3]
SEED = 4

# Generate full target, source and mapping gates
s_list, t_list, u_list, _ = hsc_utils.generate_ops(
	n_qubits=N_QUBITS, method="DDQN_TL")

# Generate the random subset of target gates to load
SEED_OP = list(np.random.RandomState(SEED).choice(s_list, size=SIZE))

# Mapping operator names
u_list_names = [u.__name__ for u in u_list]

# Lists of single and two qubit mapping gates
u_list_single = [u for u in u_list if len(
	inspect.signature(u).parameters) == 2]
u_list_double = [u for u in u_list if len(
	inspect.signature(u).parameters) > 2]

# Replace with data folder
DIR_FOLDER = "../DDQN/DDQN_RESULTS/ACTIONS/"
FILE_NAME = "HSC_DDQN_model__size_{}__cfgidx_{}__agent_{}__seed_{}_nqubits_{}_load_seed_{}__load_ep_{}_torch".format(SIZE, CFGIDX,
																													 AGENT, SEED, N_QUBITS, LOAD_SEED, LOAD_EP)

if not os.path.exists(DIR_FOLDER+"processed/"):
	os.makedirs(DIR_FOLDER+"processed/")

if __name__ == "__main__":

	actions = []
	first_solns = []
	successful_agent = False

	try:
		actions = np.load(DIR_FOLDER+FILE_NAME +
						  "_action.npy", allow_pickle=True)

		print(np.shape(actions))

		action_markers = [[None for j in range(len(actions[0]))]]

		first_solns = np.load(DIR_FOLDER+FILE_NAME +
							  "_first_solutions.npy", allow_pickle=True)

		if first_solns[-1] > -1:
			successful_agent = True

	except:
		raise ValueError("Failed to load actions")

	print("Seed: {}, Agent: {}".format(SEED, AGENT))

	if not successful_agent:
		processed_lengths = np.array(
			[{"sequence": -1, "full": -1, "inter reduced": -1, "full reduced": -1}])

	else:
		start_time = time.time()
		indexed_actions = list(
			zip(range(len(actions[0])), actions[0], action_markers[0]))
		pool = Pool()
		processed_lengths = pool.starmap(reduce_actions, tqdm(indexed_actions))
		pool.close()

		print(time.time()-start_time)
		print()

	np.save(DIR_FOLDER+"processed/"+FILE_NAME.format(SIZE, CFGIDX, AGENT, N_QUBITS,
													 LOAD_SEED, LOAD_EP)+"_action_processed.npy", np.array(processed_lengths))
