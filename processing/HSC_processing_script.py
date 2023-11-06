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
import argparse
import pathlib

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


def reduce_actions(action_ind, action_seq, action_markers):
	#print('action_ind', action_ind)
	#print('action_seq',action_seq)
	#print('action_markers', action_markers)

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


if __name__ == "__main__":
	args = get_args(sys.argv[1:])
	# Configurations to load
	N_QUBITS = 4  # Number of qubits
	SIZE = 8  # Size of target gate set
	EXP = 'cutoff100'
	CFGIDX = str(N_QUBITS)+"q_"+str(SIZE)+"t"+"_"+EXP

	SEED_LIST = np.arange(5)  # List of seed indices determining the initial target gate set
	AGENT_LIST = np.arange(3)  # List of agents

	#for par_idx in range(5,len(SEED_LIST)*len(AGENT_LIST)):
	for par_idx in range(0, 15):

		par_list = list(it.product([CFGIDX], SEED_LIST, AGENT_LIST))[par_idx]

		CFGIDX = par_list[0]
		SEED = par_list[1]
		AGENT = par_list[2]

		# Generate full target, source and mapping gates
		s_list, t_list, u_list, _ = hsc_utils.generate_ops(
			n_qubits=N_QUBITS, method="DDQN")

		# Generate the random subset of target gates to load
		#SEED_OP = list(np.random.RandomState(SEED).choice(s_list, size=SIZE))
		SEED_OP = np.load("../DDQN/results/"+CFGIDX+"/START_STATES/state_"+str(SEED)+".npy", allow_pickle=True)
		print('SEED_OP', SEED_OP)

		# Mapping operator names
		u_list_names = [u.__name__ for u in u_list]

		# Lists of single and two qubit mapping gates
		u_list_single = [u for u in u_list if len(
			inspect.signature(u).parameters) == 2]
		u_list_double = [u for u in u_list if len(
			inspect.signature(u).parameters) > 2]

		# Replace with data folder
		DIR_FOLDER = "../DDQN/results/"+CFGIDX+"/ACTIONS/"
		FILE_NAME = "HSC_DDQN_model__size_{}__cfgidx_{}__agent_{}__seed_{}_nqubits_{}_torch".format(SIZE,CFGIDX, AGENT, SEED, N_QUBITS)

		if not os.path.exists(DIR_FOLDER+"processed/"):
			os.makedirs(DIR_FOLDER+"processed/")



		actions = []
		first_solns = []
		successful_agent = False

		print(DIR_FOLDER+FILE_NAME + "_action.npy")

		try:
			actions = np.load(DIR_FOLDER+FILE_NAME +"_action.npy", allow_pickle=True)
			#actions = [actions[-1][200:]]

			# action_markers = np.load("ACTION_MARKERS/"+DIR_FOLDER+FILE_NAME.format(SIZE,
			#											  CFGIDX, AGENT, SEED, N_QUBITS)+"_action_markers.npy", allow_pickle=True)

			action_markers = [[None for j in range(len(actions[0]))]]
			#print('action markers',action_markers)

			first_solns = np.load(DIR_FOLDER+FILE_NAME+"_first_solutions.npy", allow_pickle=True)
			#print('first_solutions',first_solns)

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
			#print('indx actions',indexed_actions)
			pool = Pool()
			processed_lengths = pool.starmap(reduce_actions, tqdm(indexed_actions))
			for i in range(len(processed_lengths)):
				print(actions[0][i])
				print(processed_lengths[i])
			print(processed_lengths[-1])
			pool.close()

			print(time.time()-start_time)

		np.save(DIR_FOLDER+"processed/"+FILE_NAME.format(SIZE, CFGIDX, AGENT, SEED, N_QUBITS)+"_action_processed.npy", np.array(processed_lengths))
