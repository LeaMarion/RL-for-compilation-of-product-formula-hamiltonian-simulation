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


SIZE = 8  # Size of target gate set
N_QUBITS = 4  # Number of qubits
CFGIDX = "mcts_1"  # Configuration to load



if __name__ == "__main__":

	for SEED in range(2):
		for AGENT in range(1):
			# Generate full target, source and mapping gates
			s_list, t_list, u_list, _ = hsc_utils.generate_ops(
				n_qubits=N_QUBITS, method="MCTS")

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
			DIR_FOLDER = "../MCTS/MCTS_RESULTS/"+CFGIDX+"/ACTIONS/"
			FILE_NAME = "MCTS__size_{}__cfgidx_{}__seed_{}_agent_{}_nqubits_{}_action".format(
				SIZE, CFGIDX, SEED, AGENT, N_QUBITS)

			if not os.path.exists(DIR_FOLDER + "processed/"):
				os.makedirs(DIR_FOLDER + "processed/")

			action_reps = np.load(DIR_FOLDER+FILE_NAME+".npy", allow_pickle=True)

			action_lists = [list(reversed([int(a) for a in action_sequences]))
							for action_sequences in action_reps]

			action_lists = [[hsc_proc.action_name(a, N_QUBITS, u_list_single, u_list_double)
							 for a in action_list] for action_list in action_lists]

			action_markers = [None for j in range(len(action_lists))]

			start_time = time.time()
			indexed_actions = list(
				zip(range(len(action_lists)), action_lists, action_markers))

			pool = Pool()
			processed_lengths = pool.starmap(reduce_actions, tqdm(indexed_actions))
			pool.close()

			print(time.time()-start_time)
			print()

			np.save(DIR_FOLDER+"processed/"+FILE_NAME +
					"_processed.npy", np.array(processed_lengths))
