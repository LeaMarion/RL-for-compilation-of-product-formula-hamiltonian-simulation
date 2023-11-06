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

from importlib import reload
reload(hsc_utils)
reload(hsc_proc)



SIZE = 8  # Size of target gate set
N_QUBITS = 4  # Number of qubits
STRATEGY = "linear"  # Annealing strategy
MAX_ACTIONS = 100  # Action space SIZE
REPS = 100  # Number of experiments
CFG_TYPE = "empty"
if CFG_TYPE == "naive":
	NAIVE = True
else:
	NAIVE = False
REWARD = "nonnormalized"
EXP = "sa_"+CFG_TYPE+"init_1Mio"

SEED_LIST = np.arange(5)  # List of seed indices determining the initial target gate set
ALPHA_LIST = [0.1, 0.25, 0.5, 1.0]  # Initial temperatures
REWARD_IDS_LIST = [0]  # In case of intermediate identity rewards. Leave as 0
par_idx =0 #int(sys.argv[1])

par_list = list(it.product(SEED_LIST, ALPHA_LIST, REWARD_IDS_LIST))[par_idx]

SEED = par_list[0]
ALPHA = par_list[1]
REWARD_IDS = par_list[2]
print(par_list)

# Generate full target, source and mapping gates
s_list, t_list, u_list, _ = hsc_utils.generate_ops(
	n_qubits=N_QUBITS, method="SA")

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
DIR_FOLDER = "../SA/SA_RESULTS/"+EXP+"/ACTIONS/"



if __name__ == "__main__":

#	actions = np.load(DIR_FOLDER+FILE_NAME+"_processed.npy", allow_pickle=True)
#	print(SA_solutions)

	actions = []
	first_solns = []
	successful_agent = False

	table_content = list()
	table_dict = {}
	dict_key = ["alpha","minsoln","minsolnred","reps","evals"]
	eval_threshold = 500000


	for SEED in range(len(SEED_LIST)):
		table_dict.update({SEED:{}})
		for ALPHA in ALPHA_LIST:
			table_dict[SEED].update({ALPHA:{}})


			if NAIVE:
				FILE_NAME = "SA_actions_{}__nqubits_{}__size_{}__alpha_{}__reward_ids_{}__seed_{}__naive_init{}_fair".format(
					STRATEGY, N_QUBITS, SIZE, ALPHA, REWARD_IDS, SEED, (REWARD == "normalized") * ("_" + REWARD))
			else:
				FILE_NAME = "SA_actions_{}__maxactions_{}__nqubits_{}__size_{}__alpha_{}__reward_ids_{}__seed_{}_{}fair".format(
					STRATEGY, MAX_ACTIONS, N_QUBITS, SIZE, ALPHA, REWARD_IDS, SEED,
					(REWARD == "normalized") * ("_" + REWARD))
				print(FILE_NAME)

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
			# DIR_FOLDER = "../DDQN/DDQN_RESULTS/ACTIONS/actionstate/"
			NAME = DIR_FOLDER+'processed/'+FILE_NAME+ "_processed.npy"
			OLD_NAME = DIR_FOLDER+FILE_NAME+".npy"


			actions = []
			first_solns = []
			successful_agent = False

			processed_actions = np.load(NAME, allow_pickle=True)


			schedule = 'linear'
			if CFG_TYPE == 'naive':
				init = 'naive_init_'
				print(init)
				file_name_start = "SA_evaluations_"
				folder = '../SA/SA_RESULTS/' + EXP + '/EVALUATIONS/'
				reward_ids = 0
				file_name = file_name_start + schedule + '__nqubits_' + str(N_QUBITS) + '__size_' + str(
					SIZE) + '__alpha_' + str(ALPHA) + '__reward_ids_' + str(reward_ids) + '__seed_' + str(
					SEED) + '__' + init + 'fair'
				evaluations = np.load(folder + file_name + '.npy', allow_pickle=True)

			elif CFG_TYPE == 'empty':
				init = ''
				reward_ids = 0
				folder = '../SA/SA_RESULTS/' + EXP + '/EVALUATIONS/'
				file_name_start = "SA_evaluations_"
				file_name = file_name_start + schedule + '__maxactions_' + str(100) + '__nqubits_' + str(
					N_QUBITS) + '__size_' + str(
					SIZE) + '__alpha_' + str(ALPHA) + '__reward_ids_' + str(reward_ids) + '__seed_' + str(
					SEED) + '_' + init + 'fair'
				evaluations = np.load(folder + file_name + '.npy', allow_pickle=True)
			else:
				reps = len(processed_actions)

			eval = 0
			reps = 0
			for rep in evaluations:
				reps += 1
				eval += rep
				if eval > eval_threshold:
					break

			full_list = []

			for element in processed_actions[:reps]:
				full_list.append(element['full'])

			full_reduced_list = []
			for element in processed_actions[:reps]:
				full_reduced_list.append(element['full reduced'])



			print("Seed: {}, Alpha: {}".format(SEED, ALPHA))
			table_dict[SEED][ALPHA].update({dict_key[0]: ALPHA})
			table_dict[SEED][ALPHA].update({dict_key[1]: min(full_list)})
			table_dict[SEED][ALPHA].update({dict_key[2]: min(full_reduced_list)})
			table_dict[SEED][ALPHA].update({dict_key[3]: reps})
			table_dict[SEED][ALPHA].update({dict_key[4]: eval})


	table_path = "../SA/SA_RESULTS/min_num_gates_4_qubits_"+CFG_TYPE+"_eval_"+str(eval_threshold)+".npy"
	print(table_path)
	print(table_dict)
	np.save(table_path, table_dict)



#			table_list[AGENT] = min(full_list)
#			table_list[AGENT + 3] = min(full_reduced_list)

# 			for i in range(len(processed_actions[0])):
# #				if len(actions[0][i]) == min(action_lengths):
# 					print(len(actions[0][i]))
# 					print(processed_actions[i])
# 					break
#
# 			for i in range(len(processed_actions[0])):
# 				if processed_actions[i]['full reduced'] == min(full_reduced_list):
# 					print(len(actions[0][i]))
# 					print(processed_actions[i])
# 					break
#
# 			for i in range(len(processed_actions[0])):
# 				if processed_actions[i]['full'] == min(full_list):
# 					print(len(actions[0][i]))
# 					print(processed_actions[i])
# 					break
#
# 			print('ALL MINIMAL')
# 			for i in range(len(processed_actions[0])):
# 				if processed_actions[i]['full reduced'] == min(full_reduced_list) and processed_actions[i]['full'] == min(
# 						full_list):
# 					print(len(actions[0][i]))
# 					print(processed_actions[i])
# 					break
# 		#print(table_list)
		#table_content.append(table_list)
	#table_content = np.array(table_content)
	#table_content = table_content.T
	#np.save("../DDQN/results/" + CFGIDX + "/min_num_gates.npy", table_content)



