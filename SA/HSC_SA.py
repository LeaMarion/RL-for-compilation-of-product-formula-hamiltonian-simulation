"""
/* Copyright (C) 2023 Eleanor Scerri (Elliescerri) modified by Lea Trenkwalder (LeaMarion) - All Rights Reserved
 * You may use, distribute and modify this code under the terms of the GNU GENERAL PUBLIC LICENSE v3 license.
 *
 * You should have received access to a copy of the GNU GENERAL PUBLIC LICENSE v3 license with
 * this file.
 */
"""


from importlib import reload
import numpy as np
import os
import random
import inspect
import itertools as it
import copy
import sys
import time
from datetime import datetime

from tqdm import tqdm
from math import log
from functools import partial
from multiprocessing import Pool, Manager

sys.path.insert(1, "../shared")
import HSC_utils as hsc_utils

reload(hsc_utils)

import argparse


def get_args(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--par_idx', type=int, default=0, help='index for the parameter choice')
	parser.add_argument('--config_file', type=str, default='sa_test', help='name of the config file')
	parser.add_argument('--state', type=int, default=0, help='integer')
	parser.add_argument('--device', type=str, default='cpu', help='assigned device')
	args = parser.parse_args(argv)
	return args


args = get_args(sys.argv[1:])

CFG = args.config_file

CFGIDX_LIST = [CFG]  # List of parameter configurations
EXP = CFGIDX_LIST[0]


ACTIONS_DIR = "SA_RESULTS/"+EXP+"/ACTIONS"
EVALUATIONS_DIR = "SA_RESULTS/"+EXP+"/EVALUATIONS"
START_STATES_DIR = "SA_RESULTS/"+EXP+"/START_STATES"
hsc_utils.check_folders(os.getcwd(), [ACTIONS_DIR, EVALUATIONS_DIR, START_STATES_DIR])


REWARD_IDS_ARR = [0]  # In case of intermediate identity rewards. Leave as 0
ALPHA_ARR = [0.1, 0.25, 0.5, 1.0]  # Initial temperatures
SEED_ARR = np.arange(5)  # List of seed indices determining the initial target gate set


# Select configuration from the lists above
par_idx = args.par_idx

par_list = list(it.product(CFGIDX_LIST, SEED_ARR,
						   ALPHA_ARR, REWARD_IDS_ARR))[par_idx]


CFGIDX = par_list[0]
SEED_IDX = par_list[1]
ALPHA = par_list[2]
REWARD_IDS = par_list[3]

STRUCT_DICT = hsc_utils.load_json(CFGIDX)
N_QUBITS =  STRUCT_DICT["n_qubits"] # Number of qubits
SIZE = STRUCT_DICT["size"]  # SIZE of target gate set

STRATEGY = STRUCT_DICT["anneal_strategy"]  # Annealing strategy
ACTION_LIST_SIZE = STRUCT_DICT["action_size"]  # Action space SIZE

# Whether initial action string should be naive or empty
NAIVE_INIT = (STRUCT_DICT["init"] == "naive")


max_evaluations = STRUCT_DICT["max_evaluations"]
max_steps = STRUCT_DICT["max_steps"]

# Generate full target, source and mapping gates
s_list, t_list, u_list, action_ops = hsc_utils.generate_ops(
	n_qubits=N_QUBITS, method="SA")

# Generate the random subset of target gates to load
# Whilst a list, I left it as s_tup since I use a tuple for RL
s_tup = list(np.random.RandomState(SEED_IDX).choice(s_list, size=SIZE))
np.save(START_STATES_DIR+'/state_'+str(SEED_IDX), s_tup)

# Mapping operator names
u_list_names = [u.__name__ for u in u_list]

# Lists of single and two qubit mapping gates
u_list_single = [u for u in u_list if len(
	inspect.signature(u).parameters) == 2]
u_list_double = [u for u in u_list if len(
	inspect.signature(u).parameters) > 2]

# Number of single and two qubit mapping gates
N_SINGLE = len(u_list_single)
N_DOUBLE = len(u_list_double)

# Action space and corresponding bit space SIZE for random action selection
ACTION_SPACE_SIZE = N_SINGLE*N_QUBITS + N_DOUBLE*(N_QUBITS-1)
ACTION_BITSPACE_SIZE = int(np.ceil(log(ACTION_SPACE_SIZE, 2)))

# Saving directory
DIRECTORY = "SA_results/"

# File names
if NAIVE_INIT:
	NAME = "{}__nqubits_{}__size_{}__alpha_{}__reward_ids_{}__seed_{}__naive_init_fair".format(
		STRATEGY, N_QUBITS, SIZE, ALPHA, REWARD_IDS, SEED_IDX)

else:
	NAME = "{}__maxactions_{}__nqubits_{}__size_{}__alpha_{}__reward_ids_{}__seed_{}_fair".format(
		STRATEGY, ACTION_LIST_SIZE, N_QUBITS, SIZE, ALPHA, REWARD_IDS, SEED_IDX)


def state(s_tup):
	'''
	Create state from set of target ops. Technically an old version of the method in HSC_env.py with the same name.

	Args:
		s_tup: list of target operators.

	Returns:
		state: array representing an enumeration of the target set.

	'''

	if type(s_tup) != list:
		s_tup = [s_tup]

	conv_dict = {"I": 0,
				 "X": 1,
				 "Y": 3,
				 "Z": 2}

	s_tup_ops = [s["g"] for s in s_tup]

	state = np.array([conv_dict[p]
					  for s in s_tup_ops for p in s], dtype=object)

	return state


def action(s_tup, bit_string):
	'''
	Apply mapping gate represented by bit_string to s_tup.

	Args:
		s_tup: list of target operators.
		bit_string: binary representing the index of the mapping gate to be applied.

	Returns:
		s_tup: list of target operators after applying mapping gate.

	'''

	combs = [(q, q+1) for q in range(N_QUBITS-1)]
	action_no = int(bit_string, 2)

	if action_no < N_SINGLE*N_QUBITS:
		s_tup = [u_list_single[action_no//N_QUBITS](action_no % N_QUBITS, s)
				 if s["g"] not in [dict["g"] for dict in t_list] else s for s in s_tup]

	else:
		s_tup = [u_list_double[(action_no-N_SINGLE*N_QUBITS)//len(combs)](*combs[(action_no-N_SINGLE*N_QUBITS) % len(combs)], s) if s["g"] not in [dict["g"] for dict in t_list] else s for s in s_tup]

	return s_tup


def cost_function(s_tup):
	'''
	Cost function. Counts the number of operators in s_tup which haven't yet been mapped to source gates.

	Args:
		s_tup: list of target operators.

	Returns:
		cost: integer representing the cost function.

	'''

	cost = len([s for s in s_tup if s["g"] not in [dict["g"]
												   for dict in t_list]])  # /len(s_tup)

	return cost


def random_neighbour(s_tup, action_list):
	'''
	Choose a random bit-substring (corresponding to an action) and random but position from the action sequence, and flip the bit.
	Ex. Third bit from second action chosen  1001 0(1)00 1000 -> 1001 0(0)00 1000.
	If the new substring is not a valid action, repeat until you end up with a sequence of valid actions.

	Args:
		s_tup: list of target operators.
		action_list: list of mapping gates to apply to s_tup.

	Returns:
		s_tup: list of target operators after applying mapping gates.
		new_action_list: modified list of mapping gates to apply to s_tup.

	'''

	valid_action = False
	new_action_list = copy.deepcopy(action_list)

	while not valid_action:
		rand_position = np.random.choice(range(len(new_action_list)))
		rand_action = np.random.choice(range(ACTION_BITSPACE_SIZE))

		action_str = new_action_list[rand_position]
		action_str = "".join((action_str[:rand_action],
							  "1" if action_str[rand_action] == "0" else "0",
							  action_str[rand_action+1:]))

		if int(action_str, 2) < ACTION_SPACE_SIZE:
			valid_action = True

	new_action_list[rand_position] = action_str
	for action_str in new_action_list:
		s_tup = action(s_tup, action_str)

	return s_tup, new_action_list


def probability(cost, new_cost, temperature):
	'''
	Probability of transitioning.

	Args:
		cost: old cost.
		new_cost: new cost corresponding to changing an action in the sequence.
		temperature: temperature.

	Returns:
		p: transition probability.

	'''

	if new_cost < cost:
		p = 1
	else:
		p = np.exp(- (new_cost - cost) / temperature)

	return p


def temperature(fraction):
	'''
	Calculate annealed temperature.

	Args:
		fraction: current step divided by the maximum steps.

	Returns:
		temperature: annealed temperature.
	'''

	if STRATEGY == "linear":
		temperature = max(ALPHA*0.01, ALPHA*min(1, 1 - fraction))

	elif STRATEGY == "exp":
		temperature = np.exp(max(0.01, ALPHA*min(1, 1 - fraction)))/np.exp(ALPHA)

	return temperature


def annealing(rep, max_steps):
	'''
	Main SA function for parallel computation. Runs SA algorithm

	Args:
		rep: repetition number for statistics.

	Returns:
		action_list_int: list of actions (integers) succesful action
		evaluations: number of evaluations of the cost funtion
		steps: number of steps taken

	'''

	random.seed(rep)
	np.random.seed(rep)
	print("Started {} at {}".format(rep, time.time()))
	sys.stdout.flush()
	evaluations = 0
	action_list_int = []

	if NAIVE_INIT:
		seed_actions = hsc_utils.mapping_gate(
			s_tup, action_ops, N_QUBITS, False, None)
		actions_list_ops = hsc_utils.actionToActionSA(seed_actions)
		action_list = hsc_utils.actionSAToBinary(
			actions_list_ops, u_list_names, N_SINGLE, N_QUBITS, ACTION_BITSPACE_SIZE)
		ACTION_LIST_SIZE = len(action_list)

		if rep == 0:
			print("Max. number of steps: {}".format(max_steps))
		sys.stdout.flush()

		# Check validity of naive initial solution
		s_tup_test = np.copy(s_tup)
		for action_str in action_list:
			s_tup_test = action(s_tup_test, action_str)

		if cost_function(s_tup_test) != 0:
			raise ValueError("Cost using naive approach should be 0")

	else:
		ACTION_LIST_SIZE = STRUCT_DICT["action_size"]
		action_list = [
			"0"*ACTION_BITSPACE_SIZE for j in range(ACTION_LIST_SIZE)]
		max_steps = 500000

	cost = cost_function(s_tup)
	evaluations += 1

	if REWARD_IDS > 0 and cost > 0:
		cost -= REWARD_IDS * \
			len([action for action in action_list if int(
				action, 2) < N_QUBITS])/ACTION_LIST_SIZE

	s_tups, costs, action_lists = [s_tup], [cost], [[int(action, 2) for action in action_list]]
	step = 0
	steps = 0
	while cost != 0 and steps < max_steps:
		fraction = step / float(max_steps)
		T = temperature(fraction)
		no_actions = 1

		new_s_tup, new_action_list = random_neighbour(s_tup, action_list)
		new_cost = cost_function(new_s_tup)
		evaluations +=1
		#'print("Evaluations", evaluations)
		#sys.stdout.flush()

		if REWARD_IDS > 0 and new_cost > 0:
			new_cost -= REWARD_IDS * \
				len([action for action in new_action_list if int(
					action, 2) < N_QUBITS])/ACTION_LIST_SIZE

		if probability(cost, new_cost, T) > np.random.random():
			cost, action_list = new_cost, new_action_list
			steps += 1
			s_tups.append(new_s_tup)


			action_lists.append([int(action, 2) for action in action_list])

		if cost == 0:
			action_list_int = [int(action, 2) for action in action_list]
			break

		step += 1



	print('Evaluations',evaluations)
	print("rep: {}, steps till success: {}".format(rep, steps))
	sys.stdout.flush()
	return action_list_int, evaluations, steps


if __name__ == "__main__":
	start_time = datetime.now()

	#reps_list = tqdm(range(REPS), desc="Running reps")

	#manager = Manager()
	#costs_reps = []
	actions_reps = []
	evaluations_reps = []
	steps_reps = []
	total_evaluations = 0
	reps = 0

	while total_evaluations < max_evaluations:
		action_list, evaluations, steps = annealing(reps, max_steps)
		reps+=1
		total_evaluations += evaluations
		actions_reps.append(action_list)
		evaluations_reps.append(evaluations)
		steps_reps.append(steps)
		action_array = np.array(actions_reps, dtype=object)
		evaluation_array = np.array(evaluations_reps, dtype=object)
		steps_array = np.array(steps_reps, dtype=object)
		reps_array = np.array(reps, dtype=object)
		np.save(ACTIONS_DIR + "/SA_actions_" + NAME + ".npy", action_array)
		np.save(EVALUATIONS_DIR + "/SA_evaluations_" + NAME + ".npy", evaluation_array)
		np.save(EVALUATIONS_DIR + "/SA_steps_" + NAME + ".npy", steps_array)
		np.save(EVALUATIONS_DIR + "/SA_reps_" + NAME + ".npy", reps_array)
		end_time = datetime.now()
		print('The runtime after repetition ',reps,' and', total_evaluations,' evaluations is: ', end_time - start_time)
		sys.stdout.flush()
