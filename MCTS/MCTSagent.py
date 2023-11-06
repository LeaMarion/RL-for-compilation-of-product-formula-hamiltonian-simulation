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
import cirq

from tqdm import tqdm
from time import time

import MCTree as mct

sys.path.insert(1, "../shared")
import HSC_utils as hsc_utils

from importlib import reload
reload(mct)
reload(hsc_utils)



class MCTS():
	def __init__(self, s_list, u_list, t_list, n_qubits, struct_dict, size, seed_idx, agent_idx):

		# File name
		print(seed_idx)
		EXP = struct_dict["idx"]
		self.ACTIONS_DIR = "MCTS_RESULTS/"+EXP+"/ACTIONS"
		self.EVALUATIONS_DIR = "MCTS_RESULTS/"+EXP+"/EVALUATIONS"
		START_STATES_DIR = "MCTS_RESULTS/"+EXP+"/START_STATES"
		seeds_dir = "MCTS_RESULTS/"+EXP+"/SEEDS"
		hsc_utils.check_folders(os.getcwd(), [seeds_dir, self.ACTIONS_DIR, self.EVALUATIONS_DIR, START_STATES_DIR])
		self.NAME = "MCTS__size_{}__cfgidx_{}__seed_{}_agent_{}_nqubits_{}".format(
			size, struct_dict["idx"], seed_idx, agent_idx, n_qubits)

		self.EPS = struct_dict["eps_0"]  # Exploration parameter
		self.MAX_SOLUTIONS = struct_dict["max_solutions"]  # Maximum number of solutions desired
		self.SIZE = size  # Size of target gate set
		self.N_QUBITS = n_qubits  # Number of qubits
		self.forbidden_branches = []  # Branches already explored

		self.root = mct.Node()
		self.root.set_consts(self.N_QUBITS, struct_dict)

		self.root.set_init_vars(s_list=s_list, u_list=u_list, t_list=t_list,
								size=size, seed_idx=seed_idx, pad_idx=struct_dict["pad"])
		start_state = list(self.root.env.source.s_tup)
		print(start_state)
		np.save(START_STATES_DIR + '/state_' + str(seed_idx), start_state)


		# Calculate naive solution length at root node
		naive_gates = hsc_utils.mapping_gate(
			list(self.root.env.source.s_tup), hsc_utils.TestOps(self.N_QUBITS), self.N_QUBITS, False, None)
		naive_gates_idxs = hsc_utils.actionToActionDQN(naive_gates)
		self.naive_len = len(naive_gates_idxs)
		print("Naive solution length: {}".format(self.naive_len))

	def run_tree(self):
		'''
		Runs an iteration of MCTS algorithm. Includes pruning branches so that once a branch is fully explored,
		it's no longer accessible.

		Returns:
			current_node: final node visited, either by expansion or by UCB.
			_trial_branch: path (of actions) traversed to get to current_node.

		'''

		current_node = self.root
		_trial_branch = []
		run_flag = True

		current_node = self.root
		_trial_branch = []
		counter = 0

		while not current_node.is_terminal_node():
			counter += 1
			if not current_node.is_fully_expanded():
				current_node = current_node.expand()
				_trial_branch.append(current_node.action_attribute)

				return current_node, _trial_branch

			else:
				parent_node = current_node
				non_orphans = [child for child in current_node.children if _trial_branch+[
					child.action_attribute] not in [fb["path"] for fb in self.forbidden_branches]]
				current_node = current_node.ucb_policy(self.EPS, non_orphans)

				if not current_node:
					# family is dead, go to grandparent
					current_node = parent_node
					current_node.dead_branch = True

				else:
					_trial_branch.append(current_node.action_attribute)

		if current_node.dead_branch:
			current_node.kill_children()
			removed_count = 0
			for fb in self.forbidden_branches:
				if fb["path"][:current_node.depth] == _trial_branch:
					self.forbidden_branches.remove(fb)
					removed_count += 1

		return current_node, _trial_branch

	def run_mcts(self):
		'''
		Runs MCTS algorithm until maximum number of solutions is found (or some time limit is reached).

		'''

		start_time = time()
		max_depth = 0
		shortest_action = []
		frac_explored = []
		solutions = []
		max_reward = 0
		count = 0
		time_steps = 0
		naive_evaluations = 0

		interactions = 0
		episodes = 0
		while len(solutions) <= self.MAX_SOLUTIONS:
			episodes += 1
			time_steps += 1
			current_node, _trial_branch = self.run_tree()
			interactions += current_node.depth
			max_depth = max(current_node.depth, max_depth)
			if current_node.is_terminal_node():
				self.forbidden_branches.append(
					{"tp": current_node.depth, "path": _trial_branch})
				action_list = []
				end_node = current_node
				if len(current_node.env.source.check_list) == self.SIZE:
					action_list = self.forbidden_branches[-1]["path"]
					print('checklist', len(current_node.env.source.check_list))

					print("Solution length: {}".format(len(action_list)))


					solutions.append(list(reversed(action_list)))
					np.save("{}/{}_action.npy".format(self.ACTIONS_DIR,
													  self.NAME), solutions)


			reward, add_naive_evaluations = current_node.rollout(self.naive_len)
			naive_evaluations += add_naive_evaluations
			current_node.backprop(reward)
			count += 1
			eval_dict = {'interactions': interactions, 'naive_evaluations': naive_evaluations, 'episodes': episodes}

		print("Done", self.root.env.source.s_tup, shortest_action)
		print('Depth', current_node.depth, 'interactions', interactions, 'naive_evaluations', naive_evaluations)
		sys.stdout.flush()
		print("Tree depth: {}".format(current_node.depth))
		np.save("{}/{}_evaluations.npy".format(self.EVALUATIONS_DIR, self.NAME), eval_dict)
