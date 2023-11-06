"""
/* Copyright (C) 2023 Eleanor Scerri (Elliescerri) - All Rights Reserved
 * You may use, distribute and modify this code under the terms of the GNU GENERAL PUBLIC LICENSE v3 license.
 *
 * You should have received access to a copy of the GNU GENERAL PUBLIC LICENSE v3 license with
 * this file.
 */
"""

import numpy as np
import sys

from collections import defaultdict
from copy import deepcopy

sys.path.insert(1, "../shared")
import HSC_env as hsc_env
import HSC_utils as hsc_utils
import HSC_rewards as hsc_rewards

from importlib import reload
reload(hsc_env)
reload(hsc_utils)
reload(hsc_rewards)


class Node():

	def __init__(self, environment=None, parent=None):
		self.env = environment  # Environment copy for the node
		self.parent = parent  # Parent of node
		self.children = []  # List of child nodes
		self.num_visits = 0  # Number of node visits
		self.node_reward = 0  # Node reward
		self.depth = 0  # Depth of node
		#self.antagonist = False
		self.dead_branch = False  # Boolean to mark if entire branch was already explored


	def set_consts(self, n_qubits, struct_dict):
		'''
		Set node constants.

		Args:
			n_qubits: number of qubits at node.
			struct_dict: parameter dictionary.

		'''

		self.N_QUBITS = n_qubits  # Number of qubits
		self.STRUCT_DICT = struct_dict  # Parameter dictionary
		self.DISCOUNT = self.STRUCT_DICT['gamma']  # Discount factor for rewards
		self.MAX_DEPTH = self.STRUCT_DICT['max_ep_steps']  # Maximum depth allowed

		# If true, child is allowed to have the same actionattribute
		self.REPEAT_ACTIONS = self.STRUCT_DICT['repeat_actions']
		self.REWARD_METHOD = self.STRUCT_DICT['reward_fun']  # Reward function

	def set_init_vars(self,s_list, u_list, t_list, size, seed_idx, pad_idx):
		'''
		Set initial variables for root node, since it does not inherit from parent.

		Args:
			s_list: full list of target gates from which to choose from.
			u_list: mapping gates available.
			t_list: full list of source gates.
			size: size of target gates to use.
			seed_idx: seed for choosing the target gates.
			pad_idx: padding index, at which qubit to add identity operators.

		'''

		if self.depth==0:

			#  Create first environment instance
			seed = np.random.RandomState(seed_idx)
			self.env = hsc_env.HSC_env(s_list=s_list, u_list=u_list, t_list=t_list,
									   n_qubits=self.N_QUBITS, size=size, struct_dict=self.STRUCT_DICT, seed=seed)

			#  State representation of the current set of target gates
			state = self.env.reset(random_init=True, seed=seed, pad_idx=pad_idx)
			#start_state = self.env.s_list_0
			# next line: Transition state for the root node (state, reward, done)
			self.env.reward_funcs.transition = (state, 0, 0)
			self.action_attribute = None  # Action attribute. None for root node

			# Remaining actions avaliable
			self.remaining_actions = [
				a for a in range(self.env.ACTION_SPACE_SIZE)]

		else:
			print('Not allowed to set init vars outside root')


	def set_action_attribute(self, action):
		'''
		Set action attribute, i.e. the action represented by the traversal from the parent to this node.

		'''

		self.remaining_actions = [a for a in range(self.env.ACTION_SPACE_SIZE)]
		self.action_attribute = action
		if (not self.REPEAT_ACTIONS) and (self.parent.action_attribute == action):
			self.remaining_actions.remove(action)

	@staticmethod
	def state(source):
		'''
		Create state from set of target ops. Technically an old version of the method in HSC_env.py with the same name.

		Args:
			source: list of target operators.

		Returns:
			state: array representing an enumeration of the target set.

		'''

		if type(source) not in [tuple, np.ndarray, list]:
			source = (source,)

		conv_dict = {"I": 0,
					 "X": 1,
					 "Y": 3,
					 "Z": 2}

		s_tup_ops = [s["g"] for s in source]

		state = [conv_dict[p] for s in s_tup_ops for p in s]

		return state

	def ucb_policy(self, eps, non_orphans):
		'''
		UCB policy.

		Args:
			eps: exploration parameter
			non_orphans: list of children that don't form part of a path that's already been explored.

		Returns:
			non_orphans[chosen_one]: chosen child node.

		'''

		if len(non_orphans) == 0:
			return

		else:
			if not eps:
				eps = self.node_reward/self.num_visits
			choices_weights = [child.node_reward + eps * np.sqrt(
				np.log(self.num_visits) / child.num_visits) for child in non_orphans]
			chosen_one = np.argmax(choices_weights)

			#if chosen_one == np.argmax([child.node_reward for child in non_orphans]):
			#	non_orphans[chosen_one].antagonist = True
			#print(chosen_one)

			return non_orphans[chosen_one]

	def expand(self):
		'''
		Expand node.

		Returns:
			child_node: new child node created.

		'''

		action = self.remaining_actions.pop()
		child_env = deepcopy(self.env)
		next_transition = child_env.step(action)
		child_env.reward_funcs.transition = next_transition

		child_node = Node(environment=child_env,
						  #transition=next_transition,
						  parent=self)

		child_node.set_consts(self.N_QUBITS, self.STRUCT_DICT)
		child_node.set_action_attribute(action)
		child_node.depth = self.depth+1
		self.children.append(child_node)

		return child_node

	def is_terminal_node(self):
		'''
		Check if node is terminal. This can happen either if the node forms part of a branch already
		explored, or if 'done' parameter is true, i.e. if the target gates sets have all been successfully
		mapped to source gates or maximal number of actions (i.e. depth) has been reached.

		'''

		return self.env.reward_funcs.transition[2] or self.dead_branch

	def kill_children(self):
		'''
		Deletes the children of a node. This is done when the current node is a dead branch, i.e.
		has already been fully explored.

		NOTE: this only makes sense when reward is deterministic, otherwise branch should be visited again
		after it is fully explored.

		'''
		#print(self.children)
		del self.children

	def is_fully_expanded(self):
		'''
		Check if node is fully explanded.

		'''

		return len(self.remaining_actions) == 0

	def rollout(self, naive_len):
		'''
		Calculate rollout at node.
		Currently the naive rollout method is used since it has been the most successful.

		Args:
			naive_len: naive solution length to be compared to the naive solution at the
			node. Unless otherwise stated, this is usually taken to be the naive solution
			at root node.

		Returns:
			reward: reward at current node.

		'''

		rollout_env = deepcopy(self.env)

		# env for rollout functions isn't set up until this point
		self.env.reward_funcs.rollout_env = rollout_env

		if self.depth>0:
			# make sure that pre-rollout, node env and rollout env transitions match
			assert self.env.reward_funcs.transition == rollout_env.reward_funcs.transition, [self.env.reward_funcs.transition, rollout_env.reward_funcs.transition]

		naive_gates = hsc_utils.mapping_gate(
			list(rollout_env.source.s_tup), hsc_utils.TestOps(self.N_QUBITS), self.N_QUBITS, False, None)
		naive_gates_idxs = hsc_utils.actionToActionDQN(naive_gates)
		naive_evaluations = len(naive_gates_idxs)
		if self.REWARD_METHOD == "naive":
			reward = getattr(self.env.reward_funcs,
							 self.REWARD_METHOD)(naive_len)
		else:
			reward = getattr(self.env.reward_funcs, self.REWARD_METHOD)()

		return reward, naive_evaluations

	def backprop(self, reward):
		'''
		Backpropagation.

		'''

		self.num_visits += 1
		self.node_reward += reward

		if self.parent:
			reward *= self.DISCOUNT
			self.parent.backprop(reward)
