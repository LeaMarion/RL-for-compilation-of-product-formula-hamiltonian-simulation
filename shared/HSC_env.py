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
import random
import numpy as np
import cirq
import inspect
from collections import deque
import itertools as it
import argparse
from typing import Tuple, Union, List, Dict, Mapping, Deque
import pathlib

sys.path.insert(1, "../shared")
import HSC_utils as hsc_utils
import HSC_rewards as hsc_rewards


from importlib import reload
reload(hsc_utils)
reload(hsc_rewards)

class S_obj():

	def __init__(self,
				s_list: List[Mapping[str, Union[str, complex]]],
				u_list: List[Mapping[str, Union[str, complex]]],
				t_list: List[Mapping[str, Union[str, complex]]],
				n_qubits: int):
		#print('s_object, s_list', s_list)
		self.s_tup = tuple(s_list)  # Tuple of target gates
		#print('s_tup',self.s_tup)

		self.u_list = u_list  # List of allowed mapping gates
		self.t_list = t_list  # List of allowed target gates
		self.t_list_strs = [dict["g"] for dict in self.t_list]
		#print('actions',self.u_list)
		#print('target_list',self.t_list)

		# Lists of single and two qubit mapping gates
		self.u_list_SINGLE = [u for u in self.u_list if len(
			inspect.signature(u).parameters) == 2]
		self.u_list_DOUBLE = [u for u in self.u_list if len(
			inspect.signature(u).parameters) > 2]

		# Number of single and two qubit mapping gates
		self.N_SINGLE = len(self.u_list_SINGLE)
		self.N_DOUBLE = len(self.u_list_DOUBLE)


		self.check_list = []  # List containing the successfully mapped gates
		self.N_QUBITS = n_qubits  # Number of qubits
		self.COMBS = [(q, q+1) for q in range(self.N_QUBITS-1)]  # Qubit pairings for the 2 qubit gates
		self.prev_overlap = 0
		self.overlap = 0



	def action_name(self, choice: int) -> Dict[str, Union[int, Tuple]]:
		'''
		Get mapping gate given an index.

		Args:
			choice: index of mapping gate.

		Returns:
			mapping gate in the format {"g": gate name, "q": qubits}.

		'''

		if choice < self.N_SINGLE*self.N_QUBITS:
			return {"g": self.u_list_SINGLE[choice//self.N_QUBITS].__name__, "q": choice % self.N_QUBITS}

		else:
			return {"g": self.u_list_DOUBLE[(choice-self.N_SINGLE*self.N_QUBITS)//len(self.COMBS)].__name__, "q": self.COMBS[(choice-self.N_SINGLE*self.N_QUBITS) % len(self.COMBS)]}

	def action(self, choice: int, baseline: bool = False):
		'''
		Apply action on remaining target gates.

		'''

		#Check for elements already in t_list
		# baseline checks if this is the first step in the episode
		#print(choice)
		if baseline:
			#print(self.s_tup)
			self.check_list = [
				p_string["g"] for p_string in self.s_tup if p_string["g"] in self.t_list_strs]
			self.s_tup = tuple([s for s in self.s_tup if s["g"] not in self.check_list])
			list = [s["g"] for s in self.s_tup]
			self.overlap = self.gate_overlap(list,self.t_list_strs)
			self.prev_overlap = self.overlap
			#  print(self.overlap)
			#print('checklist start',self.check_list)
			#print('t_list_strs', self.t_list_strs)
			#if the elements are removed from the state, they also have to be removed here


		else:
			#print('prev',self.s_tup)
			if choice < self.N_SINGLE*self.N_QUBITS:
				#print('single',self.s_tup)
				#self.s_tup = tuple([self.u_list_SINGLE[choice//self.N_QUBITS](choice % self.N_QUBITS, s) if s["g"] not in self.check_list else s for s in self.s_tup])
				#this line is needed with you want to keep the elements that have already been mapped to the set
				#self.s_tup = tuple([self.u_list_SINGLE[choice // self.N_QUBITS](choice % self.N_QUBITS, s) for s in self.s_tup if s["g"] not in self.check_list])
				self.s_tup = tuple([self.u_list_SINGLE[choice // self.N_QUBITS](choice % self.N_QUBITS, s) for s in self.s_tup])
			else:
				#print('double', self.s_tup)
				#self.s_tup = tuple([self.u_list_DOUBLE[(choice-self.N_SINGLE*self.N_QUBITS)//len(self.COMBS)](*self.COMBS[(choice-self.N_SINGLE*self.N_QUBITS) % len(self.COMBS)], s) if s["g"] not in self.check_list else s for s in self.s_tup])
				#self.s_tup = tuple([self.u_list_DOUBLE[(choice-self.N_SINGLE*self.N_QUBITS)//len(self.COMBS)](*self.COMBS[(choice-self.N_SINGLE*self.N_QUBITS) % len(self.COMBS)], s) for s in self.s_tup if s["g"] not in self.check_list])
				self.s_tup = tuple([self.u_list_DOUBLE[(choice - self.N_SINGLE * self.N_QUBITS) // len(self.COMBS)](*self.COMBS[(choice - self.N_SINGLE * self.N_QUBITS) % len(self.COMBS)], s) for s in self.s_tup])
			#print('after',self.s_tup)
			# Add successfully mapped operators.
			# This should be modified to just append, and not check all remaining ops...
			#print([p_string["g"] for p_string in self.s_tup])
			#print(self.t_list_strs)
			#original line of code
			#print('target_list',self.t_list_strs)
			self.new_check_list_elements = [p_string["g"] for p_string in self.s_tup if p_string["g"] in self.t_list_strs]
			#print('new_checklist_element',self.new_check_list_elements)
			self.check_list += self.new_check_list_elements
			# remove the elements that where just found in the state and are thus elements of the self.new_check_list_elements
			self.s_tup = tuple([s for s in self.s_tup if s["g"] not in self.new_check_list_elements])
			#calculating overlap
			num_new_elements = len(self.new_check_list_elements)
			if num_new_elements > 0:
				self.prev_overlap = self.overlap-self.N_QUBITS*num_new_elements
			else:
				self.prev_overlap = self.overlap
			list = [s["g"] for s in self.s_tup]
			self.overlap = self.gate_overlap(list, self.t_list_strs)
			#print(self.overlap-self.prev_overlap)
			#print('after', self.s_tup)
			#print('checklist', self.check_list)

			#only necessary if the state is not the target state but the source gate set
			#add_list = [p_string["g"] for p_string in self.s_tup if p_string["g"] in self.t_list_strs and p_string["g"] not in self.check_list]
			#print('add_list',add_list)
			#self.check_list = self.check_list + add_list
			#print('checklist',self.check_list)

	def gate_overlap(self, state, native_set):
		""" returns the sum of the largest overlap of each gate with each gate in the navtive gate set """
		total_overlap = 0
		for gate in state:
			largest_overlap = 0
			for native_gate in native_set:
				overlap = 0
				for a, b in zip(gate, native_gate):
					if a == b:
						overlap += 1
				if overlap > largest_overlap:
					largest_overlap = overlap
			total_overlap += largest_overlap
		return total_overlap


class HSC_env():
	def __init__(self,
				s_list: List[Mapping[str, Union[str, complex]]],
				u_list: List[Mapping[str, Union[str, complex]]],
				t_list: List[Mapping[str, Union[str, complex]]],
				n_qubits: int,
				size: int,
				struct_dict: Mapping[str, Union[float, int, str]], seed: int):

		self.N_QUBITS = n_qubits  # Number of qubits
		self.SIZE = size  # Size of target gate set

		self.s_list = s_list  # list of source gates
		self.u_list = u_list  # list of allowed actions in function format
		self.t_list = t_list  # list of allowed targets
		#print('s_list',self.s_list)
		#print('u_list', self.u_list)
		#print('t_list', self.t_list)
		self.CFGIDX = struct_dict["idx"]

		self.MAX_EPISODE_STEPS = struct_dict["max_ep_steps"]  # Max. steps for each episode
		self.REWARD_METHOD = struct_dict["reward_fun"]  # Reward function
		self.STRATEGY = struct_dict["strategy"]  # Method used, ex: "ddqn", "sa",...


		if self.STRATEGY == "ddqn":
			# Length of action buffer at the end of the state
			self.STATE_MEM_SIZE = struct_dict["state_mem_size"]

			# Max. number of times the same action can be repeated before penalizing
			self.MAX_ACTION_BUFFER = self.STATE_MEM_SIZE

		else:
			self.STATE_MEM_SIZE = 0
			self.MAX_ACTION_BUFFER = 0

		try:
			# Additional size and number of qubits in case the loaded model differs in size
			if struct_dict["tl_strategy"] == "pad_destination_model":
				self.OLD_N_QUBITS = struct_dict["source_n_qubits"]
				self.OLD_SIZE = struct_dict["source_size"]
			else:
				self.OLD_N_QUBITS = self.N_QUBITS
				self.OLD_SIZE = self.SIZE

		except:
			self.OLD_N_QUBITS = self.N_QUBITS
			self.OLD_SIZE = self.SIZE

		# Lists of single and two qubit mapping gates
		self.u_list_SINGLE = [u for u in self.u_list if len(
			inspect.signature(u).parameters) == 2]
		self.u_list_DOUBLE = [u for u in self.u_list if len(
			inspect.signature(u).parameters) > 2]

		# Number of single and two qubit mapping gates
		self.N_SINGLE = len(self.u_list_SINGLE)
		self.N_DOUBLE = len(self.u_list_DOUBLE)
		#print('number of actions', self.N_SINGLE + self.N_DOUBLE)

		if size > 0:
			# Set environment space size
			self.ENV_SPACE_SIZE = 4*(self.SIZE)*self.N_QUBITS
			#uncomment: to add actions to state
			#self.ENV_SPACE_SIZE += self.STATE_MEM_SIZE


		elif size < 0:
			# Obsolete case. Constructed when input data was stacked
			self.ENV_SPACE_SIZE = self.N_QUBITS
			HSC_env.state_lookup = [cirq.I,cirq.X,cirq.Y,cirq.Z]

		else:
			# Obsolete case
			self.ENV_SPACE_SIZE = len(self.s_list)*self.N_QUBITS

		# Number of actions available
		self.ACTION_SPACE_SIZE = self.N_SINGLE*self.N_QUBITS + \
			self.N_DOUBLE*(self.N_QUBITS-1)
		#print('num_actions',self.ACTION_SPACE_SIZE)


		#print(training_set_size)
		self.seed = seed
		print('SEED',seed, self.seed)
		# training_set_type = 'target_set_size'
		# if training_set_type == 'subset':
		# 	self.training_set = self.generate_subset()
		# 	self.test_set = [self.training_set[0]]
		# elif training_set_type == 'target_set_size':
		# 	self.training_set, self.test_set = self.generate_data_set_smaller_target_set()
		# else:
		# 	self.state_samples = self.generate_data_set()
		# 	training_set_size = int(len(self.state_samples) * 8 / 10)
		# 	self.training_set = self.state_samples[:training_set_size]
		# 	self.test_set = self.state_samples[training_set_size:]


		# self.test_selection = [list(self.seed.choice(self.s_list, size=abs(self.SIZE)))]
		# for elem in self.test_selection:
		# 	for entry in elem:
		# 		entry['g'] = entry['g'][:int(self.N_QUBITS/2.)]+'I'*(self.N_QUBITS-int(self.N_QUBITS/2.))
		# 		print('entry',entry['g'])


		self.config_idx = struct_dict["idx"]



	def reset(self, random_init: bool, seed: int, pad_idx: int) -> List[Union[int, float]]:
		'''
		Reset environment. Can be used to select new target gate set if random_init==True.

		Args:
			random_init: Boolean to either select a new target gate set or not.
			seed: seed index to selecting a new the target gate set from the full target set.
			pad_idx: padding index. If -1, no padding is done.

		Returns:
			state: initial state for target set

		'''

		if random_init:
			self.s_list_0 = list(self.seed.choice(self.s_list, size=abs(self.SIZE)))
			self.s_list_0 = hsc_utils.pad(self.s_list_0, pad_idx)
			print(self.s_list_0)
		else:
			#print(self.s_list)
			#print(self.s_list_0)
			self.source = S_obj(self.s_list_0, self.u_list,
								self.t_list, self.N_QUBITS)

		self.source = S_obj(self.s_list_0, self.u_list,
							self.t_list, self.N_QUBITS)


		self.episode_step = 0

		# Count number of times action is selected
		self.action_buffer = {act: 0 for act in range(self.ACTION_SPACE_SIZE)}

		# Action buffer to include in state
		self.recent_actions = deque([-1]*self.STATE_MEM_SIZE, maxlen=self.STATE_MEM_SIZE)

		state = self.state_encoding(self.source.s_tup, self.recent_actions)

		self.action_list = []
		self.action_markers = []

		# Maximum reward agent can receive
		#print([dict["g"] for dict in self.source.t_list])
		#print([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		self.MAX_REWARD = len([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		#print('MAAAX REWARD',self.MAX_REWARD)

		# If max. episode steps is 0, then use naive solution to get a value for it.
		if self.MAX_EPISODE_STEPS == 0:
			self.MAX_EPISODE_STEPS = 2*hsc_utils.mapping_gate_count(self.s_list_0, self.N_QUBITS)

		if self.STRATEGY == "ddqn":
			self.reward_funcs = hsc_rewards.RewardFuncs(n_qubits=self.N_QUBITS,
														max_episode_steps=self.MAX_EPISODE_STEPS,
														max_action_buffer=self.MAX_ACTION_BUFFER,
														max_reward=self.MAX_REWARD)

		elif self.STRATEGY == "mcts":
			self.reward_funcs = hsc_rewards.RolloutFuncs(n_qubits=self.N_QUBITS, action_space_size=self.ACTION_SPACE_SIZE)

		return state

	def random_start_reset(self, training: bool, max_num_states: int, pad_idx: int) -> List[Union[int, float]]:
		'''
		Reset environment. Can be used to select new target gate set if random_init==True.

		Args:
			random_init: Boolean to either select a new target gate set or not.
			seed: seed index to selecting a new the target gate set from the full target set.
			pad_idx: padding index. If -1, no padding is done.

		Returns:
			state: initial state for target set

		'''

		if training:
			state_num = self.seed.randint(0,high=max_num_states)
			#print(state_num)
			self.s_list_0 = np.load("../DDQN/RESULTS/" + self.CFGIDX + "/START_STATES/state_" + str(state_num) + ".npy",
								allow_pickle=True)
			#print(self.s_list_0)

			self.source = S_obj(self.s_list_0, self.u_list,
							self.t_list, self.N_QUBITS)

			#num_sample = self.seed.choice(len(self.training_set))
			#self.s_list_0 = self.training_set[num_sample]
			#print(self.s_list_0)
		else:
			#print('num', num)
			self.s_list_0 = self.test_set[max_num_states]


		self.source = S_obj(self.s_list_0, self.u_list,
							self.t_list, self.N_QUBITS)
		#print(self.source.s_tup)

		#print(self.source.s_tup)

		self.episode_step = 0

		# Count number of times action is selected
		self.action_buffer = {act: 0 for act in range(self.ACTION_SPACE_SIZE)}

		# Action buffer to include in state
		self.recent_actions = deque([-1]*self.STATE_MEM_SIZE, maxlen=self.STATE_MEM_SIZE)

		state = self.state_encoding(self.source.s_tup, self.recent_actions)

		self.action_list = []
		self.action_markers = []

		# Maximum reward agent can receive
		#print([dict["g"] for dict in self.source.t_list])
		#print([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		self.MAX_REWARD = len([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		#print('MAAAX REWARD',self.MAX_REWARD)

		# If max. episode steps is 0, then use naive solution to get a value for it.
		if self.MAX_EPISODE_STEPS == 0:
			self.MAX_EPISODE_STEPS = 2*hsc_utils.mapping_gate_count(self.s_list_0, self.N_QUBITS)

		if self.STRATEGY == "ddqn":
			self.reward_funcs = hsc_rewards.RewardFuncs(n_qubits=self.N_QUBITS,
														max_episode_steps=self.MAX_EPISODE_STEPS,
														max_action_buffer=self.MAX_ACTION_BUFFER,
														max_reward=self.MAX_REWARD)

		elif self.STRATEGY == "mcts":
			self.reward_funcs = hsc_rewards.RolloutFuncs(n_qubits=self.N_QUBITS, action_space_size=self.ACTION_SPACE_SIZE)

		return state

	def load_state_reset(self, counter: int, CFGIDX = "3q_4t_test"):
		'''
		Reset environment. Can be used to select new target gate set if random_init==True.

		Args:
			random_init: Boolean to either select a new target gate set or not.
			seed: seed index to selecting a new the target gate set from the full target set.
			pad_idx: padding index. If -1, no padding is done.

		Returns:
			state: initial state for target set

		'''
		#print(counter)
		self.s_list_0 = np.load("../DDQN/RESULTS/"+self.CFGIDX+"/START_STATES/state_"+str(counter)+".npy", allow_pickle=True)
		#print(self.s_list_0)

		self.source = S_obj(self.s_list_0, self.u_list,
							self.t_list, self.N_QUBITS)

		#print(self.source.s_tup)

		self.episode_step = 0

		# Count number of times action is selected
		self.action_buffer = {act: 0 for act in range(self.ACTION_SPACE_SIZE)}

		# Action buffer to include in state
		self.recent_actions = deque([-1]*self.STATE_MEM_SIZE, maxlen=self.STATE_MEM_SIZE)

		state = self.state_encoding(self.source.s_tup, self.recent_actions)

		self.action_list = []
		self.action_markers = []

		# Maximum reward agent can receive
		#print([dict["g"] for dict in self.source.t_list])
		#print([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		self.MAX_REWARD = len([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		#print('MAAAX REWARD',self.MAX_REWARD)

		# If max. episode steps is 0, then use naive solution to get a value for it.
		if self.MAX_EPISODE_STEPS == 0:
			self.MAX_EPISODE_STEPS = 2*hsc_utils.mapping_gate_count(self.s_list_0, self.N_QUBITS)

		if self.STRATEGY == "ddqn":
			self.reward_funcs = hsc_rewards.RewardFuncs(n_qubits=self.N_QUBITS,
														max_episode_steps=self.MAX_EPISODE_STEPS,
														max_action_buffer=self.MAX_ACTION_BUFFER,
														max_reward=self.MAX_REWARD)

		elif self.STRATEGY == "mcts":
			self.reward_funcs = hsc_rewards.RolloutFuncs(n_qubits=self.N_QUBITS, action_space_size=self.ACTION_SPACE_SIZE)

		return state

	def specific_state_reset(self, num: int):
		'''
		Reset environment. Can be used to select new target gate set if random_init==True.

		Args:
			random_init: Boolean to either select a new target gate set or not.
			seed: seed index to selecting a new the target gate set from the full target set.
			pad_idx: padding index. If -1, no padding is done.

		Returns:
			state: initial state for target set

		'''


		self.s_list_0 = self.test_selection[num]


		self.source = S_obj(self.s_list_0, self.u_list,
							self.t_list, self.N_QUBITS)

		#print(self.source.s_tup)

		self.episode_step = 0

		# Count number of times action is selected
		self.action_buffer = {act: 0 for act in range(self.ACTION_SPACE_SIZE)}

		# Action buffer to include in state
		self.recent_actions = deque([-1]*self.STATE_MEM_SIZE, maxlen=self.STATE_MEM_SIZE)

		state = self.state_encoding(self.source.s_tup, self.recent_actions)

		self.action_list = []
		self.action_markers = []

		# Maximum reward agent can receive
		#print([dict["g"] for dict in self.source.t_list])
		#print([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		self.MAX_REWARD = len([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		#print('MAAAX REWARD',self.MAX_REWARD)

		# If max. episode steps is 0, then use naive solution to get a value for it.
		if self.MAX_EPISODE_STEPS == 0:
			self.MAX_EPISODE_STEPS = 2*hsc_utils.mapping_gate_count(self.s_list_0, self.N_QUBITS)

		if self.STRATEGY == "ddqn":
			self.reward_funcs = hsc_rewards.RewardFuncs(n_qubits=self.N_QUBITS,
														max_episode_steps=self.MAX_EPISODE_STEPS,
														max_action_buffer=self.MAX_ACTION_BUFFER,
														max_reward=self.MAX_REWARD)

		elif self.STRATEGY == "mcts":
			self.reward_funcs = hsc_rewards.RolloutFuncs(n_qubits=self.N_QUBITS, action_space_size=self.ACTION_SPACE_SIZE)

		return state

	def test_sample_reset(self):
		'''
		Reset environment.

		Returns:
			state: truely random initial state for target set

		'''
		print(self.s_list)

		#for element in it.permutations(self.s_list, 2):
			#print(element)
		self.s_list_0 = list(random.choices(self.s_list, k=abs(self.SIZE)))
		print(self.s_list_0)
		self.source = S_obj(self.s_list_0, self.u_list,
							self.t_list, self.N_QUBITS)


		self.episode_step = 0

		# Count number of times action is selected
		self.action_buffer = {act: 0 for act in range(self.ACTION_SPACE_SIZE)}

		# Action buffer to include in state
		self.recent_actions = deque([-1]*self.STATE_MEM_SIZE, maxlen=self.STATE_MEM_SIZE)

		state = self.state_encoding(self.source.s_tup, self.recent_actions)

		self.action_list = []
		self.action_markers = []

		# Maximum reward agent can receive
		#print([dict["g"] for dict in self.source.t_list])
		#print([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		self.MAX_REWARD = len([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		#print('MAAAX REWARD',self.MAX_REWARD)

		# If max. episode steps is 0, then use naive solution to get a value for it.
		if self.MAX_EPISODE_STEPS == 0:
			self.MAX_EPISODE_STEPS = 2*hsc_utils.mapping_gate_count(self.s_list_0, self.N_QUBITS)

		if self.STRATEGY == "ddqn":
			self.reward_funcs = hsc_rewards.RewardFuncs(n_qubits=self.N_QUBITS,
														max_episode_steps=self.MAX_EPISODE_STEPS,
														max_action_buffer=self.MAX_ACTION_BUFFER,
														max_reward=self.MAX_REWARD)

		elif self.STRATEGY == "mcts":
			self.reward_funcs = hsc_rewards.RolloutFuncs(n_qubits=self.N_QUBITS, action_space_size=self.ACTION_SPACE_SIZE)

		return state

	def test_target_reset(self, idx, random_init: bool, seed: int, pad_idx: int) -> List[Union[int, float]]:
		'''
		Reset environment. Can be used to select new target gate set if random_init==True.

		Args:
			random_init: Boolean to either select a new target gate set or not.
			seed: seed index to selecting a new the target gate set from the full target set.
			pad_idx: padding index. If -1, no padding is done.

		Returns:
			state: initial state for target set

		'''


		self.source = S_obj(self.s_list_0, self.u_list,
							self.t_list, self.N_QUBITS)

		#for i in range(test_idx):
		#	self.source.action(action_sequence[i])
		if idx == 0:
			new_element = ({'c': (1 + 0j), 'g': 'ZXX'},)
			self.source.s_tup = self.source.s_tup + new_element
			#print('action_sequence', 'starting state', self.source.s_tup)

		if idx == 1:
			new_element = ({'c': (1 + 0j), 'g': 'XXXZ'},)
			self.source.s_tup = self.source.s_tup + new_element

		#print(self.source.s_tup)


		#print(self.source.s_tup, 7)
		#self.source.action(7)
		#print(self.source.s_tup, 1)
		#self.source.action(1)
		#self.source.action(6)


		self.episode_step = 0

		# Count number of times action is selected
		self.action_buffer = {act: 0 for act in range(self.ACTION_SPACE_SIZE)}

		# Action buffer to include in state
		self.recent_actions = deque([-1]*self.STATE_MEM_SIZE, maxlen=self.STATE_MEM_SIZE)

		state = self.state_encoding(self.source.s_tup, self.recent_actions)

		self.action_list = []
		self.action_markers = []

		# Maximum reward agent can receive
		#print([dict["g"] for dict in self.source.t_list])
		#print([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		self.MAX_REWARD = len([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		#print('MAAAX REWARD',self.MAX_REWARD)

		# If max. episode steps is 0, then use naive solution to get a value for it.
		if self.MAX_EPISODE_STEPS == 0:
			self.MAX_EPISODE_STEPS = 2*hsc_utils.mapping_gate_count(self.s_list_0, self.N_QUBITS)

		if self.STRATEGY == "ddqn":
			self.reward_funcs = hsc_rewards.RewardFuncs(n_qubits=self.N_QUBITS,
														max_episode_steps=self.MAX_EPISODE_STEPS,
														max_action_buffer=self.MAX_ACTION_BUFFER,
														max_reward=self.MAX_REWARD)

		elif self.STRATEGY == "mcts":
			self.reward_funcs = hsc_rewards.RolloutFuncs(n_qubits=self.N_QUBITS, action_space_size=self.ACTION_SPACE_SIZE)

		return state

	def test_dimension_reset(self, idx, random_init: bool, seed: int, pad_idx: int) -> List[Union[int, float]]:
		'''
		Reset environment. Can be used to select new target gate set if random_init==True.

		Args:
			random_init: Boolean to either select a new target gate set or not.
			seed: seed index to selecting a new the target gate set from the full target set.
			pad_idx: padding index. If -1, no padding is done.

		Returns:
			state: initial state for target set

		'''


		self.source = S_obj(self.s_list_0, self.u_list,
							self.t_list, self.N_QUBITS)

		#for i in range(test_idx):
		#	self.source.action(action_sequence[i])
		if idx == 0:
			for element in self.source.s_tup:
				if len(element['g']) == self.N_QUBITS:
					element['g'] = element['g'] + 'Y'
					self.source.s_tup = self.source.s_tup
			print('action_sequence', 'starting state', self.source.s_tup)

		elif idx == 1:
			for element in self.source.s_tup:
				element['g'] = 'XYY'
			self.source.s_tup = self.source.s_tup
			print('action_sequence', 'starting state', self.source.s_tup)



		#print(self.source.s_tup)


		#print(self.source.s_tup, 7)
		#self.source.action(7)
		#print(self.source.s_tup, 1)
		#self.source.action(1)
		#self.source.action(6)


		self.episode_step = 0

		# Count number of times action is selected
		self.action_buffer = {act: 0 for act in range(self.ACTION_SPACE_SIZE)}

		# Action buffer to include in state
		self.recent_actions = deque([-1]*self.STATE_MEM_SIZE, maxlen=self.STATE_MEM_SIZE)

		state = self.state_encoding(self.source.s_tup, self.recent_actions)

		self.action_list = []
		self.action_markers = []

		# Maximum reward agent can receive
		#print([dict["g"] for dict in self.source.t_list])
		#print([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		self.MAX_REWARD = len([s for s in self.source.s_tup if s["g"] not in [dict["g"] for dict in self.source.t_list]])
		#print('MAAAX REWARD',self.MAX_REWARD)

		# If max. episode steps is 0, then use naive solution to get a value for it.
		if self.MAX_EPISODE_STEPS == 0:
			self.MAX_EPISODE_STEPS = 2*hsc_utils.mapping_gate_count(self.s_list_0, self.N_QUBITS)

		if self.STRATEGY == "ddqn":
			self.reward_funcs = hsc_rewards.RewardFuncs(n_qubits=self.N_QUBITS,
														max_episode_steps=self.MAX_EPISODE_STEPS,
														max_action_buffer=self.MAX_ACTION_BUFFER,
														max_reward=self.MAX_REWARD)

		elif self.STRATEGY == "mcts":
			self.reward_funcs = hsc_rewards.RolloutFuncs(n_qubits=self.N_QUBITS, action_space_size=self.ACTION_SPACE_SIZE)

		return state


	def state(self,
			source: Tuple[Dict[str, Union[str, complex]]],
			recent_actions: Deque[int]) -> List[Union[int, float]]:
		'''
		Convert target gate set to state.

		Args:
			source: tuple to convert to state representation.
			recent_actions: action buffer of recently picked actions.

		Returns:
			state: state representation of source.

		'''


		if type(source) not in [tuple, np.ndarray, list]:
			source = (source,)

		conv_dict = {"I": 0,
					 "X": 1,
					 "Y": 3,
					 "Z": 2}

		s_tup_ops = [s["g"] for s in source]

		# Reshaping state in case source and destination models do not match up
		if (self.OLD_N_QUBITS != self.N_QUBITS) or (self.OLD_SIZE != self.SIZE):
			state = np.zeros(self.SIZE*self.N_QUBITS)

			for s in range(self.SIZE):
				for n in range(self.N_QUBITS):
					if n<self.OLD_N_QUBITS:
						state[self.OLD_N_QUBITS*s + n] = conv_dict[s_tup_ops[s][n]]/3
					else:
						state[self.OLD_N_QUBITS*(self.SIZE-1) + (self.N_QUBITS-self.OLD_N_QUBITS)*s + n] = conv_dict[s_tup_ops[s][n]]/3

			state = np.insert(state, self.OLD_SIZE*self.OLD_N_QUBITS,[a/self.ACTION_SPACE_SIZE for a in recent_actions])
			state = state.tolist()

		else:
			state = [conv_dict[p]/3 for s in s_tup_ops for p in s]
			state += [a/self.ACTION_SPACE_SIZE for a in recent_actions]


		return state

	def state_encoding(self,
			source: Tuple[Dict[str, Union[str, complex]]],
			recent_actions: Deque[int]) -> List[Union[int, float]]:
		'''
		Convert target gate set to state.

		Args:
			source: tuple to convert to state representation.
			recent_actions: action buffer of recently picked actions.

		Returns:
			state: state representation of source.

		'''


		if type(source) not in [tuple, np.ndarray, list]:
			source = (source,)

		conv_dict = {"I": [1,-1,-1,-1],
					 "X": [-1,1,-1,-1],
					 "Y": [-1,-1,-1,1],
					 "Z": [-1,-1,1,-1]}

		#conv_dict = {"I": [1, 0, 0, 0], "X": [0, 1, 0, 0],"Y": [0, 0, 0, 1],"Z": [0, 0, 1, 0]}

		s_tup_ops = [s["g"] for s in source]
		s_tup_ops.sort()

		# Reshaping state in case source and destination models do not match up
		if (self.OLD_N_QUBITS != self.N_QUBITS) or (self.OLD_SIZE != self.SIZE):
			state = np.zeros(self.SIZE*self.N_QUBITS)

			for s in range(self.SIZE):
				for n in range(self.N_QUBITS):
					if n<self.OLD_N_QUBITS:
						state[self.OLD_N_QUBITS*s + n] = conv_dict[s_tup_ops[s][n]]/3
					else:
						state[self.OLD_N_QUBITS*(self.SIZE-1) + (self.N_QUBITS-self.OLD_N_QUBITS)*s + n] = conv_dict[s_tup_ops[s][n]]/3

			state = np.insert(state, self.OLD_SIZE*self.OLD_N_QUBITS,[a/self.ACTION_SPACE_SIZE for a in recent_actions])
			state = state.tolist()

		else:
			state = []
			for s in s_tup_ops:
				for p in s:
					state += conv_dict[p]
		encoding_length = self.ENV_SPACE_SIZE
		if state != None:
			if len(state) != encoding_length:
				state = state + [-1]*(encoding_length-len(state))
		else:
			state = [0]*encoding_length
		#state = np.array(state, dtype = float)*(1/len(state))
			#print('state',state)
			#state += [a/self.ACTION_SPACE_SIZE for a in recent_actions]
		#print(state)
		return state

	def step(self, choice: int) -> Tuple[List[Union[int, float]], float, bool]:
		'''
		Take step.

		Args:
			choice: index of mapping gate.

		Returns:
			transition tuple including state, reward, and whether episode should end or not.
		'''

		if (choice >= self.ACTION_SPACE_SIZE):
			raise TypeError("Invalid action")

		old_check_list = self.source.check_list
		#print(choice)
		self.episode_step += 1
		self.action_buffer[choice] += 1

		if self.episode_step == 1:
			self.source.action(choice, True)

		if self.REWARD_METHOD in ["sparse", "step"]:
			old_reward = 0
		else:
			old_reward = len(self.source.check_list)
		prev_checklist_length = len(self.source.check_list)
		self.source.action(choice)
		self.recent_actions.append(choice)
		self.action_list.append(self.source.action_name(choice))
		#print(self.source.action_name(choice))
		current_checklist_length = len(self.source.check_list)
		if prev_checklist_length < current_checklist_length:
			check_list_expanded = True
		else:
			check_list_expanded = False


		#print('done condition',len(self.source.check_list),len(self.source.s_tup))
		done = (len(self.source.s_tup) == 0 or self.episode_step >= self.MAX_EPISODE_STEPS)
		#print('episode',self.episode_step,self.source.check_list)
		if done:
			#print(self.action_list)
			pass
			#print(self.source.check_list)

		self.set_reward_properties()
		diff_overlap = self.source.overlap - self.source.prev_overlap
		#print('diff overlap', diff_overlap)

		if self.STRATEGY=="ddqn" or (self.STRATEGY=="mcts" and self.REWARD_METHOD=="random"):
			reward = getattr(self.reward_funcs, self.REWARD_METHOD)(check_list_expanded, prev_checklist_length, diff_overlap)
		else:
			reward = None
		#print(self.source.s_tup)
		new_state = self.state_encoding(self.source.s_tup, self.recent_actions)
		self.old_action = choice
		#print(reward)
		if done:
			pass
			#print(reward)
			#print(self.source.s_tup)
			#print(self.source.check_list)

		return new_state, reward, int(done)

	def set_reward_properties(self):
		'''
		Set reward function 'private' variables
		(i.e. the properties)
		'''

		if self.STRATEGY=="ddqn":
			self.reward_funcs.episode_step = self.episode_step
			self.reward_funcs.action_list = self.action_list
			self.reward_funcs.action_buffer = self.action_buffer
			self.reward_funcs.action_markers = self.action_markers
			self.reward_funcs.check_list = self.source.check_list
			self.reward_funcs.s_tup = self.source.s_tup

		# Add settings for MCTS instead of querying environment all the time

	def generate_data_set(self):
		'''
		creates a training set for training with random starts
		'''
		num_samples = 10000
		samples = []
		unique_samples = []
		for _ in range(num_samples):
			sample = list(self.seed.choice(self.s_list, size=abs(self.SIZE)))
			sample_name = []
			for element in sample:
				sample_name.append(element['g'])
			sample_name.sort()
			sample_name = ''.join(sample_name)
			if sample_name not in unique_samples:
				samples.append(sample)
				unique_samples.append(sample_name)
		return samples

	def generate_data_set_smaller_target_set(self):
		'''
		creates a training set for training with random starts
		'''
		num_samples = 10000
		samples = []
		unique_samples = []
		for _ in range(num_samples):
			sample = list(self.seed.choice(self.s_list, size=abs(self.SIZE-1)))
			sample_name = []
			for element in sample:
				sample_name.append(element['g'])
			sample_name.sort()
			sample_name = ''.join(sample_name)
			if sample_name not in unique_samples:
				samples.append(sample)
				unique_samples.append(sample_name)

		num_test_samples = 1000
		test_samples = []
		for _ in range(num_test_samples):
			sample = list(self.seed.choice(self.s_list, size=abs(self.SIZE)))
			sample_name = []
			for element in sample:
				sample_name.append(element['g'])
			sample_name.sort()
			sample_name = ''.join(sample_name)
			if sample_name not in unique_samples:
				test_samples.append(sample)
				unique_samples.append(sample_name)
		return samples, test_samples

	def generate_subset(self):
		samples = []
		sample = list(self.seed.choice(self.s_list, size=abs(self.SIZE)))
		print(len(sample),sample)
		for _ in range(20):
			samples.append(sample)
		for length in range(len(sample)):
			for combi in it.combinations(sample, length):
				samples.append(list(combi))
		#samples.shuffel
		return samples




def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--par_idx', type=int, default=0, help='index for the parameter choice')
    parser.add_argument('--device', type=str, default='cpu', help='assigned device')
    args = parser.parse_args(argv)
    return args

# if __name__ == "__main__":
# 	args = get_args(sys.argv[1:])
#
# 	SIZE = 4  # Size of target gate set
# 	N_QUBITS = 5  # Number of qubits
# 	CFGIDX_LIST = ["536_100s"]  # List of parameter configurations
# 	AGENT_IDX = np.arange(3)  # List of agents
# 	SEED_IDX = [4]  # List of seed indices determining the initial target gate set
# 	LOAD_SEED_IDX = np.arange(4)  # List of set seed indices to load for TL
# 	LOAD_EP = [0]  # List of episodes to load for TL
# 	SOURCE_MODEL = [None]  # List of directories of model to load for TLca
#
# 	# Select configuration from the lists above
# 	par_idx = args.par_idx
# 	print(list(it.product(CFGIDX_LIST, SEED_IDX, AGENT_IDX, LOAD_SEED_IDX, LOAD_EP, SOURCE_MODEL)))
# 	par_list = list(it.product(CFGIDX_LIST, SEED_IDX, AGENT_IDX, LOAD_SEED_IDX, LOAD_EP, SOURCE_MODEL))[par_idx]
# 	STRUCT_DICT = hsc_utils.load_json(par_list[0])
#
# 	# Bool for PADding target gate set
# 	PAD = (STRUCT_DICT['pad'] != -1)
#     # ENVIRONMENT SPECIFIC
# 	structure_dict ={"idx": "534",
# 					"strategy": "ddqn",
# 					"device": -1,
# 					"layers": [10, 10, 10],
# 					"reward_fun": "incremental_norm",
# 					"act_select": "e_greedy",
# 					"tl_strategy": "tabula_rasa",
# 					"pad": -1,
# 					"per": 0,
# 					"eps_0": 0.1,
# 					"eps_min": 0.001,
# 					"gamma": 0.75,
# 					"decay": 0.999,
# 					"lr": 1e-5,
# 					"ute": 10,
# 					"episodes": 50000,
# 					"mb_size": 1,
# 					"r_size": 2000,
# 					"mr_size": 1000,
# 					"max_ep_steps": 100,
# 					"state_mem_size": 3}
#
# 	s_list, t_list, u_list, _ = hsc_utils.generate_ops(n_qubits=N_QUBITS, method="DDQN", pad=PAD)
# 	env = HSC_env(s_list, u_list, t_list, N_QUBITS, SIZE, STRUCT_DICT)
# 	print(t_list)
# 	#for seed idxnp.random.RandomState(SET_SEED_IDX)
# 	current_state = env.reset(random_init=True,
# 								   seed=par_list[1],
# 								   pad_idx= structure_dict["pad"])
# 	observation, reward, done = env.step(0)
# 	print(observation, reward, done)