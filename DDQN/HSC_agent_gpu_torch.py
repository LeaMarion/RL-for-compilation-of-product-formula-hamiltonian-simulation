


"""
/* Copyright (C) 2023 Eleanor Scerri (Elliescerri) modified by Lea Trenkwalder (LeaMarion) - All Rights Reserved
 * You may use, distribute and modify this code under the terms of the GNU GENERAL PUBLIC LICENSE v3 license.
 *
 * You should have received access to a copy of the GNU GENERAL PUBLIC LICENSE v3 license with
 * this file.
 */
"""
from copy import deepcopy
from typing import Tuple, Union, List, Dict, Mapping
from collections import deque
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import time
import inspect
import cirq
import numpy as np
import json
import random
import os
import sys
from datetime import datetime
import pathlib


import PER_utils as per_utils

sys.path.insert(1, "../shared")
import HSC_utils as hsc_utils
import HSC_env as hsc_env
sys.path.insert(1, "../SA")
import HSC_SA_utils as hsc_sa

from importlib import reload
reload(hsc_env)
reload(hsc_utils)
reload(per_utils)




class DQN_agent():
	'''
	Agent class. Any method which is not a standard RL function is commented

	'''

	def __init__(self,
				 s_list,
				 u_list,
				 t_list,
				 n_qubits,
				 size,
				 idx,
				 set_seed_idx,
				 agent_idx,
				 load_set_seed_idx,
				 load_ep,
				 load_model,
				 device):

		# Experiment Idx
		self.idx = idx

		# Directories for saving
		self.MODELS_DIR = "RESULTS/"+self.idx+"/MODELS"
		self.SEEDS_DIR = "RESULTS/"+self.idx+"/SEEDS"
		self.REWARDS_DIR = "RESULTS/"+self.idx+"/REWARDS"
		self.LOSSES_DIR = "RESULTS/"+self.idx+"/LOSSES"
		self.ACTIONS_DIR = "RESULTS/"+self.idx+"/ACTIONS"
		self.ACTION_MARKERS_DIR = "RESULTS/"+self.idx+"/ACTION_MARKERS"
		self.STEPS_DIR = "RESULTS/"+self.idx+"/STEPS"
		self.START_STATES = "RESULTS/" + self.idx + "/START_STATES"


		hsc_utils.check_folders(os.getcwd(), [self.MODELS_DIR,
											  self.SEEDS_DIR,
											  self.REWARDS_DIR,
											  self.LOSSES_DIR,
											  self.ACTIONS_DIR,
											  self.ACTION_MARKERS_DIR,
											  self.STEPS_DIR,
											  self.START_STATES])


		self.size = size
		self.n_qubits = n_qubits
		self.seed_idx = set_seed_idx
		self.agent_idx = agent_idx
		# fixed seed for testing
		#torch.manual_seed(1)

		# Parameter dictionary
		STRUCT_DICT = hsc_utils.load_json(idx)
		print(load_model)
		if load_model:
			load_model = load_model.format(load_set_seed_idx, load_ep)

		if torch.cuda.is_available() and device!='cpu':
			print("Using GPU")
			self.DEVICE = torch.device(device)
			sys.stdout.flush()
			self.dtype = torch.cuda.FloatTensor
			self.dtypelong = torch.cuda.LongTensor

		else:
			print("Using CPU")
			self.DEVICE = torch.device(device)
			self.dtype = torch.FloatTensor
			self.dtypelong = torch.LongTensor

		self.pad_idx = STRUCT_DICT["pad"]  # Padding index, i.e. where (if at all) identities are added



		self.DISCOUNT = STRUCT_DICT["gamma"]  # Discount
		self.epsilon_0 = STRUCT_DICT["eps_0"]  # Initial exploration parameter
		self.target_update_counter = 0  # Counter for when to update target net
		self.PER = (STRUCT_DICT["per"] == 1)  # Use prioritized experience or not
		self.SET_SEED_IDX = set_seed_idx  # Seed for selecting target gates

		# Create environment instance
		self.env = hsc_env.HSC_env(
			s_list, u_list, t_list, n_qubits, size, STRUCT_DICT, seed=np.random.RandomState(
												   self.SET_SEED_IDX))

		self.MIN_EPSILON = STRUCT_DICT["eps_min"]  # Min. exploration parameter
		self.LEARNING_RATE = STRUCT_DICT["lr"]  # Learning rate
		self.EPISODES = STRUCT_DICT["episodes"]  # Number of episodes
		self.MINIBATCH_SIZE = STRUCT_DICT["mb_size"]  # Minibatch size
		self.REPLAY_SIZE = STRUCT_DICT["r_size"]  # Memory size
		self.MIN_REPLAY_SIZE = STRUCT_DICT["mr_size"]  # Min. replay size when to start samping memory
		self.UPDATE_TARGET_EVERY = STRUCT_DICT["ute"]  # Threshold for target net update
		self.LAYERS = STRUCT_DICT["layers"]  # Layer structure for nets
		self.ACTION_SELECT = STRUCT_DICT["act_select"]  # Currently only epsilon greedy is propely implemented

		self.MAX_EPISODE_STEPS = self.env.MAX_EPISODE_STEPS  # Max. steps for each episode
		self.ENV_SPACE_SIZE = self.env.ENV_SPACE_SIZE  # Length of state
		#print(self.ENV_SPACE_SIZE)
		self.ACTION_SPACE_SIZE = self.env.ACTION_SPACE_SIZE  # Number of actions available
		#print('state space', self.ENV_SPACE_SIZE)
		#print('action space', self.ACTION_SPACE_SIZE)

		try:
			self.DECAY = STRUCT_DICT["decay"]  # Epsilon decay parameter
			print('decay',self.DECAY)
		except:
			self.DECAY = (self.MIN_EPSILON/self.epsilon_0)**(1/self.EPISODES)

		# In case we want to change parameters during training.
		# Previously used to shift padding index
		try:
			self.EPISODE_STAGGER = STRUCT_DICT["ep_stag"]
		except:
			self.EPISODE_STAGGER = self.EPISODES

		# Prioritized experience replay, not used anymore.
		if self.PER:
			print("Using PER")
			self.ALPHA = STRUCT_DICT["alpha"]
			self.BETA = STRUCT_DICT["beta"]
			self.BETA_INC = (1./self.BETA)**(1/self.EPISODES)
			self.PER_memory = per_utils.PER_memory(
				self.REPLAY_SIZE, self.ALPHA, self.BETA, self.BETA_INC, 1e-16)

		else:
			self.memory = ReplayBuffer(capacity=self.REPLAY_SIZE)  # Instantiate memory

		self.scores, self.loss, self.episodes, self.average = [], [], [], []

		# Set online model depending on whether we want to use tabula rasa (1), transfer learning by padding source
		# model (2), or transfer learnin by using already padded source model (3).

		# (1)
		if STRUCT_DICT["tl_strategy"] == "tabula_rasa":
			self.online_model = self.create_model(env_space_size=self.ENV_SPACE_SIZE,
												  action_space_size=self.ACTION_SPACE_SIZE).to(self.DEVICE)
			# for name, param in self.online_model.named_parameters():
			# 	if param.requires_grad:
			# 		print(name, param.data)
			self.optimizer = optim.Adam(
				self.online_model.parameters(), lr=self.LEARNING_RATE)
			print("Created new model")

		# (2)
		elif STRUCT_DICT["tl_strategy"] == "pad_destination_model":
			self.online_model = self.create_model(env_space_size=self.env.OLD_SIZE*self.env.OLD_N_QUBITS+self.env.STATE_MEM_SIZE,
												  action_space_size=self.env.N_SINGLE*self.env.OLD_N_QUBITS + self.env.N_DOUBLE*(self.env.OLD_N_QUBITS-1)).to(self.DEVICE)


			self.optimizer = optim.Adam(
				self.online_model.parameters(), lr=self.LEARNING_RATE)

			self.load(load_model)
			print("Loaded model {}".format(
				load_model.format(load_set_seed_idx, load_ep)))

			self.expand_model(input_addon=size*n_qubits - STRUCT_DICT["source_size"]*STRUCT_DICT["source_n_qubits"],
							  output_addon=self.env.N_SINGLE*(n_qubits-STRUCT_DICT["source_n_qubits"]) + self.env.N_DOUBLE*(n_qubits-STRUCT_DICT["source_n_qubits"]))

		# (3)
		elif STRUCT_DICT["tl_strategy"] == "pad_source_model":
			self.online_model = self.create_model(env_space_size=self.ENV_SPACE_SIZE,
												  action_space_size=self.ACTION_SPACE_SIZE).to(self.DEVICE)
			self.optimizer = optim.Adam(
				self.online_model.parameters(), lr=self.LEARNING_RATE)
			#print(load_model)
			self.load(load_model)
			print("Loaded padded model {}".format(
				load_model.format(load_set_seed_idx, load_ep)))

		# (4)
		elif STRUCT_DICT["tl_strategy"] == "load_model":
			self.online_model = self.create_model(env_space_size=self.ENV_SPACE_SIZE, action_space_size=self.ACTION_SPACE_SIZE).to(self.DEVICE)
			self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.LEARNING_RATE)
			model_path = load_model+".pt"
			self.load(model_path)
			print("Loaded model {}".format(load_model.format(load_set_seed_idx, load_ep)))
			self.online_model.eval()



		#print(self.online_model)

		self.target_model = deepcopy(self.online_model).to(self.DEVICE)  # Set source model

		# Filename
		if STRUCT_DICT["tl_strategy"] == "pad_destination_model" or STRUCT_DICT["tl_strategy"] == "pad_source_model":
			self.NAME = "HSC_DDQN_model__size_{}__cfgidx_{}__agent_{}__seed_{}_nqubits_{}_load_seed_{}__load_ep_{}_torch".format(
				size, idx, agent_idx, self.SET_SEED_IDX, n_qubits, load_set_seed_idx, load_ep)

		else:
			self.NAME = "HSC_DDQN_model__size_{}__cfgidx_{}__agent_{}__seed_{}_nqubits_{}_torch".format(
				size, idx, agent_idx, self.SET_SEED_IDX, n_qubits)

		self.step_counter = 0




	def create_model(self, env_space_size, action_space_size):
		'''
		Create online model based on layer structure, state size and number of actions available.

		Args:
			env_space_size: Length of state representing the set of target gates.
			action_space_size: number of actions avaiable.

		Returns:
			model: new model (used to create online model)

		'''

		layer_list = []

		layer_neurons_list = [env_space_size]+self.LAYERS

		for layer_neurons in zip(layer_neurons_list[:-1], layer_neurons_list[1:]):

			layer = nn.Linear(layer_neurons[0], layer_neurons[1])
			#add initialization

			layer_list.append(layer)
			layer_list.append(nn.ReLU())
			#nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))

			#nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')


		layer_list.append(nn.Linear(layer_neurons_list[-1], action_space_size))

		model = nn.Sequential(*layer_list)

		print("Total params: {}".format(sum(p.numel()
											for p in model.parameters())))


		return model



	def expand_model(self, input_addon, output_addon):
		'''
		Expands source online model in case we want to load a smaller model (ex. trained on fewer qubits).

		Args:
			input_addon: size difference between inputs of source and destination models.
			output_addon: size difference between outputs of source and destination models.

		'''

		old_weights = [self.online_model.state_dict(
		)[l].data for l in self.online_model.state_dict() if ("weight" in l)]
		old_biases = [self.online_model.state_dict(
		)[l].data for l in self.online_model.state_dict() if ("bias" in l)]

		wl_input = torch.zeros([old_weights[0].shape[0], input_addon])

		#nn.init.uniform_(wl_input,
		#				 a=-np.sqrt(input_addon+old_weights[0].shape[1]**-1),
		#				 b=np.sqrt(input_addon+old_weights[0].shape[1]**-1))

		nn.init.kaiming_uniform_(wl_input, a=0, mode='fan_in', nonlinearity='relu')

		wl_output = torch.zeros([output_addon, old_weights[-1].shape[1]])
		bl_output = torch.zeros([output_addon])

		#nn.init.uniform_(wl_output,
		#				 a=-np.sqrt(old_weights[-1].shape[1]**-1),
		#				 b=np.sqrt(old_weights[-1].shape[1]**-1))

		nn.init.kaiming_uniform_(wl_output, a=0, mode='fan_in', nonlinearity='relu')

		#nn.init.uniform_(bl_output,
		#				 a=-np.sqrt(old_biases[-1].shape[0]**-1),
		#				 b=np.sqrt(old_biases[-1].shape[0]**-1))

		#nn.init.kaiming_uniform_(bl_output, a=0, mode='fan_in', nonlinearity='relu')

		new_input_weights = torch.cat([old_weights[0], wl_input], dim=1)
		new_output_weights = torch.cat([old_weights[-1], wl_output], dim=0)
		new_output_biases = torch.cat([old_biases[-1], bl_output], dim=0)

		layers_list = list(self.online_model.children())

		linear_in = nn.Linear(
			new_input_weights.shape[1], new_input_weights.shape[0])
		linear_out = nn.Linear(
			new_output_weights.shape[1], new_output_weights.shape[0])

		# with torch.no_grad():
		linear_in.weight.data.copy_(new_input_weights)
		linear_out.weight.data.copy_(new_output_weights)
		linear_out.bias.data.copy_(new_output_biases)

		layers_list[0] = linear_in
		layers_list[-1] = linear_out

		self.online_model = nn.Sequential(*layers_list)

	def act(self, state):
		if self.ACTION_SELECT == "e_greedy":
			random_no = np.random.random()
			if random_no <= self.epsilon:
				action = random.randrange(self.ACTION_SPACE_SIZE)
				action = torch.tensor(action).to(self.DEVICE)
			else:
				state = torch.tensor(np.float32(state)).type(
					self.dtype).unsqueeze(0).to(self.DEVICE)
				q_value = self.online_model.forward(state)
				action = q_value.max(1)[1].data[0]

			return action

		if self.ACTION_SELECT == "deterministic":
			#self.online_model.eval()
			state = torch.tensor(np.float32(state)).type(
				self.dtype).unsqueeze(0).to(self.DEVICE)
			with torch.no_grad():
				q_value = self.online_model.forward(state)
			#if self.step_counter <= 3:
			#	print(q_value)
			action = q_value.max(1)[1].data[0]
			return action

		elif self.ACTION_SELECT == "softmax":
			raise NotImplementedError

	def act_test(self, state):
		state = torch.tensor(np.float32(state)).type(
			self.dtype).unsqueeze(0).to(self.DEVICE)
		with torch.no_grad():
			q_value = self.online_model.forward(state)
		action = q_value.max(1)[1].data[0]
		return action

	def update_target_model(self):
		self.target_model.load_state_dict(self.online_model.state_dict())

	def update_replay_memory(self, transition):
		'''
		transition = (state, action, reward, next_state, done) to memory.

		'''

		if self.PER:
			self.PER_memory.store(transition)

		else:
			self.memory.push(transition)

	def replay(self):

		if self.PER:
			if self.PER_memory.tree.n_entries < self.MIN_REPLAY_SIZE:
				return
			else:
				leaf_idxs, minibatch, weights = self.PER_memory.sample(
					self.MINIBATCH_SIZE)

				current_states, actions, rewards, next_current_states, done = zip(
					*minibatch)

		else:
			if len(self.memory) < self.MIN_REPLAY_SIZE:
				return
			else:
				current_states, actions, rewards, next_current_states, done = self.memory.sample(
					self.MINIBATCH_SIZE)
				weights = None

		current_states = torch.tensor(
			np.float32(current_states)).type(self.dtype).to(self.DEVICE)
		next_current_states = torch.tensor(
			np.float32(next_current_states)).type(self.dtype).to(self.DEVICE)
		actions = torch.tensor(actions).type(self.dtypelong).to(self.DEVICE)
		rewards = torch.tensor(rewards).type(self.dtype).to(self.DEVICE)
		done = torch.tensor(done).type(self.dtype).to(self.DEVICE)

		# do batch prediction
		current_qs = self.online_model(current_states)
		current_q = current_qs.gather(1, actions.unsqueeze(1)).squeeze(1)

		future_online_qs = self.online_model(next_current_states)
		future_target_qs = self.target_model(next_current_states)

		max_indicies = torch.max(future_online_qs, dim=1)[1]
		future_q = torch.gather(future_target_qs, 1, max_indicies.unsqueeze(1))
		expected_q = rewards + self.DISCOUNT * future_q.squeeze() * (1 - 1*done)

		loss = (current_q - expected_q.data).pow(2).mean()
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		if self.PER:
			batch_idxs = np.arange(self.MINIBATCH_SIZE, dtype=np.int32)
			abs_delta_qs = np.abs(
				(current_qs[batch_idxs, actions]-future_online_qs[batch_idxs, actions]).detach().numpy())
			self.PER_memory.batch_update(leaf_idxs, abs_delta_qs)

		return loss

	def save(self, name, ep):
		'''
		Save model.

		Args:
			name: name of model to save.
			ep: episode at which to save.

		'''

		torch.save({
			"episode": ep,
			"model_state_dict": self.online_model.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
		}, name)

	def load(self, name):
		'''
		Load model.

		Args:
			name: name of model to load

		'''

		if self.DEVICE == torch.device("cpu"):
			checkpoint = torch.load(name, map_location=torch.device("cpu"))
		else:
			checkpoint = torch.load(name)

		self.online_model.load_state_dict(checkpoint["model_state_dict"])
		self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		self.online_model = self.online_model.to(self.DEVICE)

	def train(self, with_test_episode=False):
		# get folder to save results from this cycle's exploration
		file_name = "HSC_DDQN_model_size_{}_agent_{}_seed_{}_nqubits_{}_torch".format(self.size, self.agent_idx, self.seed_idx, self.n_qubits)
		path_steps = self.STEPS_DIR+'/Time_steps_'+file_name+'.npy'
		path_test_steps = self.STEPS_DIR+'/Test_time_steps_'+file_name+'.npy'

		start_time = datetime.now()
		ep_times = []
		number_steps = []
		test_number_steps = []


		for ep in range(self.EPISODES):
			time_steps = 0
			episode_reward = 0
			done = 0
			ep_loss = []
			ep_start_time = time.time()

			# Conditional for when we want to change some parameters during training (ex. set of source operators)
			# If it's either the first episode or it's a 'staggered' episode, then reset environment with new target gate set.
			# THIS IS OBSOLETE AND CURRENTLY NOT IN USE.
			if not ep % self.EPISODE_STAGGER:
				print("Randomizing initial state")

				self.epsilon = self.epsilon_0

				if ep == 0:
					seeds = []
					win_actions = [[] for j in range(
						self.EPISODES//self.EPISODE_STAGGER)]
					action_markers = [[] for j in range(
						self.EPISODES//self.EPISODE_STAGGER)]
					first_solutions = - \
						np.ones(self.EPISODES//self.EPISODE_STAGGER)

				current_state = self.env.reset(random_init=True,
											   seed=np.random.RandomState(
												   self.SET_SEED_IDX),
											   pad_idx=self.pad_idx)

				seeds.append(self.env.source.s_tup)


			# Otherwise simply reset the current environment.
			else:
				#print("SEED IDX", self.SET_SEED_IDX)
				current_state = self.env.reset(random_init=False,
												   seed=np.random.RandomState(
													   self.SET_SEED_IDX),
												   pad_idx=self.pad_idx)
				#print(current_state)
			if ep == 0:
				np.save(self.START_STATES+'/state_'+str(self.SET_SEED_IDX), self.env.source.s_tup)



			# In case we want to set the number of steps based on naive solution.
			# Not the case for data in the manuscript.
			if self.MAX_EPISODE_STEPS == 0:
				self.MAX_EPISODE_STEPS = self.env.MAX_EPISODE_STEPS

			while not done:
				#print('CURRENT_STATE',current_state)
				action = self.act(current_state)
				self.step_counter = time_steps
				#if time_steps <= 10:
				#	print(int(action))
				time_steps += 1
				#print(action)
				next_state, reward, done = self.env.step(int(action))
				#if self.step_counter <= 3:
				#	print(next_state)
				#print(self.env.t_list)
				#print(next_state, reward, done)

				episode_reward += reward

				self.update_replay_memory(
					(current_state, action, reward, next_state, done))
				current_state = next_state
				#if ep > 4500:
				#	print(self.env.source.s_tup)

				if done:
					self.target_update_counter += 1

					if self.target_update_counter >= self.UPDATE_TARGET_EVERY:
						self.update_target_model()
						self.target_update_counter = 0

					self.scores.append(episode_reward)
					self.episodes.append(ep)
					self.average.append(np.mean(self.scores))
					number_steps.append(time_steps)

					np.save("{}/{}.npy".format(self.REWARDS_DIR,
											   self.NAME), self.scores)
					if (ep%100) == 0:
						print("episode: {}/{}, score: {}, eps: {:.3}, average: {:.3}".format(ep,
																						 self.EPISODES, episode_reward, self.epsilon, self.average[-1]))
					#print('steps', time_steps)
					sys.stdout.flush()
					if self.env.episode_step < self.MAX_EPISODE_STEPS:
						if first_solutions[ep // self.EPISODE_STAGGER] == -1:
							first_solutions[ep //
											self.EPISODE_STAGGER] = int(ep % self.EPISODE_STAGGER)

						win_actions[ep //
									self.EPISODE_STAGGER].append(self.env.action_list)

						action_markers[ep //
									   self.EPISODE_STAGGER].append(self.env.action_markers)


					if (ep%500) == 0:
						print('steps', time_steps)
						print("Saving trained model as {}_trained.pt".format(self.NAME))
						self.save("{}/{}_trained.pt".format(self.MODELS_DIR, self.NAME), ep)

						np.save("{}/{}_action.npy".format(self.ACTIONS_DIR, self.NAME),
									np.array(win_actions, dtype=object))
						np.save("{}/{}_action_markers.npy".format(self.ACTION_MARKERS_DIR, self.NAME),
									np.array(action_markers, dtype=object))
						np.save("{}/{}_first_solutions.npy".format(self.ACTIONS_DIR, self.NAME), first_solutions)

				loss_step = self.replay()
				if loss_step:
					ep_loss.append(float(loss_step.data))

			#print('Done!', time_steps)

			if len(ep_loss) > 0:
				if (ep % 100) == 0:
					print("Av loss: ", np.mean(ep_loss))
				self.loss.append(np.mean(ep_loss))
				np.save("{}/{}_loss.npy".format(self.LOSSES_DIR,
												self.NAME), self.loss)
			else:
				if (ep % 100) == 0:
					print("No loss data available")

			if self.epsilon > self.MIN_EPSILON:
				try:
					memory_size = len(self.memory)
				except:
					memory_size = self.PER_memory.tree.n_entries

				if memory_size >= self.MIN_REPLAY_SIZE:
					self.epsilon *= self.DECAY
					self.epsilon = max(self.MIN_EPSILON, self.epsilon)

			if self.PER:
				if self.PER_memory.tree.n_entries >= self.MIN_REPLAY_SIZE:
					self.BETA *= self.BETA_INC
					print("Beta annealed to {:.3f}".format(self.BETA))

			# Save model at certain (currently hardcoded) episodes.
			if ep+1 in [i*(10**j) for j in range(3) for i in [100, 500, 1000]]:
				print("Saving trained model as {}_{}.pt".format(self.NAME, ep+1))
				self.save("{}/{}_{}.pt".format(self.MODELS_DIR,
											   self.NAME, str(ep+1)), ep)
			total_time = datetime.now()-start_time
			ep_times.append(time.time()-ep_start_time)

			if with_test_episode:
				test_time_steps = self.test_episode()
				test_number_steps.append(test_time_steps)

			if (ep % 500) == 0:
				np.save(path_steps, number_steps)
				np.save(path_test_steps, test_number_steps)
				print("Time elapsed since training started: "+str(total_time))
				print("Average time per episode: {:.3f}".format(np.mean(ep_times)))
				print("Time taken", total_time)
				sys.stdout.flush()




		np.save(path_steps, number_steps)
		print('Step path',path_steps)
		np.save(path_test_steps, test_number_steps)
		print('Test steps path',path_test_steps)
		sys.stdout.flush()
		print("Time taken", total_time)
		self.save("{}/{}_trained.pt".format(self.MODELS_DIR, self.NAME), ep)
		np.save("{}/{}_action.npy".format(self.ACTIONS_DIR, self.NAME),
				np.array(win_actions, dtype=object))
		np.save("{}/{}_action_markers.npy".format(self.ACTION_MARKERS_DIR, self.NAME),
				np.array(action_markers, dtype=object))
		np.save("{}/{}_first_solutions.npy".format(self.ACTIONS_DIR, self.NAME), first_solutions)

	def test_episode(self):
		test_time_steps = 0
		episode_reward = 0
		done = 0
		ep_loss = []
		ep_start_time = time.time()
		current_state = self.env.reset(random_init=False,
									   seed=np.random.RandomState(
										   self.SET_SEED_IDX),
									   pad_idx=self.pad_idx)

		while not done:
			action = self.act_test(current_state)
			test_time_steps += 1
			next_state, reward, done = self.env.step(int(action))
			#print(int(action))
			current_state = next_state


		#print('test_steps', test_time_steps)
		return test_time_steps

	def non_det_random_start_train(self,with_specific_test_episodes=False, num_test_episodes = 1, max_num_states = 1, test_interval = 1000):
		# get folder to save results from this cycle's exploration
		file_name = "HSC_DDQN_size_{}_agent_{}_seed_{}_nqubits_{}".format(self.size, self.agent_idx, self.seed_idx, self.n_qubits)
		path_steps = self.STEPS_DIR+'/Time_steps_'+file_name+'.npy'
		path_test_steps = self.STEPS_DIR+'/Test_time_steps_'+file_name+'.npy'

		print('test_episodes',num_test_episodes)
		start_time = datetime.now()
		ep_times = []
		number_steps = []
		#test_number_steps = np.zeros((1, num_test_episodes), int)
		test_number_steps = dict()


		for ep in range(self.EPISODES):
			time_steps = 0
			episode_reward = 0
			done = 0
			ep_loss = []
			ep_start_time = time.time()

			# Conditional for when we want to change some parameters during training (ex. set of source operators)
			# If it's either the first episode or it's a 'staggered' episode, then reset environment with new target gate set.
			# THIS IS OBSOLETE AND CURRENTLY NOT IN USE.
			if not ep % self.EPISODE_STAGGER:
				print("Randomizing initial state")

				self.epsilon = self.epsilon_0

				if ep == 0:
					seeds = []
					win_actions = [[] for j in range(
						self.EPISODES//self.EPISODE_STAGGER)]
					action_markers = [[] for j in range(
						self.EPISODES//self.EPISODE_STAGGER)]
					first_solutions = - \
						np.ones(self.EPISODES//self.EPISODE_STAGGER)


				current_state = self.env.random_start_reset(training=True, max_num_states=max_num_states,
											   pad_idx=self.pad_idx)
				#print('initial state', current_state)

				seeds.append(self.env.source.s_tup)
				np.save("{}/{}_seed.npy".format(self.SEEDS_DIR, self.NAME),
						np.array(seeds, dtype=object))

			# Otherwise simply reset the current environment.
			else:
				#print(self.epsilon)
				current_state = self.env.random_start_reset(training=True,
											   max_num_states=max_num_states,
											   pad_idx=self.pad_idx)
				#print(current_state)




			# In case we want to set the number of steps based on naive solution.
			# Not the case for data in the manuscript.
			if self.MAX_EPISODE_STEPS == 0:
				self.MAX_EPISODE_STEPS = self.env.MAX_EPISODE_STEPS

			while not done:
				action = self.act(current_state)
				#if time_steps <= 1:
				#	print(int(action))
				time_steps += 1
				next_state, reward, done = self.env.step(int(action))
				#print(self.env.t_list)
				#print(next_state, reward, done)

				episode_reward += reward

				self.update_replay_memory(
					(current_state, action, reward, next_state, done))
				current_state = next_state
				#if ep > 4500:
				#	print(self.env.source.s_tup)

				if done:
					self.target_update_counter += 1

					if self.target_update_counter >= self.UPDATE_TARGET_EVERY:
						self.update_target_model()
						self.target_update_counter = 0

					self.scores.append(episode_reward)
					self.episodes.append(ep)
					self.average.append(np.mean(self.scores))
					number_steps.append(time_steps)

					np.save("{}/{}.npy".format(self.REWARDS_DIR,
											   self.NAME), self.scores)
					if (ep%100) == 0:
						print("episode: {}/{}, score: {}, eps: {:.3}, average: {:.3}".format(ep,
																						 self.EPISODES, episode_reward, self.epsilon, self.average[-1]))
					#print('steps', time_steps)
					sys.stdout.flush()
					if self.env.episode_step < self.MAX_EPISODE_STEPS:
						if first_solutions[ep // self.EPISODE_STAGGER] == -1:
							first_solutions[ep //
											self.EPISODE_STAGGER] = int(ep % self.EPISODE_STAGGER)

						win_actions[ep //
									self.EPISODE_STAGGER].append(self.env.action_list)

						action_markers[ep //
									   self.EPISODE_STAGGER].append(self.env.action_markers)


					if (ep%500) == 0:
						print("Saving trained model as {}_trained.pt".format(self.NAME))
						self.save("{}/{}_trained.pt".format(self.MODELS_DIR, self.NAME), ep)

						np.save("{}/{}_action.npy".format(self.ACTIONS_DIR, self.NAME),
									np.array(win_actions, dtype=object))
						np.save("{}/{}_action_markers.npy".format(self.ACTION_MARKERS_DIR, self.NAME),
									np.array(action_markers, dtype=object))
						np.save("{}/{}_first_solutions.npy".format(self.ACTIONS_DIR, self.NAME), first_solutions)

				loss_step = self.replay()
				if loss_step:
					ep_loss.append(float(loss_step.data))

			if len(ep_loss) > 0:
				if (ep % 100) == 0:
					print("Av loss: ", np.mean(ep_loss))
				self.loss.append(np.mean(ep_loss))
				np.save("{}/{}_loss.npy".format(self.LOSSES_DIR,
												self.NAME), self.loss)
			else:
				if (ep % 100) == 0:
					print("No loss data available")



			if self.PER:
				if self.PER_memory.tree.n_entries >= self.MIN_REPLAY_SIZE:
					self.BETA *= self.BETA_INC
					print("Beta annealed to {:.3f}".format(self.BETA))

			# Save model at certain (currently hardcoded) episodes.
			if ep+1 in [i*(10**j) for j in range(3) for i in [100, 500, 1000]]:
				print("Saving trained model as {}_{}.pt".format(self.NAME, ep+1))
				self.save("{}/{}_{}.pt".format(self.MODELS_DIR,
											   self.NAME, str(ep+1)), ep)
			total_time = datetime.now()-start_time
			ep_times.append(time.time()-ep_start_time)
			all_counts = np.array([])
			if ((ep+1)%test_interval)==0 and with_specific_test_episodes and ep>0:
				for counter in range(num_test_episodes):
					test_time_steps = self.non_det_specific_test_episode(counter)
					print(test_time_steps)
					all_counts = np.append(all_counts,test_time_steps)

				test_number_steps.update({ep:all_counts})
				print(test_number_steps)

			if self.epsilon > self.MIN_EPSILON:
				try:
					memory_size = len(self.memory)
				except:
					memory_size = self.PER_memory.tree.n_entries

				if memory_size >= self.MIN_REPLAY_SIZE:
					self.epsilon *= self.DECAY
					self.epsilon = max(self.MIN_EPSILON, self.epsilon)


			if ((ep+1) % test_interval) == 0:
				np.save(path_steps, number_steps)
				np.save(path_test_steps, test_number_steps)
				print("Time elapsed since training started: "+str(total_time))
				print("Average time per episode: {:.3f}".format(np.mean(ep_times)))
				print("Time taken", total_time)
				sys.stdout.flush()

		np.save(path_steps, number_steps)
		np.save(path_test_steps, test_number_steps)
		print("Time taken", total_time)
		self.save("{}/{}_trained.pt".format(self.MODELS_DIR, self.NAME), ep)
		np.save("{}/{}_action.npy".format(self.ACTIONS_DIR, self.NAME),
				np.array(win_actions, dtype=object))
		np.save("{}/{}_action_markers.npy".format(self.ACTION_MARKERS_DIR, self.NAME),
				np.array(action_markers, dtype=object))
		np.save("{}/{}_first_solutions.npy".format(self.ACTIONS_DIR, self.NAME), first_solutions)


	def random_start_train(self,with_specific_test_episodes=False, num_test_episodes = 1, max_num_states = 1):
		# get folder to save results from this cycle's exploration
		file_name = "HSC_DDQN_size_{}_agent_{}_seed_{}_nqubits_{}".format(self.size, self.agent_idx, self.seed_idx, self.n_qubits)
		path_steps = self.STEPS_DIR+'/Time_steps_'+file_name+'.npy'
		path_test_steps = self.STEPS_DIR+'/Test_time_steps_'+file_name+'.npy'


		start_time = datetime.now()
		ep_times = []
		number_steps = []
		test_number_steps = []


		for ep in range(self.EPISODES):
			time_steps = 0
			episode_reward = 0
			done = 0
			ep_loss = []
			ep_start_time = time.time()

			# Conditional for when we want to change some parameters during training (ex. set of source operators)
			# If it's either the first episode or it's a 'staggered' episode, then reset environment with new target gate set.
			# THIS IS OBSOLETE AND CURRENTLY NOT IN USE.
			if not ep % self.EPISODE_STAGGER:
				print("Randomizing initial state")

				self.epsilon = self.epsilon_0

				if ep == 0:
					seeds = []
					win_actions = [[] for j in range(
						self.EPISODES//self.EPISODE_STAGGER)]
					action_markers = [[] for j in range(
						self.EPISODES//self.EPISODE_STAGGER)]
					first_solutions = - \
						np.ones(self.EPISODES//self.EPISODE_STAGGER)


				current_state = self.env.random_start_reset(training=True, max_num_states=max_num_states,
											   pad_idx=self.pad_idx)
				#print('initial state', current_state)

				seeds.append(self.env.source.s_tup)
				np.save("{}/{}_seed.npy".format(self.SEEDS_DIR, self.NAME),
						np.array(seeds, dtype=object))

			# Otherwise simply reset the current environment.
			else:
				#print(self.epsilon)
				current_state = self.env.random_start_reset(training=True,
											   max_num_states=max_num_states,
											   pad_idx=self.pad_idx)
				#print(current_state)




			# In case we want to set the number of steps based on naive solution.
			# Not the case for data in the manuscript.
			if self.MAX_EPISODE_STEPS == 0:
				self.MAX_EPISODE_STEPS = self.env.MAX_EPISODE_STEPS

			while not done:
				action = self.act(current_state)
				#if time_steps <= 1:
				#	print(int(action))
				time_steps += 1
				next_state, reward, done = self.env.step(int(action))
				#print(self.env.t_list)
				#print(next_state, reward, done)

				episode_reward += reward

				self.update_replay_memory(
					(current_state, action, reward, next_state, done))
				current_state = next_state
				#if ep > 4500:
				#	print(self.env.source.s_tup)

				if done:
					self.target_update_counter += 1

					if self.target_update_counter >= self.UPDATE_TARGET_EVERY:
						self.update_target_model()
						self.target_update_counter = 0

					self.scores.append(episode_reward)
					self.episodes.append(ep)
					self.average.append(np.mean(self.scores))
					number_steps.append(time_steps)

					np.save("{}/{}.npy".format(self.REWARDS_DIR,
											   self.NAME), self.scores)
					if (ep%100) == 0:
						print("episode: {}/{}, score: {}, eps: {:.3}, average: {:.3}".format(ep,
																						 self.EPISODES, episode_reward, self.epsilon, self.average[-1]))
					#print('steps', time_steps)
					sys.stdout.flush()
					if self.env.episode_step < self.MAX_EPISODE_STEPS:
						if first_solutions[ep // self.EPISODE_STAGGER] == -1:
							first_solutions[ep //
											self.EPISODE_STAGGER] = int(ep % self.EPISODE_STAGGER)

						win_actions[ep //
									self.EPISODE_STAGGER].append(self.env.action_list)

						action_markers[ep //
									   self.EPISODE_STAGGER].append(self.env.action_markers)


					if (ep%500) == 0:
						print("Saving trained model as {}_trained.pt".format(self.NAME))
						self.save("{}/{}_trained.pt".format(self.MODELS_DIR, self.NAME), ep)

						np.save("{}/{}_action.npy".format(self.ACTIONS_DIR, self.NAME),
									np.array(win_actions, dtype=object))
						np.save("{}/{}_action_markers.npy".format(self.ACTION_MARKERS_DIR, self.NAME),
									np.array(action_markers, dtype=object))
						np.save("{}/{}_first_solutions.npy".format(self.ACTIONS_DIR, self.NAME), first_solutions)

				loss_step = self.replay()
				if loss_step:
					ep_loss.append(float(loss_step.data))

			if len(ep_loss) > 0:
				if (ep % 100) == 0:
					print("Av loss: ", np.mean(ep_loss))
				self.loss.append(np.mean(ep_loss))
				np.save("{}/{}_loss.npy".format(self.LOSSES_DIR,
												self.NAME), self.loss)
			else:
				if (ep % 100) == 0:
					print("No loss data available")

			if self.epsilon > self.MIN_EPSILON:
				try:
					memory_size = len(self.memory)
				except:
					memory_size = self.PER_memory.tree.n_entries

				if memory_size >= self.MIN_REPLAY_SIZE:
					self.epsilon *= self.DECAY
					self.epsilon = max(self.MIN_EPSILON, self.epsilon)

			if self.PER:
				if self.PER_memory.tree.n_entries >= self.MIN_REPLAY_SIZE:
					self.BETA *= self.BETA_INC
					print("Beta annealed to {:.3f}".format(self.BETA))

			# Save model at certain (currently hardcoded) episodes.
			if ep+1 in [i*(10**j) for j in range(3) for i in [100, 500, 1000]]:
				print("Saving trained model as {}_{}.pt".format(self.NAME, ep+1))
				self.save("{}/{}_{}.pt".format(self.MODELS_DIR,
											   self.NAME, str(ep+1)), ep)
			total_time = datetime.now()-start_time
			ep_times.append(time.time()-ep_start_time)
			test_interval = num_test_episodes
			if (ep%test_interval)==0 and with_specific_test_episodes:
				for counter in range(test_interval):
					test_time_steps = self.specific_test_episode(counter)
					test_number_steps.append(test_time_steps)
					#print('test_steps',test_time_steps)

			if (ep % 500) == 0:
				np.save(path_steps, number_steps)
				np.save(path_test_steps, test_number_steps)
				print("Time elapsed since training started: "+str(total_time))
				print("Average time per episode: {:.3f}".format(np.mean(ep_times)))
				print("Time taken", total_time)
				sys.stdout.flush()

		np.save(path_steps, number_steps)
		np.save(path_test_steps, test_number_steps)
		print("Time taken", total_time)
		self.save("{}/{}_trained.pt".format(self.MODELS_DIR, self.NAME), ep)
		np.save("{}/{}_action.npy".format(self.ACTIONS_DIR, self.NAME),
				np.array(win_actions, dtype=object))
		np.save("{}/{}_action_markers.npy".format(self.ACTION_MARKERS_DIR, self.NAME),
				np.array(action_markers, dtype=object))
		np.save("{}/{}_first_solutions.npy".format(self.ACTIONS_DIR, self.NAME), first_solutions)

	def specific_test_episode(self,counter):
		test_time_steps = 0
		episode_reward = 0
		done = 0
		ep_loss = []
		ep_start_time = time.time()
		current_state = self.env.load_state_reset(counter)
		#print('test_state',current_state)

		while not done:
			action = self.act_test(current_state)
			test_time_steps += 1
			next_state, reward, done = self.env.step(int(action))
			#print(int(action))
			current_state = next_state


		#print('test_steps', test_time_steps)
		return test_time_steps

	def non_det_specific_test_episode(self,counter):
		test_time_steps = 0
		episode_reward = 0
		done = 0
		ep_loss = []
		ep_start_time = time.time()
		current_state = self.env.load_state_reset(counter)
		#print('test_state',current_state)
		while not done:
			action = self.act(current_state)
			test_time_steps += 1
			next_state, reward, done = self.env.step(int(action))
			#print(int(action))
			current_state = next_state


		#print('test_steps', test_time_steps)
		return test_time_steps



	def test(self):
		# get folder to save results from this cycle's exploration
		file_name = "/HSC_DDQN_model_size_{}_agent_{}_seed_{}_nqubits_{}_torch".format(self.size, self.agent_idx, self.seed_idx, self.n_qubits)
		path_steps = self.STEPS_DIR+file_name+'.npy'

		start_time = datetime.now()
		ep_times = []
		number_steps = []
		sol_counter = 0

		for ep in range(self.EPISODES):

			time_steps = 0
			episode_reward = 0
			done = 0
			ep_loss = []
			ep_start_time = time.time()
			self.epsilon = self.epsilon_0

			# Conditional for when we want to change some parameters during training (ex. set of source operators)
			# If it's either the first episode or it's a 'staggered' episode, then reset environment with new target gate set.
			# THIS IS OBSOLETE AND CURRENTLY NOT IN USE.
			if ep == 0:
				print("Converged solution")
				seeds = []

				current_state = self.env.reset(random_init=True,
											   seed=np.random.RandomState(
												   self.SET_SEED_IDX),
											   pad_idx=self.pad_idx)

				seeds.append(self.env.source.s_tup)
				np.save("{}/{}_seed.npy".format(self.SEEDS_DIR, self.NAME),
						np.array(seeds, dtype=object))

			# Otherwise simply reset the current environment.
			else:
				current_state = self.env.test_sample_reset()



			while not done:
				action = self.act(current_state)
				time_steps += 1
				next_state, reward, done = self.env.step(int(action))
				episode_reward += reward
				current_state = next_state



				if done:

					self.scores.append(episode_reward)
					self.episodes.append(ep)
					self.average.append(np.mean(self.scores))
					number_steps.append(time_steps)

					np.save("{}/{}.npy".format(self.REWARDS_DIR,
											   self.NAME), self.scores)
					if (ep%100) == 0:
						print("episode: {}/{}, score: {}, average: {:.3}".format(ep,self.EPISODES, episode_reward, self.average[-1]))

					if time_steps < self.MAX_EPISODE_STEPS:
						sol_counter+= 1
						print(self.env.source.s_tup)
						print('steps', time_steps)
						sys.stdout.flush()
					if time_steps > 12:
						print(self.env.source.s_tup)
						print('steps', time_steps)
						sys.stdout.flush()


			total_time = datetime.now()-start_time
			ep_times.append(time.time()-ep_start_time)
			if (ep % 500) == 0:
				np.save(path_steps, number_steps)
				np.save(test_path_steps, number_steps)
				print("Time elapsed since training started: "+str(total_time))
				print("Average time per episode: {:.3f}".format(np.mean(ep_times)))
				print("Time taken", total_time)
				sys.stdout.flush()

		print('all found', sol_counter)

	def random_start_test(self, gatefactor = 1):
		# get folder to save results from this cycle's exploration
		file_name = "/HSC_DDQN_model_size_{}_agent_{}_seed_{}_nqubits_{}_torch".format(self.size, self.agent_idx, self.seed_idx, self.n_qubits)
		path_steps = self.STEPS_DIR+file_name+'.npy'

		start_time = datetime.now()
		ep_times = []
		number_steps = np.array([], dtype=float)
		naive_solution_length = np.array([])
		fail_counter = 0
		N_QUBITS = self.env.N_QUBITS
		s_list, t_list, u_list, action_ops = hsc_utils.generate_ops(
			n_qubits=N_QUBITS, method="SA")


		for ep in range(self.EPISODES):

			time_steps = 0
			episode_reward = 0
			done = 0
			ep_loss = []
			ep_start_time = time.time()
			self.epsilon = self.epsilon_0

			# Conditional for when we want to change some parameters during training (ex. set of source operators)
			# If it's either the first episode or it's a 'staggered' episode, then reset environment with new target gate set.
			# THIS IS OBSOLETE AND CURRENTLY NOT IN USE.

			current_state = self.env.random_start_reset(training=False,
											   num=ep,
											   pad_idx=self.pad_idx)

			N_QUBITS = 4

			s_tup = list(self.env.source.s_tup)
			actions = hsc_sa.mapping_gate(s_tup, action_ops, N_QUBITS, False, None)
			length = 0
			for action in actions:
				length += len(action)

			naive_solution_length = np.append(naive_solution_length,length*gatefactor)



			while not done:
				action = self.act(current_state)
				time_steps += 1
				next_state, reward, done = self.env.step(int(action))
				episode_reward += reward
				current_state = next_state



				if done:
					#print('action_list', self.env.action_list)
					self.scores.append(episode_reward)
					self.episodes.append(ep)
					self.average.append(np.mean(self.scores))
					if time_steps < self.MAX_EPISODE_STEPS:
						number_steps = np.append(number_steps,time_steps*gatefactor)
					else:
						fail_counter+=1

					np.save("{}/{}.npy".format(self.REWARDS_DIR,
											   self.NAME), self.scores)
					if (ep%100) == 0:
						print("episode: {}/{}, score: {}, average: {:.3}".format(ep,self.EPISODES, episode_reward, self.average[-1]))

					print('steps', time_steps)


			total_time = datetime.now()-start_time
			ep_times.append(time.time()-ep_start_time)
			if (ep % 500) == 0:
				np.save(path_steps, number_steps)
				print("Time elapsed since training started: "+str(total_time))
				print("Average time per episode: {:.3f}".format(np.mean(ep_times)))
				print("Time taken", total_time)
				sys.stdout.flush()

		np.save(path_steps, number_steps)
		print(len(number_steps), fail_counter, np.mean(number_steps), np.std(number_steps), np.mean(naive_solution_length),np.std(naive_solution_length))

	def specific_state_train(self, state_num):
		# get folder to save results from this cycle's exploration
		file_name = "/HSC_DDQN_model_size_{}_agent_{}_seed_{}_nqubits_{}_torch".format(self.size, self.agent_idx,
																					  self.seed_idx, self.n_qubits)

		path_steps = self.STEPS_DIR+file_name + '.npy'

		start_time = datetime.now()
		ep_times = []
		number_steps = []

		for ep in range(self.EPISODES):
			time_steps = 0
			episode_reward = 0
			done = 0
			ep_loss = []
			ep_start_time = time.time()

			# Conditional for when we want to change some parameters during training (ex. set of source operators)
			# If it's either the first episode or it's a 'staggered' episode, then reset environment with new target gate set.
			# THIS IS OBSOLETE AND CURRENTLY NOT IN USE.
			if not ep % self.EPISODE_STAGGER:
				print("Randomizing initial state")

				self.epsilon = self.epsilon_0

				if ep == 0:
					seeds = []
					win_actions = [[] for j in range(
						self.EPISODES // self.EPISODE_STAGGER)]
					action_markers = [[] for j in range(
						self.EPISODES // self.EPISODE_STAGGER)]
					first_solutions = - \
						np.ones(self.EPISODES // self.EPISODE_STAGGER)

				print('win_actions',win_actions)

				current_state = self.env.specific_state_reset(num=state_num)
				print('initial state', current_state)

				seeds.append(self.env.source.s_tup)
				np.save("{}/{}_seed.npy".format(self.SEEDS_DIR, self.NAME),
						np.array(seeds, dtype=object))

			# Otherwise simply reset the current environment.
			else:
				current_state = self.env.specific_state_reset(num = state_num)


			# In case we want to set the number of steps based on naive solution.
			# Not the case for data in the manuscript.
			if self.MAX_EPISODE_STEPS == 0:
				self.MAX_EPISODE_STEPS = self.env.MAX_EPISODE_STEPS

			while not done:
				action = self.act(current_state)
				time_steps += 1
				# print(action)
				next_state, reward, done = self.env.step(int(action))
				# print(self.env.t_list)
				# print(next_state, reward, done)

				episode_reward += reward

				self.update_replay_memory(
					(current_state, action, reward, next_state, done))
				current_state = next_state
				# if ep > 4500:
				#	print(self.env.source.s_tup)

				if done:
					self.target_update_counter += 1

					if self.target_update_counter >= self.UPDATE_TARGET_EVERY:
						self.update_target_model()
						self.target_update_counter = 0

					self.scores.append(episode_reward)
					self.episodes.append(ep)
					self.average.append(np.mean(self.scores))
					number_steps.append(time_steps)

					np.save("{}/{}.npy".format(self.REWARDS_DIR,
											   self.NAME), self.scores)
					if (ep % 100) == 0:
						print("episode: {}/{}, score: {}, eps: {:.3}, average: {:.3}".format(ep,
																							 self.EPISODES,
																							 episode_reward,
																							 self.epsilon,
																							 self.average[-1]))
					print('steps', time_steps)
					sys.stdout.flush()
					if self.env.episode_step < self.MAX_EPISODE_STEPS:
						if first_solutions[ep // self.EPISODE_STAGGER] == -1:
							first_solutions[ep //
											self.EPISODE_STAGGER] = int(ep % self.EPISODE_STAGGER)

						win_actions[ep //
									self.EPISODE_STAGGER].append(self.env.action_list)

						action_markers[ep //
									   self.EPISODE_STAGGER].append(self.env.action_markers)

					if (ep % 500) == 0:
						print("Saving trained model as {}_trained.pt".format(self.NAME))
						self.save("{}/{}_trained.pt".format(self.MODELS_DIR, self.NAME), ep)

						np.save("{}/{}_action.npy".format(self.ACTIONS_DIR, self.NAME),
								np.array(win_actions, dtype=object))
						np.save("{}/{}_action_markers.npy".format(self.ACTION_MARKERS_DIR, self.NAME),
								np.array(action_markers, dtype=object))
						np.save("{}/{}_first_solutions.npy".format(self.ACTIONS_DIR, self.NAME), first_solutions)

				loss_step = self.replay()
				if loss_step:
					ep_loss.append(float(loss_step.data))

			if len(ep_loss) > 0:
				if (ep % 100) == 0:
					print("Av loss: ", np.mean(ep_loss))
				self.loss.append(np.mean(ep_loss))
				np.save("{}/{}_loss.npy".format(self.LOSSES_DIR,
												self.NAME), self.loss)
			else:
				if (ep % 100) == 0:
					print("No loss data available")

			if self.epsilon > self.MIN_EPSILON:
				try:
					memory_size = len(self.memory)
				except:
					memory_size = self.PER_memory.tree.n_entries

				if memory_size >= self.MIN_REPLAY_SIZE:
					self.epsilon *= self.DECAY
					self.epsilon = max(self.MIN_EPSILON, self.epsilon)

			if self.PER:
				if self.PER_memory.tree.n_entries >= self.MIN_REPLAY_SIZE:
					self.BETA *= self.BETA_INC
					print("Beta annealed to {:.3f}".format(self.BETA))

			# Save model at certain (currently hardcoded) episodes.
			if ep + 1 in [i * (10 ** j) for j in range(3) for i in [100, 500, 1000]]:
				print("Saving trained model as {}_{}.pt".format(self.NAME, ep + 1))
				self.save("{}/{}_{}.pt".format(self.MODELS_DIR,
											   self.NAME, str(ep + 1)), ep)
			total_time = datetime.now() - start_time
			ep_times.append(time.time() - ep_start_time)
			if (ep % 500) == 0:
				np.save(path_steps, number_steps)
				print("Time elapsed since training started: " + str(total_time))
				print("Average time per episode: {:.3f}".format(np.mean(ep_times)))
				print("Time taken", total_time)
				sys.stdout.flush()

		np.save(path_steps, number_steps)
		print("Time taken", total_time)
		self.save("{}/{}_trained.pt".format(self.MODELS_DIR, self.NAME), ep)
		np.save("{}/{}_action.npy".format(self.ACTIONS_DIR, self.NAME),
				np.array(win_actions, dtype=object))
		np.save("{}/{}_action_markers.npy".format(self.ACTION_MARKERS_DIR, self.NAME),
				np.array(action_markers, dtype=object))
		np.save("{}/{}_first_solutions.npy".format(self.ACTIONS_DIR, self.NAME), first_solutions)


	def save_start_states(self):
		current_state = self.env.reset(random_init=False,
									   seed=np.random.RandomState(
										   self.SET_SEED_IDX),
									   pad_idx=self.pad_idx)
		print(current_state)
		np.save(self.START_STATES + '/state_' + str(self.SET_SEED_IDX), self.env.source.s_tup)




class ReplayBuffer(object):
	'''
	Memory class

	'''

	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)

	def push(self, transition):
		state, action, reward, next_state, done = transition
		state = np.expand_dims(state, 0)
		next_state = np.expand_dims(next_state, 0)

		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size):
		state, action, reward, next_state, done = zip(
			*random.sample(self.buffer, batch_size)
		)
		return np.concatenate(state), action, reward, np.concatenate(next_state), done

	def __len__(self):
		return len(self.buffer)
