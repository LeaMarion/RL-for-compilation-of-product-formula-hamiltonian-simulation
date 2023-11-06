"""
/* Copyright (C) 2023 Eleanor Scerri (Elliescerri) modified by Lea Trenkwalder (LeaMarion) - All Rights Reserved
 * You may use, distribute and modify this code under the terms of the GNU GENERAL PUBLIC LICENSE v3 license.
 *
 * You should have received access to a copy of the GNU GENERAL PUBLIC LICENSE v3 license with
 * this file.
 */
"""

import HSC_utils as hsc_utils

class RewardFuncs:

	def __init__(self, n_qubits, max_episode_steps, max_action_buffer, max_reward):
		self._episode_step = None
		self._action_list = None
		self._action_buffer = None
		self._action_markers = None
		self._s_tup = None
		self._check_list = None
		self.N_QUBITS = n_qubits
		self.MAX_EPISODE_STEPS = max_episode_steps
		self.MAX_ACTION_BUFFER = max_action_buffer
		self.MAX_REWARD = max_reward

	@property
	def episode_step(self):
		return self._episode_step

	@episode_step.setter
	def episode_step(self, step):
		self._episode_step = step

	@property
	def action_list(self):
		return self._action_list

	@action_list.setter
	def action_list(self, lst):
		self._action_list = lst

	@property
	def action_buffer(self):
		return self._action_buffer

	@action_buffer.setter
	def action_buffer(self, buffer):
		self._action_buffer = buffer

	@property
	def action_markers(self):
		return self._action_markers

	@action_markers.setter
	def action_markers(self, markers):
		self._action_markers = markers

	@property
	def check_list(self):
		return self._check_list

	@check_list.setter
	def check_list(self, lst):
		self._check_list = lst

	@property
	def s_tup(self):
		return self._s_tup

	@s_tup.setter
	def s_tup(self, tup):
		self._s_tup = tup

	def sparse(self, choice, old_reward):
		if (len(self._check_list) == len(self._s_tup)):
			reward = self.MAX_EPISODE_STEPS

		elif (action_buffer[choice] >= self.MAX_ACTION_BUFFER) and (self._episode_step > 1):
			reward = -1e-1

		else:
			reward = 0

		return reward

	def grid_reward_exp(self, check_list_expanded):
		if check_list_expanded:
			reward = 2**(len(self.check_list)-1)/2**(self.MAX_REWARD)
		else:
			reward = -0.0001
		return reward

	def grid_reward(self, check_list_expanded, prev_checklist_length, diff_overlap):
		if check_list_expanded:
			reward = len(self.check_list)-prev_checklist_length
		else:
			reward = -0.00001
		return reward


	def grid_overlap_reward(self, check_list_expanded, prev_checklist_length, diff_overlap):
		reward = diff_overlap*0.1
		if check_list_expanded:
			reward += len(self.check_list)-prev_checklist_length
		else:
			reward += -0.00001
		return reward

	def incremental(self, choice, old_reward):

		if (self._action_buffer[choice] >= self.MAX_ACTION_BUFFER) and (self._episode_step > 1):
			reward = -1e-1

		else:
			reward = len(self._check_list)-old_reward

		if (len(self._check_list) == len(self._s_tup)):
			reward += len(self._s_tup)/2
		return reward

	def incremental_norm(self, choice, old_reward):
		#print('action buffer', self._action_buffer, self.MAX_ACTION_BUFFER)
		if (self._action_buffer[choice] >= self.MAX_ACTION_BUFFER) and (self._episode_step > 1):
			reward = -1/(self.MAX_EPISODE_STEPS-self.MAX_ACTION_BUFFER+1)
		else:
			reward = (len(self._check_list)-old_reward)/self.MAX_REWARD

		if len(self._check_list) > old_reward:
			self._action_markers.append(self._episode_step)

		return reward

	def incremental_norm_tailreward(self, choice, old_reward):

		if self._action_buffer[choice] >= self.MAX_ACTION_BUFFER and self._episode_step > 1:
			reward = -1/(self.MAX_EPISODE_STEPS-self.MAX_ACTION_BUFFER+1)
		else:
			reward = (len(self._check_list)-old_reward)/self.MAX_REWARD

			if len(self._check_list) > old_reward:
				self._action_markers.append(self._episode_step)

				if (len(self._action_markers) > 1 and (len(self._check_list) == len(self._s_tup))):
					marked_actions = hsc_proc.mark_actions(self._action_list, self._action_markers)
					qubit_register = hsc_proc.matpr(marked_actions, self.N_QUBITS)
					_, tail_discount = hsc_proc.tail_simplify(qubit_register, self.N_QUBITS)

					reward += tail_discount/len(self._action_list)

		return reward

	def step(self, choice, old_reward):

		if self._action_buffer[choice] >= self.MAX_ACTION_BUFFER and self._episode_step > 1:
			reward = -.1

		else:
			reward = len(self._check_list)

		if (len(self._check_list) == len(self._s_tup)):
			reward += len(self._s_tup)/2

		return reward

	def mcts(self, choice, old_reward):

		reward = (len(self._check_list)-old_reward)#/self.MAX_REWARD

		return reward


class RolloutFuncs:

	def __init__(self, n_qubits, action_space_size):
		self._rollout_env = None
		self._transition = None
		self.N_QUBITS = n_qubits
		self.ACTION_SPACE_SIZE = action_space_size

	@property
	def rollout_env(self):
		return self._rollout_env

	@rollout_env.setter
	def rollout_env(self, env):
		self._rollout_env = env

	@property
	def transition(self):
		return self._transition

	@transition.setter
	def transition(self, trans):
		self._transition = trans

	@property
	def remaining_actions(self):
		return self._remaining_actions

	@remaining_actions.setter
	def remaining_actions(self, actions):
		self._remaining_actions = actions

	def naive(self, naive_len):
		naive_gates = hsc_utils.mapping_gate(
			list(self.rollout_env.source.s_tup), hsc_utils.TestOps(self.N_QUBITS), self.N_QUBITS, False, None)
		naive_gates_idxs = hsc_utils.actionToActionDQN(naive_gates)
		reward = max(0,(naive_len - len(naive_gates_idxs))/naive_len)

		return reward

	def random(self):
		while not self.transition[2]:
			action = np.random.choice(self.rollout_env.ACTION_SPACE_SIZE)
			self.transition = self.rollout_env.step(action)

		reward = self.transition[1]

		return reward

	def deltagreedy(self):
		# CHECK THIS, self._remaining_actions should always be the same for each node.
		reward = 0
		rollout_count = 1
		action_values = [self.N_QUBITS*len(self.rollout_env.source.s_tup)]

		while not self.transition[2]:
			env_copies = [deepcopy(self.rollout_env)]*len(self._remaining_actions)

			action_values = []

			for a in self._remaining_actions:
				env_copy = deepcopy(self.rollout_env)
				env_copy.step(a)
				action_reward = 0
				for s in env_copy.source.s_tup:
					action_reward += max([np.count_nonzero(np.array(Node.state(t))-np.array(Node.state(s)) == 0) for t in env_copy.source.t_list])

				action_values.append(action_reward)

			if action_values.count(max(action_values)) > 1:
				action = np.random.choice([a for a, v in enumerate(action_values) if v == max(action_values)])

			else:
				action = self._remaining_actions.index(np.argmax(action_values))

			self.transition = self.rollout_env.step(action)
			rollout_count+=1

		reward=max(action_values)/(self.N_QUBITS*len(self.rollout_env.source.s_tup))
		reward/=rollout_count

		return reward
