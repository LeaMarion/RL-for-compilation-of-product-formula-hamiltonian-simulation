"""
/* Copyright (C) 2023 Eleanor Scerri (Elliescerri) - All Rights Reserved
 * You may use, distribute and modify this code under the terms of the GNU GENERAL PUBLIC LICENSE v3 license.
 *
 * You should have received access to a copy of the GNU GENERAL PUBLIC LICENSE v3 license with
 * this file.
 */
"""

import numpy as np
import torch

class SumTree(object):

	data_pointer = 0

	def __init__(self, capacity):

		self.capacity = capacity

		#collection of nodes
		self.tree = np.zeros(2*capacity-1) #assuming even number of leaves

		#data stored in leaves
		self.experiences = np.zeros(capacity, dtype=object)

		self.n_entries = 0


	def update_leaf(self, leaf_idx, priority):

		tree_idx = leaf_idx

		while tree_idx!=0:
			tree_idx = (tree_idx-1)//2
			self.tree[tree_idx] += priority - self.tree[leaf_idx]

		self.tree[leaf_idx] = priority



	def add_leaf(self, priority, experience):

		leaf_idx = self.data_pointer + self.capacity - 1 #populate leaves, starting from self.capacity - 1 to 2*capacity-2

		self.experiences[self.data_pointer] = experience

		#update leaf
		self.update_leaf(leaf_idx, priority)

		# go to next entry
		self.data_pointer += 1

		# If we're above the capacity, we go back to first index to overwrite, unlike deque which keeps adding at the end 
		if self.data_pointer >= self.capacity:
			self.data_pointer = 0

		if self.n_entries < self.capacity:
			self.n_entries += 1

	def get_leaf(self, value):

		parent_idx = 0

		# the while loop is faster than the method in the reference code
		while True:
			left_child_idx = 2 * parent_idx + 1
			right_child_idx = left_child_idx + 1

			if left_child_idx >= len(self.tree):
				leaf_idx = parent_idx
				break

			else:
				if value <= self.tree[left_child_idx]:
					parent_idx = left_child_idx

				else:
					parent_idx = right_child_idx
					value -= self.tree[left_child_idx]

		data_idx = leaf_idx - self.capacity + 1

		return leaf_idx, self.tree[leaf_idx], self.experiences[data_idx]

	def npy_to_tensor(self):
		self.tree = torch.tensor(self.tree)

	@property
	def total_priority(self):
		return self.tree[0] # Returns the root node

class PER_memory(object):

	def __init__(self, capacity, alpha = .6, beta = .4, beta_inc = .001, tolerance = 1e-16):
		self.alpha = alpha
		self.beta = beta
		self.beta_inc = beta_inc
		self.tolerance = tolerance
		self.tree = SumTree(capacity)

	def store(self, experience):

		#Find max priority from leaves
		max_priority = np.max(self.tree.tree[-self.tree.capacity:])

		if max_priority == 0:
			max_priority = self.tolerance

		self.tree.add_leaf(max_priority, experience)

	def sample(self, batch_size):

		minibatch = []
		batch_idxs = []
		priorities = []

		priority_segment = self.tree.total_priority / batch_size

		self.beta = np.min([1., self.beta + self.beta_inc])

		for i in range(batch_size):

			value = np.random.uniform(i*priority_segment, (i+1)*priority_segment)

			batch_idx, priority, experience = self.tree.get_leaf(value)

			minibatch.append(list(experience))
			priorities.append(priority)
			batch_idxs.append(batch_idx)

		sampling_probabilities = priorities / self.tree.total_priority
		is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
		is_weight /= np.max(is_weight)

		return batch_idxs, minibatch, is_weight

	def batch_update(self, leaf_idxs, abs_delta_qs):

		abs_delta_qs += self.tolerance #add tolerance to guarantee non zero priority
		priorities = np.power(abs_delta_qs, self.alpha)

		for leaf_idx, priority in zip(leaf_idxs, priorities):
			self.tree.update_leaf(leaf_idx, priority)



