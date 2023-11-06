"""
/* Copyright (C) 2023 Eleanor Scerri (Elliescerri) - All Rights Reserved
 * You may use, distribute and modify this code under the terms of the GNU GENERAL PUBLIC LICENSE v3 license.
 *
 * You should have received access to a copy of the GNU GENERAL PUBLIC LICENSE v3 license with
 * this file.
 */
"""

import numpy as np
from copy import deepcopy
from typing import Tuple, Union, List, Dict, Mapping

def mark_actions(action_list: List[Mapping[str, Union[str, int]]],
				 action_markers: List[int]) -> List[List[Mapping[str, Union[str, int]]]]:

	'''
	Separates list of mapping gates based on when different target gates get mapped to source gates
	(given by action_markers)

	Args:
		action_list: list of mapping gates to be separated.
		action_markers: list of indices marking at which point target gates were mapped to
						source gates.

	Returns:
		actions_marked: action_list separated based on action_markers.

	'''

	action_markers_pad = [0]+action_markers

	if type(action_list) != list:
		action_list = action_list.tolist()

	actions_marked = [action_list[m[0]:m[1]] for m in zip(
		action_markers_pad[:-1], action_markers_pad[1:])]

	return actions_marked


def matr(actions_marked: List[List[Mapping[str, Union[str, int]]]], n_qubits: int) -> List[Dict]:
	'''
	(m)arked (a)ctions (t)o (r)egister.
	Transforms separated action list to a list of dictionaries, one for each qubit.
	The value for a dictionary is a list of operations in the order they appear for that qubit.
	Use only for tail simplifications.

	Args:
		actions_marked: list of separated mapping gates.
		n_qubits: number of qubits in total.

	Returns:
		qubit_reg: list of ordered gates and which qubits they act on.

	'''

	qubit_reg = []

	for sub_action_list in actions_marked:
		qubit_subreg = {q: [] for q in range(n_qubits)}
		for a_idx in range(len(sub_action_list)):
			action = sub_action_list[a_idx]

			# add gate to corresponding qubit in register
			if type(action['q']) == tuple:
				# if 2 qubit gate, add which other qubit is involved
				qubit_subreg[action['q'][1]].append([action['g'], action['q']])
			else:
				# add gate to qubit subregister
				qubit_subreg[action['q']].append(action['g'])

		qubit_reg.append(qubit_subreg)

	return qubit_reg


def matpr(actions_marked: List[List[Mapping[str, Union[str, int]]]], n_qubits: int) -> List[Dict]:
	'''
	(m)arked (a)ctions (t)o (p)added (r)egister. Transforms separated action list to a list of dictionaries, one for each qubit.
	The value for a dictionary is a list of operations in the order they appear for that qubit.
	Unlike matr function, the dictionary values are padded so as to keep order of the operations as they appear
	in the full circuit, similar to the INLINE insert startegy used in Cirq.

	Use for simplifications within each sublist of the separated action list.

	Args:
		actions_marked: list of separated mapping gates.
		n_qubits: number of qubits in total.

	Returns:
		qubit_reg: list of properly ordered gates and which qubits they act on.

	'''

	qubit_reg = []

	for sub_action_list in actions_marked:
		qubit_subreg = {q: ['I'] for q in range(n_qubits)}

		for a_idx in range(len(sub_action_list)):
			action = sub_action_list[a_idx]
			if type(action['q']) == tuple:
				control, target = action['q']
				if (qubit_subreg[control][-1]
					== qubit_subreg[target][-1]
						== 'I'):
					qubit_subreg[target][-1] = qubit_subreg[control][-1] = [
						action['g'], action['q']]

				else:
					qubit_subreg[control].append([action['g'], action['q']])
					qubit_subreg[target].append([action['g'], action['q']])

					for q_idx in np.r_[0:min(action['q']), min(action['q'])+1:max(action['q']), max(action['q'])+1:n_qubits]:
						qubit_subreg[q_idx].append('I')

			else:
				if qubit_subreg[action['q']][-1] == 'I':
					qubit_subreg[action['q']][-1] = action['g']

				else:
					qubit_subreg[action['q']].append(action['g'])

					for q_idx in np.r_[0:action['q'], action['q']+1:n_qubits]:
						qubit_subreg[q_idx].append('I')

		qubit_reg.append(qubit_subreg)

	return qubit_reg


def _next_op(register: List, r: bool) -> Union[str, List[Union[str, Tuple[int]]]]:
	'''
	Returns next or previous operator in action register.

	Args:
		register: list of gates applied on a particular qubit.
		r: Boolean. If true, then search previous gate instead of next.

	Returns:
		next, or previous, non trivial operator in the queue.

	'''

	if r:
		return next((op for op in reversed(register) if op != 'I'), None)
	else:
		return next((op for op in register if op != 'I'), None)


def _next_op_idx(register: List, r: bool) -> Union[str, List[Union[str, Tuple[int]]]]:
	'''
	Returns index of next or previous operator in action register.

	Args:
		register: list of gates applied on a particular qubit.
		r: Boolean. If true, then search previous gate instead of next.

	Returns:
		next, or previous, non trivial operator in the queue.

	'''

	if r:
		return next((j for j, op in reversed(list(enumerate(register))) if op != 'I'), None)
	else:
		return next((j for j, op in enumerate(register) if op != 'I'), None)


def tail_simplify(qubit_reg: List[Mapping], n_qubits: int) -> Tuple[List[Mapping], int]:
	'''
	Cancel mapping gates in the tail of the full sequence. See manuscript for full description.

	Args:
		qubit_reg: list of properly ordered gates and which qubits they act on.
		n_qubits: number of qubits in total.

	Returns:
		qubit_reg_redux: simplified action registers.
		cost_discount: number of gates cancelled.

	'''

	qubit_reg_redux = deepcopy(qubit_reg)
	cost_discount = 0

	for i in range(len(qubit_reg_redux)-1):
		done = False
		left_reg = qubit_reg_redux[i]
		right_reg = qubit_reg_redux[i+1]

		while not done:
			q_count = 0
			for q_idx in range(n_qubits):
				if left_reg[q_idx] == ['I']*len(left_reg[q_idx]) or right_reg[q_idx] == ['I']*len(right_reg[q_idx]):
					q_count += 1
					continue

				elif _next_op(left_reg[q_idx], r=True) == _next_op(right_reg[q_idx], r=False):
					if type(_next_op(left_reg[q_idx], r=True)) == str:
						left_reg[q_idx][_next_op_idx(
							left_reg[q_idx], r=True)] = 'I'
						right_reg[q_idx][_next_op_idx(
							right_reg[q_idx], r=False)] = 'I'
						cost_discount += 2
					else:
						control, target = left_reg[q_idx][_next_op_idx(
							left_reg[q_idx], r=True)][1]
						if (_next_op(left_reg[control], r=True) == _next_op(left_reg[target], r=True)
								and _next_op(right_reg[control], r=False) == _next_op(right_reg[target], r=False)):
							left_reg[control][_next_op_idx(
								left_reg[control], r=True)] = 'I'
							left_reg[target][_next_op_idx(
								left_reg[target], r=True)] = 'I'
							right_reg[control][_next_op_idx(
								right_reg[control], r=False)] = 'I'
							right_reg[target][_next_op_idx(
								right_reg[target], r=False)] = 'I'
							cost_discount += 2

						else:
							q_count += 1
				else:
					q_count += 1

			if q_count == n_qubits:
				done = True

	return qubit_reg_redux, cost_discount


def mid_simplify(qubit_reg: List[Mapping], n_qubits: int) -> Tuple[List[Mapping], int]:
	'''
	Cancel mapping gates in the mid part of the full sequence. See manuscript for full description.

	Args:
		qubit_reg: list of properly ordered gates and which qubits they act on.
		n_qubits: number of qubits in total.

	Returns:
		qubit_reg_redux: simplified action registers.
		cost_discount: number of gates cancelled.

	'''

	qubit_reg_redux = deepcopy(qubit_reg)
	cost_discount = 0

	for i in range(len(qubit_reg_redux)):
		done = False

		while not done:
			q_count = 0
			for q_idx in range(n_qubits):
				qubit_reg_redux_clean = list(
					filter(lambda op: op != 'I', qubit_reg_redux[i][q_idx]))
				tupled_reg = list(
					zip(qubit_reg_redux_clean[:-1], qubit_reg_redux_clean[1:]))

				if all([tup[0] != tup[1] for tup in tupled_reg]):
					q_count += 1

				else:
					for j in range(len(qubit_reg_redux[i][q_idx])):
						if qubit_reg_redux[i][q_idx][j] == _next_op(qubit_reg_redux[i][q_idx][j+1:], r=False) != 'I':
							next_op_idx = _next_op_idx(
								qubit_reg_redux[i][q_idx][j+1:], r=False)+j+1

							if type(qubit_reg_redux[i][q_idx][j]) == str:
								qubit_reg_redux[i][q_idx][j] = 'I'
								qubit_reg_redux[i][q_idx][next_op_idx] = 'I'
								cost_discount += 2

							else:
								control, target = qubit_reg_redux[i][q_idx][j][1]
								assert q_idx == control or q_idx == target

								qubit_reg_redux[i][control][j] = 'I'
								qubit_reg_redux[i][target][j] = 'I'
								qubit_reg_redux[i][control][next_op_idx] = 'I'
								qubit_reg_redux[i][target][next_op_idx] = 'I'
								cost_discount += 2

			if q_count == n_qubits:
				done = True

	return qubit_reg_redux, cost_discount


def action(op: Mapping[str, Union[complex, str]],
		   action: Mapping[str, Union[Union[int,Tuple[int]], str]],
		   n_qubits: int,
		   u_list_single: List,
		   u_list_double: List,
		   t_op_list: List) -> Dict[str, Union[complex, str]]:
	'''
	Applies mapping gate to target gate not yet mapped to a source gate and returns it.

	Args:
		op: target gate to transform
		action: action to apply to op
		n_qubits: number of qubits in total
		u_list_single: list of single qubit mapping gates
		u_list_double: list of two qubit mapping gates
		t_op_list: list of source gates

	Return:
		op: transformed operator (if it is not already a source gate)

	'''

	combs = [(q, q+1) for q in range(n_qubits-1)]

	if len(np.shape(action['q'])) == 0:
		if op['g'] not in [dict['g'] for dict in t_op_list]:
			try:
				op = next((o for o in u_list_single if o.__name__ ==
						   action['g']), None)(action['q'], op)
			except:
				print(action['g'], action['q'], len(np.shape(action['q'])))

	else:
		if op['g'] not in [dict['g'] for dict in t_op_list]:
			op = next((o for o in u_list_double if o.__name__ == action['g']), None)(*action['q'], op)

	return op


def action_name(action_no: int,
				n_qubits: int,
				u_list_single: List,
				u_list_double: List) -> Dict[str, Union[str, Union[int, Tuple[int]]]]:
	'''
	Creates the action dictionary given the index.

	Args:
		action_no: action index.
		n_qubits: number of qubits in total.
		u_list_single: list of single qubit mapping gates.
		u_list_double: list of double qubit mapping gates.

	Returns:
		action dictionary.

	'''

	combs = [(q, q+1) for q in range(n_qubits-1)]

	if action_no < len(u_list_single)*n_qubits:
		return {'g': u_list_single[action_no//n_qubits].__name__, 'q': action_no % n_qubits}

	else:
		return {'g': u_list_double[(action_no-len(u_list_single)*n_qubits)//len(combs)].__name__, 'q': combs[(action_no-len(u_list_single)*n_qubits) % len(combs)]}


def action_length(action_list: List,
				  n_qubits: int,
				  u_list_single: List,
				  u_list_double: List,
				  s_op_list: List,
				  t_op_list: List,
				  markers: List = None,
				  check: bool = False,
				  simplify_tail: bool = False,
				  simplify_mid: bool = False):

	'''
	Calculate total number of mapping gates used. Allows for tail and mid simplification.

	Args:
		action_list: list of mapping gates to be counted.
		n_qubits: number of qubits in total.
		u_list_single: list of single qubit mapping gates.
		u_list_double: list of double qubit mapping gates.
		s_op_list: list of target gates.
		t_op_list: list of source gates.
		markers: action markers to separate the actions.
		check: check whether the list of mapping gates maps all target to source gates.
				If True, action markers are also created.
		simplify_tail: whether to apply tail simplifications or not.
		simplify_mid: whether to apply mid simplifications or not.

	Returns:
		action_markers: action markers.
		count: mapping gate count.
		tail_discount: number of gates cancelled from the end.
		mid_discount: number of gates cancelled from the middle.

	'''

	current_buffer = 0
	count = 0
	tail_discount = 0
	mid_discount = 0
	done = False
	s_op_list = [op for op in s_op_list if op['g']
				 not in [t['g'] for t in t_op_list]]

	op_dict = {j: [] for j in range(n_qubits)}
	_current_tuple = (-1, -1)

	if check:
		action_markers = []
		for a in range(len(action_list)):

			#print('source',s_op_list)
			#print('target',t_op_list)
			if action_list[a]['g'] == 'I':
				continue
			else:
				count += 1
				#print(action_list[a])
				new_s_op_list = [action(op, action_list[a], n_qubits, u_list_single, u_list_double, t_op_list) for op in s_op_list]
				new_s_op_list = [op for op in new_s_op_list if op['g']
								 not in [t['g'] for t in t_op_list]]
				#print('after',new_s_op_list)

				if len(new_s_op_list) < len(s_op_list):
					action_markers.append(a+1)

				if len(new_s_op_list) == 0:
					done = True
					break
				else:
					s_op_list = new_s_op_list

		count = 2*count

	else:
		done = True
		action_markers = markers
		#print('marks',action_markers)
		#print([a for a in action_list if a['g']!='I'])
		count = 2*len([a for a in action_list if a['g']!='I'])
		#print('count',count)

	if simplify_tail:
		marked_actions = mark_actions(action_list, action_markers)
		#print('marked_actions',marked_actions)
		qubit_register = matpr(marked_actions, n_qubits)
		#print('qubit register', qubit_register)
		_, tail_discount = tail_simplify(qubit_register, n_qubits)
		count -= tail_discount

	if simplify_mid:
		marked_actions = mark_actions(action_list, action_markers)
		qubit_register = matpr(marked_actions, n_qubits)
		_, mid_discount = mid_simplify(qubit_register, n_qubits)
		count -= mid_discount
		#print(count)

	if not done:
		raise ValueError("Sequence didn't obtain gate set")

	return action_markers, count, tail_discount, mid_discount


def check_action(action_list: List,
				 n_qubits: int,
				 u_list_single: List,
				 u_list_double: List,
				 s_op_list: List,
				 t_op_list: List) -> bool:

	'''
	Check whether the list of mapping gates maps all target to source gates.

	Args:
		action_list: list of mapping gates to be counted.
		n_qubits: number of qubits in total.
		u_list_single: list of single qubit mapping gates.
		u_list_double: list of double qubit mapping gates.
		s_op_list: list of target gates.
		t_op_list: list of source gates.

	Returns:
		done: Boolean. If true, all target gates have been successfully mapped to source gates.

	'''

	done = False
	s_op_list = [op for op in s_op_list if op['g']
				 not in [t['g'] for t in t_op_list]]

	for a in range(len(action_list)):
		if action_list[a]['g'] == 'I':
			continue

		else:
			new_s_op_list = [action(op, action_list[a], n_qubits, u_list_single, u_list_double, t_op_list) for op in s_op_list]
			new_s_op_list = [op for op in new_s_op_list if op['g']
							 not in [t['g'] for t in t_op_list]]

			if len(new_s_op_list) == 0:
				done = True
				break
			else:
				s_op_list = new_s_op_list

	return done
