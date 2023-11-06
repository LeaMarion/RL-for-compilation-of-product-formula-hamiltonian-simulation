"""
/* Copyright (C) 2023 Eleanor Scerri (Elliescerri) - All Rights Reserved
 * You may use, distribute and modify this code under the terms of the GNU GENERAL PUBLIC LICENSE v3 license.
 *
 * You should have received access to a copy of the GNU GENERAL PUBLIC LICENSE v3 license with
 * this file.
 */
"""

import json
import numpy as np
import random
import openfermion as of
import cirq
import copy
import re
import os
import itertools as it

from copy import deepcopy
from typing import Tuple, Union, List, Dict, Mapping

def HtoPS(h: of.QubitOperator, nqubits: int, index: int) -> cirq.PauliString:
	'''
	Deprecated, PauliStrings not used anymore.
	Convert Hamiltonian term to Paulistring.

	Args:
		h: Hamiltonian
		nqubits: number of qubits in the system
		index: index of Hamiltonian term to be converted

	Returns:
		cirq.PauliString

	'''

	operator = list(h)[index]

	if len(list(operator.terms.keys())[0]) == 0:
		return cirq.PauliString(cirq.I(cirq.LineQubit(0)))

	else:
		op_subops = np.array(list(operator.terms.keys())[0])[:, 1]
		op_qubits = [int(q_str) for q_str in np.array(
			list(operator.terms.keys())[0])[:, 0]]

		op_str = ['{}{}'.format(o, q) for (o, q) in zip(op_subops, op_qubits)]
		operator = of.QubitOperator(' '.join(op_str))

		return QOtoPS(operator)


def HtoGate(h: of.QubitOperator, qubits: List[cirq.LineQubit], index: int) -> Tuple[cirq.Circuit, Tuple[str]]:
	'''
	Deprecated, PauliStrings not used anymore. Mainly for testing purposes.
	Convert Hamiltonian term to circuit.

	Args:
		h: Hamiltonian
		qubits: qubits in the system
		index: index of Hamiltonian term to be converted

	Returns:
		circuit: circuit corresponding to Hamiltonian term
		meas_key: measurement key in case of measurement

	'''

	operator = list(h)[index]

	circuit = cirq.Circuit()

	rot_dic = {'X': lambda q, inv: cirq.H.on(q),
			   'Y': lambda q, inv: cirq.rx(-inv*np.pi/2).on(q),
			   'Z': lambda q, inv: cirq.I.on(q)}

	paulis = list(operator.terms.keys())[0]
	meas_key = ()

	for qbt, pau in paulis:
		meas_key += (pau+str(qbt),)
		circuit.append(rot_dic[pau](qubits[qbt], 1))
		circuit.append(cirq.measure(qubits[qbt], key=pau+str(qbt)))

	return circuit, meas_key


def QOtoPS(qobj: of.QubitOperator) -> cirq.PauliString:
	'''
	Deprecated, PauliStrings not used anymore.
	Convert of.QubitOperator to cirq.PauliString

	Args:
		qobj: QubitOperator to be converted

	Returns:
		cirq.PauliString

	'''

	rot_dic = {'X': lambda q: cirq.X(q),
			   'Y': lambda q: cirq.Y(q),
			   'Z': lambda q: cirq.Z(q)}

	coeff = list(qobj.terms.values())[0]

	keys = np.array(list(qobj.terms.keys())[0])

	if len(keys) == 0:
		return cirq.PauliString(coeff, cirq.I(cirq.LineQubit(0)))

	else:
		qubit_idx = [int(idx) for idx in keys[:, 0]]
		pauli_idx = list(keys[:, 1])

		ops = [rot_dic[p_idx](cirq.LineQubit(q_idx))
			   for (p_idx, q_idx) in zip(pauli_idx, qubit_idx)]

		return cirq.PauliString(coeff, *ops)


def PStoQO(pobj: cirq,PauliString) -> of.QubitOperator:

	'''
	Deprecated, PauliStrings not used anymore.
	Convert cirq.PauliString to of.QubitOperator

	Args:
		pobj: cirq.PauliString

	Returns:
		of.QubitOperator

	'''

	rot_dic = {cirq.X: lambda q: of.QubitOperator('X{}'.format(q)),
			   cirq.Y: lambda q: of.QubitOperator('Y{}'.format(q)),
			   cirq.Z: lambda q: of.QubitOperator('Z{}'.format(q))}

	coeff = pobj.coefficient

	qubit_idx = [q.x for q in pobj.qubits]
	pauli_idx = list(pobj.values())

	qubit_op = of.QubitOperator('', coeff)

	for idx in range(len(qubit_idx)):
		qubit_op = qubit_op*rot_dic[pauli_idx[idx]](qubit_idx[idx])

	return qubit_op


def StringtoPS(operator: List[Mapping[str, Union[complex, str]]]) -> cirq.PauliString:
	'''
	Convert dict-format operator to cirq.PauliString.

	Args:
		operator: Operator to be converted

	Returns:
		cirq.PauliString

	'''

	qubits = cirq.LineQubit.range(len(operator["g"]))

	ps_dict = {}
	conv_dict = {"I": cirq.I,
				 "X": cirq.X,
				 "Y": cirq.Y,
				 "Z": cirq.Z}

	for idx in range(len(operator["g"])):
		ps_dict[qubits[idx]] = conv_dict[operator["g"][idx]]

	return operator["c"]*cirq.PauliString(ps_dict)


def PStoString(paulistring: cirq.PauliString, nqubits: int) -> Dict[str, Union[complex, str]]:

	'''
	Convert cirq.PauliString to dict-format operator.

	Args:
		paulistring: cirq.PauliString
		nqubits: number of qubits

	Returns:
		operator in dict-format

	'''

	qubits_idx = [q.x for q in paulistring.qubits]
	gates = paulistring.gate
	coeff = paulistring.coefficient
	op_str = ""

	for idx in range(nqubits):
		if idx not in qubits_idx:
			op_str += "I"

		else:
			if gates[qubits_idx.index(idx)] == cirq.X:
				op_str += "X"
			elif gates[qubits_idx.index(idx)] == cirq.Y:
				op_str += "Y"
			elif gates[qubits_idx.index(idx)] == cirq.Z:
				op_str += "Z"

	return {"c": coeff, "g": op_str}


def load_json(idx: Union[int, str]) -> Dict[str, Union[int, float, str]]:
	print(idx)
	with open('../JSON/config_{}.json'.format(idx)) as json_data_file:
		dict = json.load(json_data_file)

	return dict


def save_json(dir: str, dict: Mapping[str, Union[int, float, str]]):

	with open(dir, "w") as outfile:
		  json.dump(dict, outfile)


def check_folders(directory: str, folder_names: List[str]):

	for folder in folder_names:
		if not os.path.exists(directory+"/"+folder):
			print("Created {}".format(folder))
			os.makedirs(directory+"/"+folder)


def generate_ops(n_qubits: int, method: str, pad: bool = False) -> Tuple:
	'''
	Generate target (s_list), source (t_list) and mapping gates (u_list) given number of qubits.

	Args:
		n_qubits: number of qubits
		pad: Boolean to see whether to pad target gates or not

	Returns:
		target, source, mapping gates

	'''

	all_paulis = []
	for tup in it.product([cirq.I, cirq.X, cirq.Y, cirq.Z], repeat=n_qubits):
		all_paulis.append(cirq.PauliString(
			[tup[j](cirq.LineQubit(j)) for j in range(n_qubits)]))

	s_list = [PStoString(ps, n_qubits) for ps in all_paulis]

	q_combs = [(q,) for q in range(n_qubits+pad)]+[(q, q+1) for q in range(n_qubits+pad-1)]

	t_list = [cirq.PauliString(cirq.I(cirq.LineQubit(0)))]+[cirq.PauliString(
		cirq.Z(cirq.LineQubit(q_idx)) for q_idx in comb) for comb in q_combs]
	t_list = [PStoString(t, n_qubits+pad) for t in t_list]

	action_ops = TestOps(n_qubits+pad)

	if method == "SA":
		u_list = [action_ops.I, action_ops.H, action_ops.S, action_ops.CNOT, action_ops.SWAP]

	else:
		u_list = [action_ops.H, action_ops.S, action_ops.CNOT, action_ops.SWAP]

	return s_list, t_list, u_list, action_ops


def pad(s_list: List[Mapping[str, Union[complex, str]]], pad_idx: int) -> List[Dict[str, Union[complex, str]]]:
	'''
	Padding function for padded training. Add identity operations as desired location.
	ex: pad([{"c":1, "g": "XXX"}], 2) returns [{"c":1, "g": "XXIX"}]

	Args:
		s_list: list of operations to be padded
		pad_idx: padding index

	Returns:
		padded list of operations

	'''

	if pad_idx == -1:
		return s_list

	else:
		return [{"c":s["c"], "g": s["g"][:pad_idx]+"I"+s["g"][pad_idx:]} for  s in s_list]


def mapping_gate_count(p_dict_list: List[Mapping[str, Union[complex, str]]], nqubits: int) -> int:
	'''
	Count number of gates requred to map set to set of implementable gates (single Z gates or neighbouring ZZ gates)
	using the naive approach.

	Currently the set of mapping gates is assumed to be SWAP, CNOT (both on nearest neighbours), H and S

	Args:
		p_dict_list: list of gates to be mapped
		nqubits: number of qubits in the system

	Returns:
		gate_count: number of actions required (NOT number of total gates involved, which is O(2*gate_count))

	'''

	state = [p_dict["g"] for p_dict in p_dict_list]
	gate_count = 0

	for s_sub in state:
		id_pos = [m.start() for m in re.finditer('I', 'I'+s_sub+'I')]
		swap_count = 0

		if np.count_nonzero(np.diff(id_pos)-1)<2:
			pass

		else:
			for j in range(len(s_sub)//2):
				if s_sub[j]=="I":
					swap_count+=len(s_sub[:j])-s_sub[:j].count("I")

			for j in reversed(range(len(s_sub)//2, len(s_sub))):
				if s_sub[j]=="I":
					swap_count+=len(s_sub[j+1:])-s_sub[j+1:].count("I")

		cnot_count = max(0,len(s_sub)-s_sub.count("I")-2)
		basis_count = s_sub.count("X")+2*s_sub.count("Y")

		gate_count += swap_count+cnot_count+basis_count

	return gate_count


def mapping_gate(p_dict_list: List[Mapping[str, Union[complex, str]]],
				mapping_ops: 'TestOps',
				nqubits: int,
				shuffle: bool,
				shuffle_seed: int) -> List[List[Dict[str, Union[int, str, Tuple[int]]]]]:

	'''
	Get list of gates required to map set to set of implementable gates (single Z gates or neighbouring ZZ gates)
	using the naive approach.

	Args:
		p_dict_list: list of gates to be mapped
		mapping_ops: mapping operators
		nqubits: number of qubits in the system
		shuffle: shuffle seed operators

	Returns:
		actions: actions required

	'''

	seed = deepcopy(p_dict_list)

	if shuffle:
		random.Random(shuffle_seed).shuffle(seed)

	swapped_dict = deepcopy(seed)
	state = [p_dict["g"] for p_dict in seed]
	actions = [[] for j in range(len(state))]

	for s in range(len(state)):
		id_pos = [m.start() for m in re.finditer('I', 'I'+state[s]+'I')]
		swap = True

		if np.count_nonzero(np.diff(id_pos)-1)<2:
			swap = False

		for j in range(len(state[s])//2):
			if state[s][j]=="I" and swap:
				swap_count = len(state[s][:j])-state[s][:j].count("I")
				for i in range(swap_count):
					swapped_dict[s] = mapping_ops.SWAP(j-i-1,j-i,swapped_dict[s])
					actions[s].append({"g":"SWAP", "q":(j-i-1,j-i)})

			elif state[s][j]=="X":
				swapped_dict[s] = mapping_ops.H(j,swapped_dict[s])
				actions[s].append({"g":"H", "q":j})

			elif state[s][j]=="Y":
				swapped_dict[s] = mapping_ops.H(j,mapping_ops.S(j,swapped_dict[s]))
				actions[s].append({"g":"S", "q":j})
				actions[s].append({"g":"H", "q":j})

		for j in reversed(range(len(state[s])//2, len(state[s]))):
			if state[s][j]=="I" and swap:
				swap_count = len(state[s][j+1:])-state[s][j+1:].count("I")
				for i in range(swap_count):
					swapped_dict[s] = mapping_ops.SWAP(j+i,j+i+1,swapped_dict[s])
					actions[s].append({"g":"SWAP", "q":(j+i,j+i+1)})

			elif state[s][j]=="X":
				swapped_dict[s] = mapping_ops.H(j,swapped_dict[s])
				actions[s].append({"g":"H", "q":j})

			elif state[s][j]=="Y":
				swapped_dict[s] = mapping_ops.H(j,mapping_ops.S(j,swapped_dict[s]))
				actions[s].append({"g":"S", "q":j})
				actions[s].append({"g":"H", "q":j})

		z_pos = [m.start() for m in re.finditer('Z', swapped_dict[s]['g'])]
		if len(z_pos)>2:
			for j in range(len(z_pos)-2):
				swapped_dict[s] = mapping_ops.CNOT(z_pos[j],z_pos[j+1],swapped_dict[s])
				actions[s].append({"g":"CNOT","q":(z_pos[j],z_pos[j+1])})

	return actions


def decimalToBinary(n, action_bitspace_size):
	bin_num = bin(n).replace("0b", "")
	bin_num = "0"*(action_bitspace_size-len(bin_num))+bin_num

	return bin_num

def actionToActionSA(action_list):

	action_list_SA = []

	for j in range(len(action_list)):
		if j==0:
			action_list_SA += action_list[0]
		else:
			action_list_SA += list(reversed(action_list[j-1]))+action_list[j]

	return action_list_SA

def actionSAToBinary(action_list, mapping_ops_names, n_single, n_qubits, action_bitspace_size):

	combs = [(q,q+1) for q in range(n_qubits-1)]
	bin_actions = []

	for action in action_list:
		g_idx = mapping_ops_names.index(action["g"])

		if g_idx < n_single:
			bin_idx = g_idx*n_qubits + action["q"]

		else:
			bin_idx = n_single*n_qubits + (g_idx-n_single)*len(combs) + combs.index(action["q"])

		bin_actions.append(decimalToBinary(bin_idx, action_bitspace_size))

	return bin_actions


def actionToActionDQN(action_list: List[List[Mapping[str, Union[int, str, Tuple[int]]]]]) -> List[Mapping[str, Union[int, str, Tuple[int]]]]:
	'''
	Convert from list of actions for naive solutions to list of solutions compatible with Tom's format.

	Args:
		action_list: list of lists for naive actions, nth list mapping the nth operator to an implementable gate

	Returns:
		action_list_DDQN: list of actions compatible with Tom's format

	'''

	action_list_DDQN = []

	for j in range(len(action_list)):
		if j==0:
			action_list_DDQN += action_list[0]
		else:
			action_list_DDQN += list(reversed(action_list[j-1]))+action_list[j]

	return action_list_DDQN


def actionToInd(action_list: List[Mapping[str, Union[int, str, Tuple[int]]]],
				mapping_ops_names: List[str],
				n_single: int,
				nqubits: int) -> List[int]:

	'''
	Convert actions from actionToActionDQN to indices according to list of mapping operators and qubit index.

	Args:
		action_list: list of actions to map entire set of gates to implementable gates
		mapping_op_names: name of mapping operations used in action_list
		n_single: number of single qubit mapping operations
		nqubits: number of qubits involved

	Returns:
		int_actions: list of mapping gate indices

	'''

	combs = [(q,q+1) for q in range(nqubits-1)]
	int_actions = []

	for action in action_list:
		g_idx = mapping_ops_names.index(action["g"])

		if g_idx < n_single:
			act_idx = g_idx*nqubits + action["q"]

		else:
			act_idx = n_single*nqubits + (g_idx-n_single)*len(combs) + combs.index(action["q"])

		int_actions.append(act_idx)

	return int_actions


class TestOps():

	def __init__(self, nqubits: int):

		self.nqubits = nqubits


	def CNOT(self, p: int, q: int, operator: Mapping[str, Union[complex, str]]) -> Dict[str, Union[complex, str]]:

		paulistring = StringtoPS(operator)
		qubits = [q.x for q in paulistring.qubits]
		pauli_ops = list(paulistring.values())

		IX = {"c": 1, "g": "".join(
			["X" if j == q else "I" for j in range(self.nqubits)])}
		ZI = {"c": 1, "g": "".join(
			["Z" if j == p else "I" for j in range(self.nqubits)])}
		ZX = {"c": -1, "g": "".join(["Z" if j == p else "X" if j ==
									 q else "I" for j in range(self.nqubits)])}

		cnot_ops = [IX, ZI, ZX]

		for op in cnot_ops:
			op_pstring = StringtoPS(op)
			if paulistring*op_pstring != op_pstring*paulistring:
				paulistring = 1j*paulistring*op_pstring

		return PStoString(paulistring, self.nqubits)


	def SWAP(self, p: int, q: int, operator: Mapping[str, Union[complex, str]]) -> Dict[str, Union[complex, str]]:

		swapped_operator = copy.deepcopy(operator)
		swapped_list = list(swapped_operator["g"])

		swapped_list[p], swapped_list[q] = operator["g"][q], operator["g"][p]
		swapped_operator["g"] = "".join(swapped_list)

		return swapped_operator


	def expXX(self, p: int, q: int, operator: Mapping[str, Union[complex, str]]) -> Dict[str, Union[complex, str]]:

		paulistring = StringtoPS(operator)
		qubits = [q.x for q in paulistring.qubits]
		pauli_ops = list(paulistring.values())

		XX = cirq.PauliString([cirq.X(cirq.LineQubit(j)) for j in range(self.nqubits) if j in [p,q]])

		if paulistring*XX != XX*paulistring:
			paulistring = 1j*paulistring*XX

		return PStoString(paulistring, self.nqubits)


	def S(self, p: int, operator: Mapping[str, Union[complex, str]]) -> Dict[str, Union[complex, str]]:

		assert p < self.nqubits, 'Invalid qubit'

		gates = operator["g"]
		coeff = operator["c"]

		new_op = ""

		for idx in range(self.nqubits):
			if idx != p:
				op = gates[idx]

			else:
				if gates[idx] == "I":
					op = "I"
				elif gates[idx] == "X":
					op = "Y"
					coeff *= -1
				elif gates[idx] == "Y":
					op = "X"
				elif gates[idx] == "Z":
					op = "Z"

			new_op += op

		return {"c": coeff, "g": new_op}


	def Sdag(self, p: int, operator: Mapping[str, Union[complex, str]]) -> Dict[str, Union[complex, str]]:

		assert p < self.n_qubits, 'Invalid qubit'

		gates = operator["g"]
		coeff = operator["c"]

		new_op = ""

		for idx in range(self.n_qubits):
			if idx != p:
				op = gates[idx]

			else:
				if gates[idx] == "I":
					op = "I"
				elif gates[idx] == "X":
					op = "Y"
				elif gates[idx] == "Y":
					op = "X"
					coeff *= -1
				elif gates[idx] == "Z":
					op = "Z"

			new_op += op

		return {"c": coeff, "g": new_op}


	def H(self, p: int, operator: Mapping[str, Union[complex, str]]) -> Dict[str, Union[complex, str]]:

		assert p < self.nqubits, 'Invalid qubit'

		gates = operator["g"]
		coeff = operator["c"]

		new_op = ""

		for idx in range(self.nqubits):
			if idx != p:
				op = gates[idx]

			else:
				if gates[idx] == "I":
					op = "I"
				elif gates[idx] == "X":
					op = "Z"
				elif gates[idx] == "Y":
					op = "Y"
					coeff *= -1
				elif gates[idx] == "Z":
					op = "X"

			new_op += op

		return {"c": coeff, "g": new_op}


	def I(self, p: int, operator: Mapping[str, Union[complex, str]]) -> Dict[str, Union[complex, str]]:

		return operator
