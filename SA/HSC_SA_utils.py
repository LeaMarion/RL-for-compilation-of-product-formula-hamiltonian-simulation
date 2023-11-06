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
import random
import json
import numpy as np
import openfermion as of
import cirq
import inspect
import glob
import itertools as it
import copy
import time
import timeit
import statistics
import re

from collections import deque
from tqdm import tqdm
from copy import deepcopy
from math import log
import sys
from typing import Tuple, Union, List, Dict, Mapping
sys.path.insert(1, "../shared")
import HSC_utils as hsc_utils


def StringtoPS(operator):

    qubits = cirq.LineQubit.range(len(operator["g"]))

    ps_dict = {}
    conv_dict = {"I": cirq.I,
                 "X": cirq.X,
                 "Y": cirq.Y,
                 "Z": cirq.Z}

    for idx in range(len(operator["g"])):
        ps_dict[qubits[idx]] = conv_dict[operator["g"][idx]]

    return operator["c"]*cirq.PauliString(ps_dict)


def PStoString(paulistring, n_qubits):

    qubits_idx = [q.x for q in paulistring.qubits]
    gates = paulistring.gate
    coeff = paulistring.coefficient
    op_str = ""

    for idx in range(n_qubits):
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


def lc_symbol(a, b, c):

    if [a, b, c] in [[["X", "Y", "Z"][i - j] for i in range(3)] for j in range(3)]:
        return 1

    elif [a, b, c] in [[["X", "Y", "Z"][i - j] for i in range(3)] for j in range(3)]:
        return -1

    else:
        return 0

def mapping_gate(p_dict_list: List[Mapping[str, Union[complex, str]]],
                mapping_ops: 'TestOps',
                nqubits: int,
                shuffle: bool,
                shuffle_seed: int) -> List[List[Dict[str, Union[int, str, Tuple[int,int]]]]]:

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

def actionToInd(action_list, mapping_ops_names, n_single, n_qubits):

    combs = [(q,q+1) for q in range(n_qubits-1)]
    bin_actions = []

    for action in action_list:
        g_idx = mapping_ops_names.index(action["g"])

        if g_idx < n_single:
            bin_idx = g_idx*n_qubits + action["q"]

        else:
            bin_idx = n_single*n_qubits + (g_idx-n_single)*len(combs) + combs.index(action["q"])

        bin_actions.append(bin_idx)

    return bin_actions

class TestOps():

    def __init__(self, n_qubits):

        self.n_qubits = n_qubits

    def CNOT_ps(self, p, q, operator):

        paulistring = StringtoPS(operator)
        qubits = [q.x for q in paulistring.qubits]
        pauli_ops = list(paulistring.values())

        IX = {"c": 1, "g": "".join(
            ["X" if j == q else "I" for j in range(self.n_qubits)])}
        ZI = {"c": 1, "g": "".join(
            ["Z" if j == p else "I" for j in range(self.n_qubits)])}
        ZX = {"c": -1, "g": "".join(["Z" if j == p else "X" if j ==
                                     q else "I" for j in range(self.n_qubits)])}

        cnot_ops = [IX, ZI, ZX]

        for op in cnot_ops:
            op_pstring = StringtoPS(op)
            if paulistring*op_pstring != op_pstring*paulistring:
                paulistring = 1j*paulistring*op_pstring

        return PStoString(paulistring, self.n_qubits)

    def CNOT(self, p, q, operator):

        return self.CNOT_ps(p, q, operator)


    def SWAP_ps(self, p, q, operator):

        swapped_operator = copy.deepcopy(operator)
        swapped_list = list(swapped_operator["g"])

        swapped_list[p], swapped_list[q] = operator["g"][q], operator["g"][p]
        swapped_operator["g"] = "".join(swapped_list)

        return swapped_operator

    def SWAP(self, p, q, operator):

        return self.SWAP_ps(p, q, operator)

    def S(self, p, operator):

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
                    coeff *= -1
                elif gates[idx] == "Y":
                    op = "X"
                elif gates[idx] == "Z":
                    op = "Z"

            new_op += op

        return {"c": coeff, "g": new_op}

    def Sdag(self, p, operator):

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

    def H(self, p, operator):

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
                    op = "Z"
                elif gates[idx] == "Y":
                    op = "Y"
                    coeff *= -1
                elif gates[idx] == "Z":
                    op = "X"

            new_op += op

        return {"c": coeff, "g": new_op}

    def I(self, p, operator):

        return operator



if __name__ == "__main__":
    N_QUBITS = 4
    s_list, t_list, u_list, action_ops = hsc_utils.generate_ops(
        n_qubits=N_QUBITS, method="SA")
    s_tup = [{'c': (1+0.j), 'g':'XZII'},{'c': (1+0.j), 'g':'ZYXZ'},{'c': (1+0.j),'g':'YXXX'}]#,{'c': (1+0.j), 'g':'XZZZ'},{'c': (1+0.j), 'g':'XZYI'},{'c': (1+0.j), 'g':'ZYYY'},{'c': (1+0.j), 'g':'YYXX'},{'c': (1+0.j), 'g':'XYZZ'}]
    #a list that contains lists of actions to transform each gate into one of the target gates
    actions = mapping_gate(s_tup,action_ops, N_QUBITS,False, None)
    DQN_actions = hsc_utils.actionToActionDQN(actions)
    SA_actions = hsc_utils.actionToActionSA(actions)
    length = 0
    for action in actions:
        length += len(action)



    iters = list(it.permutations(range(len(actions)),len(actions)))
    count = 0
    seed = np.random.RandomState()

    seed.shuffle(iters)
    integer = seed.randint(0,high=len(iters))
    avg_num_actions = 0

    avg = 100

    for i in range(avg):
        num_actions = 0
        num_target = len(actions)
        integer = seed.randint(0, high=len(iters))
        for idx in iters[integer][:-1]:
            num_actions+=2*len(actions[idx])
        idx = iters[integer][-1]

        num_actions+=len(actions[idx])

        avg_num_actions += num_actions
    avg_num_actions=avg_num_actions/avg







