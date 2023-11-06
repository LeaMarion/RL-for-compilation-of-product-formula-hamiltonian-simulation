"""
/* Copyright (C) 2023 Lea Trenkwalder (LeaMarion) - All Rights Reserved
 * You may use, distribute and modify this code under the terms of the GNU GENERAL PUBLIC LICENSE v3 license.
 *
 * You should have received access to a copy of the GNU GENERAL PUBLIC LICENSE v3 license with
 * this file.
 */
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib
import itertools as it
sys.path.insert(1, "../shared")
sys.path.insert(1, "../SA")
print(sys.path)
import HSC_utils as hsc_utils
from HSC_SA_utils import *


def generate_csv(table_dict_naive, table_dict,  config, table_type):

    f = open("../SA/SA_RESULTS/table_"+config+"_"+table_type+".csv", "+w")

    header_list = ['seed','alpha','naivelen', 'naivelenOG','minsoln','naiveminsoln','minsolnred','naiveminsolnred','reps','evals']
    numbers = range(len(header_list))
    header_dict = dict(zip(numbers, header_list))
    headers = ";".join(header_list) +';\n'
    #print(headers)
    f.write(headers)
    ALPHA = [0.1,0.25,0.5,1]
    if table_type == 'percent':
        for seed in table_dict.keys():
            notdone = True
            for alpha in ALPHA:
                if alpha == 0.1:
                    line_content = str(seed)
                    line_content += ';' + str(alpha)
                    line_content += ';'+str(table_dict[seed]['naivelen'])+';'+str(table_dict[seed]['naivelenOG'])
                    percent_minsoln = int(np.round(table_dict[seed][alpha]['minsoln']/table_dict[seed]['naivelenOG']*100,0))
                    percent_minsolnred = int(np.round(table_dict[seed][alpha]['minsolnred']/table_dict[seed]['naivelenOG']*100,0))
                    percent_naiveminsoln = int(
                        np.round(table_dict_naive[seed][alpha]['minsoln'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    percent_naiveminsolnred = int(
                        np.round(table_dict_naive[seed][alpha]['minsolnred'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    line_content += ';'+str(percent_minsoln)+'\%;'+str(percent_naiveminsoln)+'\%;'+str(percent_minsolnred)+'\%;'+str(percent_naiveminsolnred)+'\%;'
                    reps = str(table_dict[seed][alpha]['reps'])
                    evals = str(table_dict[seed][alpha]['evals'])
                    line_content += reps + ';' + evals + ';\n'
                else:
                    line_content += ';'+ str(alpha)+';;'
                    percent_minsoln = int(
                        np.round(table_dict[seed][alpha]['minsoln'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    percent_minsolnred = int(
                        np.round(table_dict[seed][alpha]['minsolnred'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    percent_naiveminsoln = int(
                        np.round(table_dict_naive[seed][alpha]['minsoln'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    percent_naiveminsolnred = int(
                        np.round(table_dict_naive[seed][alpha]['minsolnred'] / table_dict[seed]['naivelenOG'] * 100,
                                 0))
                    line_content += ';'+str(percent_minsoln)+'\%;'+str(percent_naiveminsoln)+'\%;'+str(percent_minsolnred)+'\%;'+str(percent_naiveminsolnred)+'\%;'
                    reps = table_dict[seed][alpha]['reps']
                    evals = table_dict[seed][alpha]['evals']
                    line_content += reps + ';' + evals + ';\n'
            print(line_content)

            f.write(line_content)
        f.close()
    else:
        for seed in table_dict.keys():
            notdone = True
            for alpha in ALPHA:
                if alpha == 0.1:
                    line_content = str(seed)
                    line_content += ';' + str(alpha)
                    line_content += ';' + str(table_dict[seed]['naivelen']) + ';' + str(table_dict[seed]['naivelenOG'])
                    minsoln = table_dict[seed][alpha]['minsoln']
                    minsolnred = table_dict[seed][alpha]['minsolnred']
                    naiveminsoln = table_dict_naive[seed][alpha]['minsoln']
                    naiveminsolnred = table_dict_naive[seed][alpha]['minsolnred']
                    line_content += ';' + str(minsoln) + ';' + str(naiveminsoln) + ';' + str(
                        minsolnred) + ';' + str(naiveminsolnred)+';'
                    reps = str(table_dict[seed][alpha]['reps'])
                    evals = str(table_dict[seed][alpha]['evals'])
                    line_content += reps + ';' + evals + ';\n'
                else:
                    line_content += ';' + str(alpha) + ';;'
                    minsoln = table_dict[seed][alpha]['minsoln']
                    minsolnred = table_dict[seed][alpha]['minsolnred']
                    naiveminsoln = table_dict_naive[seed][alpha]['minsoln']
                    naiveminsolnred = table_dict_naive[seed][alpha]['minsolnred']
                    line_content += ';' + str(minsoln) + ';' + str(naiveminsoln) + ';' + str(
                        minsolnred) + ';' + str(naiveminsolnred)+';'
                    reps = str(table_dict[seed][alpha]['reps'])
                    evals = str(table_dict[seed][alpha]['evals'])
                    line_content += reps + ';' + evals + ';\n'
            print(line_content)

            f.write(line_content)
        f.close()


def generate_csv_reps(table_dict_naive, table_dict,  config, table_type):

    f = open("../SA/SA_RESULTS/table_"+config+"_"+table_type+".csv", "+w")

    header_list = ['seed','alpha','naivelen', 'naivelenOG','minsoln','minsolnred','reps','evals','naiveminsoln','naiveminsolnred','naivereps','naiveevals']
    numbers = range(len(header_list))
    header_dict = dict(zip(numbers, header_list))
    headers = ";".join(header_list) +';\n'
    #print(headers)
    f.write(headers)
    ALPHA = [0.1,0.25,0.5,1]
    if table_type == 'percent':
        for seed in table_dict.keys():
            notdone = True
            for alpha in ALPHA:
                if alpha == 0.1:
                    line_content = str(seed)
                    line_content += ';' + str(alpha)
                    line_content += ';'+str(table_dict[seed]['naivelen'])+';'+str(table_dict[seed]['naivelenOG'])
                    percent_minsoln = int(np.round(table_dict[seed][alpha]['minsoln']/table_dict[seed]['naivelenOG']*100,0))
                    percent_minsolnred = int(np.round(table_dict[seed][alpha]['minsolnred']/table_dict[seed]['naivelenOG']*100,0))
                    percent_naiveminsoln = int(
                        np.round(table_dict_naive[seed][alpha]['minsoln'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    percent_naiveminsolnred = int(
                        np.round(table_dict_naive[seed][alpha]['minsolnred'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    reps = table_dict[seed][alpha]['reps']
                    evals = table_dict[seed][alpha]['evals']
                    naivereps = table_dict[seed][alpha]['naive_reps']
                    naiveevals = table_dict[seed][alpha]['naive_evals']
                    line_content += ';' + str(percent_minsoln) + '\%;' + str(percent_minsolnred) + '\%;'
                    line_content += str(reps) + ';' + str(evals) + ';'
                    line_content +=  str(percent_naiveminsoln)+ '\%;' + str(percent_naiveminsolnred) + '\%;'
                    line_content += str(naivereps) + ';' + str(naiveevals) + ';\n'
                else:
                    line_content += ';'+ str(alpha)+';;'
                    percent_minsoln = int(
                        np.round(table_dict[seed][alpha]['minsoln'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    percent_minsolnred = int(
                        np.round(table_dict[seed][alpha]['minsolnred'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    percent_naiveminsoln = int(
                        np.round(table_dict_naive[seed][alpha]['minsoln'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    percent_naiveminsolnred = int(
                        np.round(table_dict_naive[seed][alpha]['minsolnred'] / table_dict[seed]['naivelenOG'] * 100,
                                 0))
                    reps = table_dict[seed][alpha]['reps']
                    evals = table_dict[seed][alpha]['evals']
                    naivereps = table_dict[seed][alpha]['naive_reps']
                    naiveevals = table_dict[seed][alpha]['naive_evals']
                    line_content += ';'+str(percent_minsoln)+'\%;'+str(percent_minsolnred)+'\%;'
                    line_content +=str(reps) + ';' + str(evals) + ';'
                    line_content +=str(percent_naiveminsoln)+'\%;'+str(percent_naiveminsolnred)+'\%;'
                    line_content += str(naivereps) + ';' + str(naiveevals) + ';\n'


            print(line_content)

            f.write(line_content)
        f.close()
    else:
        for seed in table_dict.keys():
            notdone = True
            for alpha in ALPHA:
                if alpha == 0.1:
                    line_content = str(seed)
                    line_content += ';' + str(alpha)
                    line_content += ';' + str(table_dict[seed]['naivelen']) + ';' + str(table_dict[seed]['naivelenOG'])
                    minsoln = table_dict[seed][alpha]['minsoln']
                    minsolnred = table_dict[seed][alpha]['minsolnred']
                    naiveminsoln = table_dict_naive[seed][alpha]['minsoln']
                    naiveminsolnred = table_dict_naive[seed][alpha]['minsolnred']
                    line_content += ';' + str(minsoln) + ';' + str(naiveminsoln) + ';' + str(
                        minsolnred) + ';' + str(naiveminsolnred)+';'
                    reps = str(table_dict[seed][alpha]['reps'])
                    evals = str(table_dict[seed][alpha]['evals'])
                    line_content += reps + ';' + evals + ';\n'
                else:
                    line_content += ';' + str(alpha) + ';;'
                    minsoln = table_dict[seed][alpha]['minsoln']
                    minsolnred = table_dict[seed][alpha]['minsolnred']
                    naiveminsoln = table_dict_naive[seed][alpha]['minsoln']
                    naiveminsolnred = table_dict_naive[seed][alpha]['minsolnred']
                    line_content += ';' + str(minsoln) + ';' + str(naiveminsoln) + ';' + str(
                        minsolnred) + ';' + str(naiveminsolnred)+';'
                    reps = str(table_dict[seed][alpha]['reps'])
                    evals = str(table_dict[seed][alpha]['evals'])
                    line_content += reps + ';' + evals + ';\n'
            print(line_content)

            f.write(line_content)
        f.close()

def generate_csv_single(table_dict_naive, table_dict,  config, table_type):

    f = open("../SA/SA_RESULTS/table_"+config+"_"+table_type+".csv", "+w")

    header_list = ['seed','alpha','naivelen', 'naivelenOG','minsoln','minsolnred','reps','evals','naiveminsoln','naiveminsolnred','naivereps','naiveevals']
    numbers = range(len(header_list))
    header_dict = dict(zip(numbers, header_list))
    headers = ";".join(header_list) +';\n'
    #print(headers)
    f.write(headers)
    ALPHA = [0.25]
    if table_type == 'percent':
        for seed in table_dict.keys():
            notdone = True
            for alpha in ALPHA:
                if alpha == 0.25:
                    line_content = str(seed)
                    line_content += ';' + str(alpha)
                    line_content += ';'+str(table_dict[seed]['naivelen'])+';'+str(table_dict[seed]['naivelenOG'])
                    percent_minsoln = int(np.round(table_dict[seed][alpha]['minsoln']/table_dict[seed]['naivelenOG']*100,0))
                    percent_minsolnred = int(np.round(table_dict[seed][alpha]['minsolnred']/table_dict[seed]['naivelenOG']*100,0))
                    percent_naiveminsoln = int(
                        np.round(table_dict_naive[seed][alpha]['minsoln'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    percent_naiveminsolnred = int(
                        np.round(table_dict_naive[seed][alpha]['minsolnred'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    reps = table_dict[seed][alpha]['reps']
                    evals = table_dict[seed][alpha]['evals']
                    naivereps = table_dict_naive[seed][alpha]['reps']
                    naiveevals = table_dict_naive[seed][alpha]['evals']
                    line_content += ';' + str(percent_minsoln) + '\%;' + str(percent_minsolnred) + '\%;'
                    line_content += str(reps) + ';' + str(evals) + ';'
                    line_content +=  str(percent_naiveminsoln)+ '\%;' + str(percent_naiveminsolnred) + '\%;'
                    line_content += str(naivereps) + ';' + str(naiveevals) + ';\n'
                else:
                    line_content += ';'+ str(alpha)+';;'
                    percent_minsoln = int(
                        np.round(table_dict[seed][alpha]['minsoln'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    percent_minsolnred = int(
                        np.round(table_dict[seed][alpha]['minsolnred'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    percent_naiveminsoln = int(
                        np.round(table_dict_naive[seed][alpha]['minsoln'] / table_dict[seed]['naivelenOG'] * 100, 0))
                    percent_naiveminsolnred = int(
                        np.round(table_dict_naive[seed][alpha]['minsolnred'] / table_dict[seed]['naivelenOG'] * 100,
                                 0))
                    reps = table_dict[seed][alpha]['reps']
                    evals = table_dict[seed][alpha]['evals']
                    naivereps = table_dict_naive[seed][alpha]['reps']
                    naiveevals = table_dict_naive[seed][alpha]['evals']
                    line_content += ';'+str(percent_minsoln)+'\%;'+str(percent_minsolnred)+'\%;'
                    line_content +=str(reps) + ';' + str(evals) + ';'
                    line_content +=str(percent_naiveminsoln)+'\%;'+str(percent_naiveminsolnred)+'\%;'
                    line_content += str(naivereps) + ';' + str(naiveevals) + ';\n'
            print(line_content)
            f.write(line_content)
        f.close()


def calculate_singular_solution(qubit = 4, size = 8, x_type = 'action', set_seed_idx=4, exp = ''):
    """
    Args:

    """
    if x_type == 'gate':
        gate_factor = 2
    else:
        gate_factor = 1

    folder_name = 'SA_RESULTS/START_STATES/'

    s_list, t_list, u_list, action_ops = hsc_utils.generate_ops(n_qubits=qubit, method="SA")
    #print(folder_name)
    s_tup = np.load(folder_name+'state_'+str(set_seed_idx)+'.npy', allow_pickle=True)
    actions = mapping_gate(s_tup, action_ops, qubit, False, None)
    length = 0
    for action in actions:
        length += len(action)
    return length*gate_factor

def calculate_sequential_solution(qubit = 4, size = 8, x_type = 'action', set_seed_idx=4, exp = 'overlap', average_over=100):
    """
    Args:

    """
    if x_type == 'gate':
        gate_factor = 2
    else:
        gate_factor = 1

    folder_name = 'SA_RESULTS/START_STATES/'

    s_list, t_list, u_list, action_ops = hsc_utils.generate_ops(n_qubits=qubit, method="SA")

    s_tup = np.load(folder_name+'state_'+str(set_seed_idx)+'.npy', allow_pickle=True)
    actions = mapping_gate(s_tup, action_ops, qubit, False, None)

    seed = np.random.RandomState(set_seed_idx)
    avg_num_actions = 0
    avg = average_over
    for i in range(avg):
        num_actions = 0
        iters = seed.permutation(len(actions))
        for idx in iters[:-1]:
            num_actions+=2*len(actions[idx])
        idx = iters[-1]
        num_actions += len(actions[idx])
        avg_num_actions += num_actions
    avg_num_actions=avg_num_actions/avg
    return avg_num_actions*gate_factor


if __name__ == "__main__":
    TABLE = True
    COUNT_STEPS = True

    par_idx = 4

    ACTIONS_DIR = "SA_RESULTS/ACTIONS"
    COSTS_DIR = "SA_RESULTS/COSTS"
    EVALUATIONS_DIR = "SA_RESULTS/EVALUATIONS"

    REWARD_IDS_ARR = [0]  # In case of intermediate identity rewards. Leave as 0
    ALPHA_ARR = [0.1, 0.25, 0.5, 1.0]  # Initial temperatures
    SEED_ARR = np.arange(5)  # List of seed indices determining the initial target gate set
    CFGIDX_LIST = ["sa_naiveinit_1Mio","sa_emptyinit_1Mio"]  # List of parameter configurations
    EXP = CFGIDX_LIST[0]

    # Select configuration from the lists above

    par_list = list(it.product(CFGIDX_LIST, SEED_ARR,
                               ALPHA_ARR, REWARD_IDS_ARR))[par_idx]

    CFGIDX = par_list[0]
    SEED_IDX = par_list[1]
    ALPHA = par_list[2]
    REWARD_IDS = par_list[3]

    STRUCT_DICT = hsc_utils.load_json(CFGIDX)
    N_QUBITS = STRUCT_DICT["n_qubits"]  # Number of qubits
    SIZE = STRUCT_DICT["size"]  # SIZE of target gate set

    print(par_list)



    if COUNT_STEPS:
        qubit = STRUCT_DICT["n_qubits"]
        size = STRUCT_DICT["size"]

        seed = SEED_IDX
        alpha = ALPHA
        schedule = STRUCT_DICT["anneal_strategy"]
        init =  STRUCT_DICT["init"]+'_init'
        reward_ids = 0
        folder = 'SA_RESULTS/'+EXP+'/EVALUATIONS/'
        file_name_start = "SA_evaluations_"
        file_name = file_name_start +schedule+'__nqubits_' + str(qubit)+'__size_' + str(size) +'__alpha_'+str(alpha)+'__reward_ids_' + str(reward_ids) + '__seed_' + str(
            seed)+'__'+init+'_fair'
        evaluations = np.load(folder + file_name + '.npy', allow_pickle=True)
        print(evaluations)
        eval = 0
        for rep in evaluations:
            eval += rep
        print(eval)

        if TABLE:
            qubit = 4
            size = 8
            seeds = 5
            seq_average_over = 100
            table_type = 'percent'
            x_type = 'gate'
            config = 'naive'
            eval_threshold = 500000
            table_path_naive = "../SA/SA_RESULTS/min_num_gates_" + str(qubit) + "_qubits_" + config + "_eval_"+str(eval_threshold)+".npy"
            data_naive = np.load(table_path_naive, allow_pickle=True).item()
            config_2 = 'empty'
            table_path = "../SA/SA_RESULTS/min_num_gates_" + str(qubit) + "_qubits_" + config_2 +"_eval_"+str(eval_threshold)+ ".npy"
            data = np.load(table_path, allow_pickle=True).item()
            # LOAD singluar solution and add it to the table:
            N_sin_qubit = []
            for seed in range(seeds):
                N_sin = calculate_singular_solution(qubit=qubit, size=size, x_type=x_type, set_seed_idx=seed)
                N_sin_qubit.append(N_sin)
                # print(N_sin)
                data[seed].update({"naivelenOG": N_sin})

            for seed in range(seeds):
                N_seq = calculate_sequential_solution(qubit=qubit, size=size, x_type=x_type, set_seed_idx=seed,
                                                      average_over=seq_average_over)
                data[seed].update({"naivelen": N_seq})
            print(data)
            generate_csv_single(data_naive, data, config, table_type)


