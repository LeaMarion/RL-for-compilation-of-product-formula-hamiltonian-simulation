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




def learning_curve(file_name_start, file_end = '', exp = '', with_N_sin = False, N_sin = [], x_type = 'action', max_steps = 1000, qubit_list = [], target_list=[], seed = 0, num_agents = 1, episodes = 15000, trial_avg = 100):
    """
    Create plot.
    Args:
    	file_name: Name of the input file.
    	num_agents: Number of agents to be averaged.
    	episodes: Number of episodes

    """
    fontsize_labels = 10
    plot_color_list = ['#1790ff','orange', '#04b605' , '#ec0c10' ,'y','m','k']
    print('Episodes', episodes)
    if x_type == 'gate':
        gate_factor = 2
        max_steps*=2
    else:
        gate_factor = 1

    if type(qubit_list) == list and type(target_list)== int:
        size = target_list
        plot_list = qubit_list

    elif type(qubit_list) == int  and type(target_list)== list:
        n_qubits = qubit_list
        plot_list = target_list

    else:
        raise(NotImplementedError)



    plt.figure(figsize=(8.5, 5.5))
    print('Episodes', episodes)
    plot_name = file_name_start+'_qubits='+str(qubit_list)+'_size='+str(target_list)
    min_values = []
    mean_list = []
    std_list = []
    for counter in plot_list:
        if type(target_list) == int:
            n_qubits = counter
        elif type(qubit_list) == int:
            size = counter
        EXPERIMENT = str(n_qubits) + 'q_' + str(size)+'t' + exp

        pathlib.Path('results/').mkdir(parents=True, exist_ok=True)
        plots_folder = 'results/' + EXPERIMENT + '/STEPS/'
        print(n_qubits)
        print('Episodes', episodes)
        # Initialize array to stack performances of all agents under consideration.
        step_array = np.zeros((1,episodes),int)
        print('Episodes', episodes)
        print('ary',step_array)
        num_loaded_agents = 0
        for ag in range(0,num_agents):
            # Loaded agent files.
            num_loaded_agents +=1
            file_name = file_name_start+'_size_'+str(size)+'_agent_'+str(ag)+'_seed_'+str(seed)+'_nqubits_'+str(n_qubits)+'_torch'
            print(EXPERIMENT)
            print(file_name)
            print('#######LOAD DOC.'+plots_folder + file_name + '.npy')
            steps = gate_factor * np.load(plots_folder + file_name + '.npy')
            print(plots_folder + file_name + '.npy')
            print(steps)


            steps = np.expand_dims(steps, axis=0)
            # Add loaded list to the array.
            print(ag, steps)
            print('Episodes', episodes)


            step_array = np.append(step_array, steps[:episodes], axis=0)
        print("Number of loaded agents:", num_loaded_agents)

        # remove entries with unsuccessful trials before taking the mean over different agents
        #step_array = step_array[step_array!=gate_factor*max_steps]
        print(step_array)

        # Min, Mean, Std over the different agents
        min_values.append(np.min(step_array[1:, :]))
        mean_array = np.mean(step_array[1:,:], axis=0)
        std_array = np.std(step_array[1:,:],axis=0)


        # Compute mean and std
        mean = np.zeros(int(episodes-trial_avg))
        std = np.zeros(int(episodes-trial_avg))

        if trial_avg == 1:
            mean = mean_array
            std = std_array

        else:
            for episode in range(episodes-trial_avg):
                mean[episode] = np.mean(mean_array[episode : episode + trial_avg])
                std[episode] = np.std(std_array[episode : episode + trial_avg])

        mean_list.append(mean)
        print('MEAN LISZ',mean_list)
        std_list.append(std)
        print('STD LIST',std_list)
    plt.subplot(1, 1, 1)
    X = np.linspace(0, len(mean_list[0]), len(mean_list[0]))
    if type(target_list) == int:
        plt.title('Number of elements in target set |T|='+str(target_list)+', $N_{agents}=$'+str(num_loaded_agents), fontsize=fontsize_labels)
        label = 'Q'
    elif type(qubit_list) == int:
        plt.title('Number of qubits N='+str(qubit_list)+', $N_{agents}=$'+str(num_loaded_agents), fontsize=fontsize_labels)
        label = '|T|'

    for idx in range(len(plot_list)):
        print(idx)
        print(mean_list[idx])
        if trial_avg > 1:
            upper_bound = np.zeros(int(episodes - trial_avg))
            lower_bound = np.zeros(int(episodes - trial_avg))
        else:
            upper_bound = np.zeros(int(episodes))
            lower_bound = np.zeros(int(episodes))
        for iter, elem in enumerate(mean_list[idx]):
            diff_plus = mean_list[idx][iter] + std_list[idx][iter]
            if diff_plus > max_steps:
                diff_plus = max_steps
            upper_bound[iter] = diff_plus
            diff_minus = mean_list[idx][iter] - std_list[idx][iter]
            if diff_minus < min_values[idx]:
                diff_minus = min_values[idx]
            lower_bound[iter] = diff_minus

        print('UPPER', upper_bound)
        print('LOWER', lower_bound)


        plt.fill_between(X, upper_bound, lower_bound, alpha=0.2)
        plt.plot(mean_list[idx], color = plot_color_list[idx], label=label+'='+str(plot_list[idx]))



    plt.ylabel('Average gate count $\\overline{A_g}$', fontsize=fontsize_labels)
    if file_name_start[:2] == 'Te':
        plt.xlabel('Test episode $e$', fontsize=fontsize_labels)
    else:
        plt.xlabel('Training episode $e$', fontsize=fontsize_labels)

    if with_N_sin:
        for iter, N in enumerate(N_sin):
            plt.axhline(y=gate_factor*N, color = plot_color_list[iter], linestyle='-', linewidth=1.,label='$N_{sin}$='+str(gate_factor*N))  # optimal reward
    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    order = [2, 1, 0, 5, 4, 3]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(0.97, 0.98))

    # ticks
    # ticks = np.arange(0, episodes + 1, episodes / 2)
    ticks = [0,2500,]
    print(ticks)
    labels = [str(0), str(int(episodes / 2)), str(episodes)]
    if trial_avg > 1:
        plt.xticks([0, int(episodes / 2), episodes])
    else:
        plt.xticks([0, int(episodes / 2), episodes])
    plt.yticks([0, 1000, 2000])

    # set font size
    axes = plt.gca()
    axes.tick_params(axis='both', which='major', labelsize=fontsize_labels)

    plt.tight_layout()
    plots_save_folder = 'results/plots/'
    pathlib.Path(plots_save_folder).mkdir(parents=True, exist_ok=True)


    plt.savefig(plots_save_folder + 'Plot_'+plot_name +exp + '.png', dpi=200)


def learning_curve_inset(file_name_start, file_end = '', exp = '', with_N_sin = False, N_sin = [], x_type = 'action', max_steps = 1000, qubit_list = [], target_list=[], seed = 0, num_agents = 1, episodes = 15000, trial_avg = 100, diff=100):
    """
    Create plot.
    Args:
    	file_name: Name of the input file.
    	num_agents: Number of agents to be averaged.
    	episodes: Number of episodes

    """
    plot_color_list = ['#1790ff','orange', '#04b605' , '#ec0c10' ,'y','m','k']
    print('Episodes', episodes)
    if x_type == 'gate':
        gate_factor = 2
        max_steps*=2
    else:
        gate_factor = 1

    if type(qubit_list) == list and type(target_list)== int:
        size = target_list
        plot_list = qubit_list

    elif type(qubit_list) == int  and type(target_list)== list:
        n_qubits = qubit_list
        plot_list = target_list

    else:
        raise(NotImplementedError)



    plt.figure(figsize=(8.5, 5.5))
    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.68, 0.3, 0.25, 0.25]
    ax2 = fig.add_axes([left, bottom, width, height])
    print('Episodes', episodes)
    plot_name = file_name_start+'_qubits='+str(qubit_list)+'_size='+str(target_list)
    mean_list = []
    std_list = []
    min_values = []
    ax2_labels = [120]
    fontsize_labels = 10
    for counter in plot_list:
        if type(target_list) == int:
            n_qubits = counter
        elif type(qubit_list) == int:
            size = counter
        EXPERIMENT = str(n_qubits) + 'q_' + str(size)+'t' + exp

        pathlib.Path('results/').mkdir(parents=True, exist_ok=True)
        plots_folder = 'results/' + EXPERIMENT + '/STEPS/'
        print(n_qubits)
        print('Episodes', episodes)
        # Initialize array to stack performances of all agents under consideration.
        step_array = np.zeros((1,episodes),int)
        print('Episodes', episodes)
        print('ary',step_array)
        num_loaded_agents = 0
        for ag in range(0,num_agents):
            # Loaded agent files.
            num_loaded_agents +=1
            file_name = file_name_start+'_size_'+str(size)+'_agent_'+str(ag)+'_seed_'+str(seed)+'_nqubits_'+str(n_qubits)+'_torch'
            print(EXPERIMENT)
            print(file_name)
            steps = gate_factor * np.load(plots_folder + file_name + '.npy')[:episodes]
            print(plots_folder + file_name + '.npy')
            print(steps)


            steps = np.expand_dims(steps, axis=0)
            # Add loaded list to the array.
            print(ag, steps)
            print('Episodes', episodes)


            step_array = np.append(step_array, steps[:episodes], axis=0)
        print("Number of loaded agents:", num_loaded_agents)

        # remove entries with unsuccessful trials before taking the mean over different agents
        #step_array = step_array[step_array!=gate_factor*max_steps]
        print(step_array)

        # Mean over the different agents
        min_values.append(np.min(step_array[1:,:]))
        mean_array = np.mean(step_array[1:,:], axis=0)
        std_array = np.std(step_array[1:,:],axis=0)





        # Compute mean and std
        mean = np.zeros(int(episodes-trial_avg))
        std = np.zeros(int(episodes-trial_avg))

        if trial_avg == 1:
            mean = mean_array
            std = std_array

        else:
            for episode in range(episodes-trial_avg):
                mean[episode] = np.mean(mean_array[episode : episode + trial_avg])
                std[episode] = np.std(std_array[episode : episode + trial_avg])

        mean_list.append(mean)
        print('MEAN LISZ',mean_list)
        std_list.append(std)
        min_x_ax2 = episodes-diff
        max_x_ax2 = episodes

        #ax2_labels.append(np.mean(mean_list[min_x_ax2:max_x_ax2]))

        print('STD LIST',std_list)

    X = np.linspace(0, len(mean_list[0]), len(mean_list[0]))
    if type(target_list) == int:
        ax1.set_title('Number of elements in target set |T|='+str(target_list)+', $N_{agents}=$'+str(num_loaded_agents), fontsize=fontsize_labels)
        label = 'Q'
    elif type(qubit_list) == int:
        ax1.set_title('Number of qubits N='+str(qubit_list)+', $N_{agents}=$'+str(num_loaded_agents), fontsize=fontsize_labels)
        label = '|T|'

    for idx in range(len(plot_list)):
        ax2_labels.append(np.round(np.mean(mean_list[idx][min_x_ax2:]),0))
        print(plot_list, min_values)
        print(idx)
        print(mean_list[idx])
        if trial_avg > 1:
            upper_bound = np.zeros(int(episodes - trial_avg))
            lower_bound = np.zeros(int(episodes - trial_avg))
        else:
            upper_bound = np.zeros(int(episodes))
            lower_bound = np.zeros(int(episodes))
        for iter, elem in enumerate(mean_list[idx]):
            diff_plus = mean_list[idx][iter] + std_list[idx][iter]
            if diff_plus > max_steps:
                diff_plus = max_steps
            upper_bound[iter] = diff_plus
            diff_minus = mean_list[idx][iter] - std_list[idx][iter]
            if diff_minus < min_values[idx]:
                diff_minus = min_values[idx]
            lower_bound[iter] = diff_minus

        print('UPPER', upper_bound)
        print('LOWER', lower_bound)


        ax1.fill_between(X, upper_bound, lower_bound, alpha=0.2)
        ax1.plot(mean_list[idx], color = plot_color_list[idx], label=label+'='+str(plot_list[idx]))
        ax2.fill_between(X, upper_bound, lower_bound, alpha=0.2)
        ax2.plot(mean_list[idx], color = plot_color_list[idx], label=label+'='+str(plot_list[idx]))
        ax2.set_xlim([min_x_ax2, max_x_ax2])
        ax2.set_ylim([0, 160])



    ax1.set_ylabel('Average gate count $\\overline{A_g}$', fontsize=fontsize_labels)
    if file_name_start[:1] == 'Te':
        ax1.set_xlabel('Test episode $e$', fontsize=fontsize_labels)
    else:
        ax1.set_xlabel('Training episode $e$', fontsize=fontsize_labels)

    if with_N_sin:
        #ax2_labels = []
        for iter, N in enumerate(N_sin):
            ax1.axhline(y=gate_factor*N, color = plot_color_list[iter], linestyle='-', linewidth=1.,label='$N_{sin}$='+str(gate_factor*N))  # optimal reward
            ax2.axhline(y=gate_factor * N, color=plot_color_list[iter], linestyle='-', linewidth=1., label='$N_{sin}$=' + str(gate_factor * N))  # optimal reward
            #ax2_labels.append(gate_factor * N)

    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    order = [2,1,0,5,4,3]
    ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(0.97,0.98))

    # ticks
    # ticks = np.arange(0, episodes + 1, episodes / 2)
    ticks = [0,2500,]
    print(ticks)
    labels = [str(0), str(int(episodes / 2)), str(episodes)]
    if trial_avg > 1:
        ax1.set_xticks([0, int(episodes / 2), episodes])
    else:
        ax1.set_xticks([0, int(episodes / 2), episodes])
    ax1.set_yticks([0, 1000, 2000])

    ax2.set_xticks([episodes-diff, episodes])
    ax2.set_yticks(ax2_labels)

    # set font size
    axes = plt.gca()
    ax1.tick_params(axis='both', which='major', labelsize=fontsize_labels)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize_labels-2)



    plt.tight_layout()
    plots_save_folder = 'results/plots/'
    pathlib.Path(plots_save_folder).mkdir(parents=True, exist_ok=True)


    plt.savefig(plots_save_folder + 'Plot_'+plot_name +exp + '.png', dpi=200)


def learning_curve_fails_removed(file_name_start, exp = '', with_N_sin = False, N_sin = [], x_type = 'action', max_steps = 1000, qubit_list = [], target_list=[], seed = 0, num_agents = 1, episodes = 15000, trial_avg = 500):
    """
    Create plot.
    Args:
    	file_name: Name of the input file.
    	num_agents: Number of agents to be averaged.
    	episodes: Number of episodes

    """
    plot_color_list = ['#1790ff','orange', '#04b605' , '#ec0c10' ,'y','m','k']
    fontsize_labels = 12
    print('Episodes', episodes)
    if x_type == 'gate':
        gate_factor = 2
        max_steps*=2
    else:
        gate_factor = 1

    if type(qubit_list) == list and type(target_list)== int:
        size = target_list
        plot_list = qubit_list

    elif type(qubit_list) == int  and type(target_list)== list:
        n_qubits = qubit_list
        plot_list = target_list

    else:
        raise(NotImplementedError)



    plt.figure(figsize=(8.5, 5.5))
    print('Episodes', episodes)
    plot_name = 'Fail_comparision_qubits='+str(qubit_list)+'_size='+str(target_list)
    mean_list = []
    min_values = []
    std_list = []
    min_list = []
    for idx, counter in enumerate(plot_list):
        if type(target_list) == int:
            n_qubits = 7
        elif type(qubit_list) == int:
            size = counter
        EXPERIMENT = str(n_qubits) + 'q_' + str(size)+'t' + exp

        pathlib.Path('results/').mkdir(parents=True, exist_ok=True)
        plots_folder = 'results/' + EXPERIMENT + '/STEPS/'
        print(n_qubits)
        print('Episodes', episodes)
        # Initialize array to stack performances of all agents under consideration.
        step_array = np.zeros((1,episodes),int)
        print('Episodes', episodes)
        print('ary',step_array)
        num_loaded_agents = 0
        for ag in range(0,num_agents):
            # Loaded agent files.
            num_loaded_agents +=1
            file_name = file_name_start[idx]+'_size_'+str(size)+'_agent_'+str(ag)+'_seed_'+str(seed)+'_nqubits_'+str(n_qubits)+'_torch'
            print(EXPERIMENT)
            print(file_name)
            steps = gate_factor * np.load(plots_folder + file_name + '.npy')
            print(plots_folder + file_name + '.npy')
            print(steps)


            steps = np.expand_dims(steps, axis=0)
            # Add loaded list to the array.
            print(ag, steps)
            print('Episodes', episodes)


            step_array = np.append(step_array, steps, axis=0)
        print("Number of loaded agents:", num_loaded_agents)
        # remove entries with unsuccessful trials before taking the mean over different agents
        #step_array = step_array[step_array!=gate_factor*max_steps]
        #print(step_array)

        # Mean over the different agents removing failures
        #print('ARRAY',step_array)
        step_array = step_array.astype(float)
        #collect minima
        min_values.append(np.min(step_array[1:, :]))
        # step_array[step_array==max_steps*gate_factor] = np.nan
        mean_array = np.mean(step_array[1:,:], axis=0)
        #print('MEAN_ARRAY_SHAPE',mean_array.shape)
        std_array = np.std(step_array[1:,:],axis=0)




        for col in range(0,np.shape(step_array)[1]):
                #print('ALL AGENTS',step_array[1:,col])
                if idx == 2 and np.std(step_array[1:,col], axis=0) != 0:
                    step_array_col = step_array[1:, col]
                    copy_step_array_col = step_array_col.copy()

                    #print(step_array_col)

                    copy_step_array_col[step_array_col == float(max_steps)] = np.nan
                    boolarr = np.isnan(copy_step_array_col)
                    if boolarr.sum()<=50:
                        mean_array[col]=np.nanmean(copy_step_array_col, axis=0)
                        std_array[col]=np.nanstd(copy_step_array_col, axis=0)
                    else:
                        mean_array[col] = np.nanmean(step_array_col, axis=0)
                        std_array[col] = np.nanstd(step_array_col, axis=0)


                else:
                    step_array_col = step_array[1:, col]
                    mean_array[col]=np.nanmean(step_array_col, axis=0)
                    #print(np.nanmean(step_array_col, axis=0))
                    #print('step_array_col',step_array_col)
                    std_array[col]=np.nanstd(step_array_col, axis=0)

                    #print('ERROR',std_array[col])
                    #print(np.nanstd(step_array_col, axis=0))

        #print(mean_array[10000:10050])
        #print(std_array[10000:10050])


        # Compute mean and std
        mean = np.zeros(int(episodes-trial_avg))
        std = np.zeros(int(episodes-trial_avg))

        if trial_avg == 1:
            mean = mean_array
            std = std_array

        else:
            for episode in range(episodes-trial_avg):
                mean[episode] = np.mean(mean_array[episode : episode + trial_avg])
                std[episode] = np.std(std_array[episode : episode + trial_avg])

        #print(mean[10000:10050])
        #print(std[10000:10050])




        mean_list.append(mean)
        #print('MEAN LISZ',mean_list)
        std_list.append(std)
        #print('STD LIST',std_list)
    plt.subplot(1, 1, 1)
    X = np.linspace(0, len(mean_list[0]), len(mean_list[0]))
    if type(target_list) == int:
        plt.title('Number of elements in target set |T|='+str(target_list)+', $N_{agents}=$'+str(num_loaded_agents), fontsize=fontsize_labels)
        label = 'Q'
    elif type(qubit_list) == int:
        plt.title('Number of qubits N='+str(qubit_list)+', $N_{agents}=$'+str(num_loaded_agents), fontsize=fontsize_labels)
        label = '|T|'

    for idx in range(len(plot_list)):
        #print(idx)
        #print(mean_list[idx])
        if trial_avg > 1:
            upper_bound = np.zeros(int(episodes - trial_avg))
            lower_bound = np.zeros(int(episodes - trial_avg))
        else:
            upper_bound = np.zeros(int(episodes))
            lower_bound = np.zeros(int(episodes))
        for iter, elem in enumerate(mean_list[idx]):
            diff_plus = mean_list[idx][iter] + std_list[idx][iter]
            if diff_plus > max_steps:
                diff_plus = max_steps
            upper_bound[iter] = diff_plus
            diff_minus = mean_list[idx][iter] - std_list[idx][iter]
            if diff_minus < min_values[idx]:
                diff_minus = min_values[idx]
            lower_bound[iter] = diff_minus

        #print('UPPER', upper_bound)
        #print('LOWER', lower_bound)


        plt.fill_between(X, upper_bound, lower_bound, alpha=0.2)
        plt.plot(mean_list[idx], color = plot_color_list[idx], label=str(plot_list[idx]))



    plt.ylabel('Average gate count $\\overline{A_g}$ \n with a moving average over '+str(trial_avg)+' episodes', fontsize=fontsize_labels)
    plt.xlabel('Episode $e$', fontsize=fontsize_labels)

    if with_N_sin:
        for iter, N in enumerate(N_sin):
            plt.axhline(y=gate_factor*N, color = 'red', linestyle='-', linewidth=1.,label='$N_{sin}$='+str(gate_factor*N))  # optimal reward
    plt.legend()

    # ticks
    # ticks = np.arange(0, episodes + 1, episodes / 2)
    ticks = [0,2500,]
    print(ticks)
    labels = [str(0), str(int(episodes / 2)), str(episodes)]
    if trial_avg > 1:
        plt.xticks([0, int(episodes/2), episodes-trial_avg])
    else:
        plt.xticks([0,  int(episodes/2), episodes- trial_avg])
    plt.yticks([0, 1000, 2000])
    #plt.xlim(2000,3000)

    # set font size
    axes = plt.gca()
    axes.tick_params(axis='both', which='major', labelsize=fontsize_labels)

    plt.tight_layout()
    plots_save_folder = 'results/plots/'
    pathlib.Path(plots_save_folder).mkdir(parents=True, exist_ok=True)

    print(plot_name +exp)
    plt.savefig(plots_save_folder + plot_name +exp + '.png', dpi=200)

def draw_table(table_content):
   #draw table
    print(table_content)
    table_content = np.array(table_content)
    #table_content = table_content.T
    names = ['$A_^1f$','$A_2^f$','$A_3^f$','$A_1^c$','$A_2^c$','$A_3^c$','$N_{sin}$']
    header = names
    print(tabulate(table_content, headers=names, tablefmt='latex'))


    fig = go.Figure(data=[go.Table(
        header=dict(values=header,  # 1st row
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='right'),
        cells=dict(values=table_content,  # 2nd row onwards
                   line_color='darkslategray',
                   fill_color='lightcyan',
                   align='right'))
    ])

    fig.update_layout(width=500, height=500)
    fig.write_image("../DDQN/results/table"+ CFGIDX_LIST[0]+".png",format="PNG")
    fig.show()


def generate_csv(table_dict, config):

    f = open("../DDQN/results/table_"+config+"_5000.csv", "+w")

    header_list = ['seed','naivelen','naivelenOG','actionlena','actionlenb','actionlenc','actionlenatail', 'actionlenbtail', 'actionlenctail']
    numbers = range(len(header_list))
    header_dict = dict(zip(numbers, header_list))
    print(header_dict)
    headers = ";".join(header_list) +';\n'
    #print(headers)
    f.write(headers)
    print(header_list)
    for seed in table_dict.keys():
        line_content = str(seed)
        for header in header_list:
            if header in table_dict[seed].keys() and header[0]=='a':
                line_content+= str(int(np.round(table_dict[seed][header]/table_dict[seed]['naivelenOG']*100,0)))+'\%;'
            elif header in table_dict[seed].keys() and header[0]!='a':
                line_content+= str(table_dict[seed][header])+';'
            else:
                line_content += ';'
        print(line_content)


        f.write(line_content + '\n')
    f.close()


def generate_csv_eval(table_dict, config):

    f = open("../DDQN/results/table_"+config+"_5000.csv", "+w")

    header_list = ['seed','naivelen','naivelenOG','actionlena','actionlenb','actionlenc','actionlenatail', 'actionlenbtail', 'actionlenctail', 'eval']
    numbers = range(len(header_list))
    header_dict = dict(zip(numbers, header_list))
    print(header_dict)
    headers = ";".join(header_list) +';\n'
    #print(headers)
    f.write(headers)
    print(header_list)
    for seed in table_dict.keys():
        line_content = str(seed)
        for header in header_list:
            if header in table_dict[seed].keys() and header[0]=='a':
                line_content+= str(int(np.round(table_dict[seed][header]/table_dict[seed]['naivelenOG']*100,0)))+'\%;'
            elif header in table_dict[seed].keys() and header[0]!='a':
                line_content+= str(table_dict[seed][header])+';'
            else:
                line_content += ';'
        print(line_content)


        f.write(line_content + '\n')
    f.close()






def calculate_singular_solution(qubit = 4, size = 8, x_type = 'action', set_seed_idx=4, exp = 'overlap'):
    """
    Args:

    """
    if x_type == 'gate':
        gate_factor = 2
    else:
        gate_factor = 1

    folder_name = str(qubit)+'q_'+str(size)+'t'+exp

    s_list, t_list, u_list, action_ops = hsc_utils.generate_ops(n_qubits=qubit, method="SA")
    #print(folder_name)
    s_tup = np.load('results/'+folder_name+'/START_STATES/state_'+str(set_seed_idx)+'.npy', allow_pickle=True)
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

    folder_name = str(qubit)+'q_'+str(size)+'t'+exp

    s_list, t_list, u_list, action_ops = hsc_utils.generate_ops(n_qubits=qubit, method="SA")

    s_tup = np.load('results/'+folder_name+'/START_STATES/state_'+str(set_seed_idx)+'.npy', allow_pickle=True)
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


def comparision_plot(n_qubits = 4, target_size = 8, exp='_test', num_agents=5, max_steps=500, file_name_start='', trial_avg = 10, num_runs=1, episodes = 15000):
    """
    Generates a plot that compares a single agent trained on many states with many agents trained separately on a single state

    Args:

    """

    plot_color_list = ['#1790ff','orange', '#04b605' , '#ec0c10' ,'y','m','k']
    x_type = 'gate'
    print('Episodes', episodes)
    if x_type == 'gate':
        gate_factor = 2
        max_steps*=2
    else:
        gate_factor = 1


    plt.figure(figsize=(8.5, 5.5))
    print('Episodes', episodes)
    plot_name = file_name_start+'_qubits='+str(n_qubits)+'_size='+str(target_size)+'_compare'
    mean_list = []
    std_list = []
    EXPERIMENT = str(n_qubits) + 'q_' + str(target_size)+'t' + exp

    pathlib.Path('results/').mkdir(parents=True, exist_ok=True)
    plots_folder = 'RESULTS/' + EXPERIMENT + '/STEPS/'
    plots_folder_all = 'RESULTS/' + EXPERIMENT + '_all/STEPS/'

    # Initialize array to stack performances of all agents under consideration.
    steps_array = np.zeros((1,episodes),int)


    num_loaded_agents = 0
    for ag in range(0,1):
        # Loaded agent files.
        num_loaded_agents +=1
        file_name = file_name_start+'_size_'+str(size)+'_agent_'+str(ag)+'_seed_'+str(ag)+'_nqubits_'+str(n_qubits)+'_torch'
        steps = gate_factor * np.load(plots_folder + file_name + '.npy')
        steps = np.expand_dims(steps, axis=0)
        steps_array = np.append(steps_array, steps, axis=0)


    steps_array = steps_array[1:].transpose()
    steps_array = steps_array.ravel()[:int(episodes*num_agents)]
    ag=0
    file_name = file_name_start[:-6] + '_size_' + str(size) + '_agent_' + str(ag) + '_seed_' + str(ag) + '_nqubits_' + str( n_qubits)
    compare_steps = gate_factor * np.load(plots_folder_all + file_name + '.npy')

    # Mean over the different agents
    if num_runs == 1:
        mean_array = steps_array
        std_array = np.zeros(len(steps_array))

    else:
        mean_array = np.mean(steps_array[1:, :], axis=0)
        std_array = np.std(steps_array[1:, :], axis=0)

    # Compute mean and std
    mean = np.zeros(int(episodes - trial_avg))
    std = np.zeros(int(episodes - trial_avg))

    if trial_avg == 1:
        mean = mean_array
        std = std_array

    else:
        for episode in range(episodes - trial_avg):
            mean[episode] = np.mean(mean_array[episode: episode + trial_avg])
            std[episode] = np.std(std_array[episode: episode + trial_avg])
            #mean_comp[episode] = compare_steps = gate_factor * np.load(plots_folder_all + file_name + '.npy')

    plt.plot(mean, color=plot_color_list[0], label='separate agents')
    #plt.plot(compare_steps, color=plot_color_list[1], label='single agent')
    plt.legend()
    plt.tight_layout()
    plots_save_folder = 'results/plots/'
    pathlib.Path(plots_save_folder).mkdir(parents=True, exist_ok=True)

    plt.savefig(plots_save_folder + plot_name + exp + '.png', dpi=200)

def direct_comparison_plot(exp=[], exp_end='', with_N_sin=False, N_sin=[], x_type='action', max_steps=1000,
                       file_name_starts=[], file_name_ends=[], n_qubits=4, size=5, seed=8, num_agents=1, episodes=15000, trial_avg=100, labels = [], perf_avg=10):
    """
    Generates a plot that compares a single agent trained on many states with many agents trained separately on a single state
    Args:
        file_name: Name of the input file.
        num_agents: Number of agents to be averaged.
        episodes: Number of episodes

    """
    plot_color_list = ['#1790ff', 'orange', '#04b605', '#ec0c10', 'y', 'm', 'k']
    if x_type == 'gate':
        gate_factor = 2
        max_steps *= 2
    else:
        gate_factor = 1
    fontsize_labels = 10

    plt.figure(figsize=(8.5, 5.5))
    plot_name = file_name_starts[0] + '_qubits=' + str(n_qubits) + '_size=' + str(size)
    mean_list = []
    std_list = []
    average_performance = []
    for counter in range(len(exp)):
        EXPERIMENT = str(n_qubits) + 'q_' + str(size) + 't' + exp[counter]+exp_end

        pathlib.Path('results/').mkdir(parents=True, exist_ok=True)
        plots_folder = 'results/' + EXPERIMENT + '/STEPS/'
        # Initialize array to stack performances of all agents under consideration.
        step_array = np.zeros((1, episodes), int)
        num_loaded_agents = 0
        for ag in range(0, num_agents):
            # Loaded agent files.
            num_loaded_agents += 1
            file_name = file_name_starts[counter] + '_size_' + str(size) + '_agent_' + str(ag) + '_seed_' + str(ag) + '_nqubits_' + str(n_qubits) + file_name_ends[counter]
            steps = gate_factor * np.load(plots_folder + file_name + '.npy')


            steps = np.expand_dims(steps, axis=0)
            # Add loaded list to the array.


            step_array = np.append(step_array, steps[:episodes], axis=0)
        print("Number of loaded agents:", num_loaded_agents)

        # remove entries with unsuccessful trials before taking the mean over different agents
        # step_array = step_array[step_array!=gate_factor*max_steps]

        # Mean over the different agents
        mean_array = np.mean(step_array[1:, :], axis=0)
        std_array = np.std(step_array[1:, :], axis=0)
        perf_mean = sum(mean_array[-perf_avg:]) / perf_avg
        perf_std = sum(std_array[-perf_avg:]) / perf_avg
        average_performance.append([perf_mean, perf_std])

        # Compute mean and std
        mean = np.zeros(int(episodes - trial_avg))
        std = np.zeros(int(episodes - trial_avg))

        if trial_avg == 1:
            mean = mean_array
            std = std_array

        else:
            for episode in range(episodes - trial_avg):
                mean[episode] = np.mean(mean_array[episode: episode + trial_avg])
                std[episode] = np.std(std_array[episode: episode + trial_avg])

        mean_list.append(mean)
        std_list.append(std)

    plt.subplot(1, 1, 1)
    X = np.linspace(0, len(mean_list[0]), len(mean_list[0]))

    for idx in range(len(exp)):
        if trial_avg > 1:
            upper_bound = np.zeros(int(episodes - trial_avg))
            lower_bound = np.zeros(int(episodes - trial_avg))
        else:
            upper_bound = np.zeros(int(episodes))
            lower_bound = np.zeros(int(episodes))
        for iter, elem in enumerate(mean_list[idx]):
            diff_plus = mean_list[idx][iter] + std_list[idx][iter]
            if diff_plus > max_steps * gate_factor:
                diff_plus = max_steps * gate_factor
            upper_bound[iter] = diff_plus
            diff_minus = mean_list[idx][iter] - std_list[idx][iter]
            if diff_minus < 0:
                diff_minus = 0
            lower_bound[iter] = diff_minus


        plt.fill_between(X, upper_bound, lower_bound, alpha=0.1, color=plot_color_list[idx])
        plt.plot(mean_list[idx], color=plot_color_list[idx], label= '$|S_{0}|$=' + str(labels[idx]))

    #ylabel_text = 'Number of gates with a moving average over ' + str(trial_avg) + ' episodes'
    ylabel_text = 'Average Gate Count $\overline{A_g}$'
    plt.ylabel(ylabel_text, fontsize=fontsize_labels)

    plt.xlabel('Test episode $e$', fontsize=fontsize_labels)

    if with_N_sin:
        for iter, N in enumerate(N_sin):
            plt.axhline(y=gate_factor * N, color=plot_color_list[iter], linestyle='-', linewidth=1.,
                        label='$N_{sin}$=' + str(gate_factor * N))  # optimal reward
    plt.legend()

    # ticks
    # ticks = np.arange(0, episodes + 1, episodes / 2)
    ticks = [0, 2500, ]
    labels = [str(0), str(int(episodes / 2)), str(episodes)]
    if trial_avg > 1:
        plt.xticks([0, int(episodes / 2), episodes])
    else:
        plt.xticks([0, int(episodes / 2), episodes])
    plt.yticks([0, 1000, 2000])

    # set font size
    axes = plt.gca()
    axes.tick_params(axis='both', which='major', labelsize=fontsize_labels)

    plt.tight_layout()
    plots_save_folder = 'results/plots/'
    pathlib.Path(plots_save_folder).mkdir(parents=True, exist_ok=True)
    file_name = 'Plot_' + plot_name + exp[0] + '.png'
    plt.savefig(plots_save_folder + file_name, dpi=200)


def direct_comparison_table(exp=[], with_N_sin=False, N_sin=[], x_type='action', max_steps=1000,
                       file_name_starts=[], file_name_ends=[], n_qubits=4, size=5, seed=8, num_agents=1, episodes=15000, trial_avg=100, labels = [], perf_avg=10):
    """
    Generates a plot that compares a single agent trained on many states with many agents trained separately on a single state
    Args:
        file_name: Name of the input file.
        num_agents: Number of agents to be averaged.
        episodes: Number of episodes

    """
    if x_type == 'gate':
        gate_factor = 2
        max_steps *= 2
    else:
        gate_factor = 1


    plt.figure(figsize=(8.5, 5.5))
    config = '_qubits=' + str(n_qubits) + '_size=' + str(size)
    plot_name = file_name_starts[0] + config
    mean_list = []
    std_list = []
    average_performance_dict = {}
    for counter in range(len(exp)):
        EXPERIMENT = str(n_qubits) + 'q_' + str(size) + 't' + exp[counter]

        pathlib.Path('results/').mkdir(parents=True, exist_ok=True)
        plots_folder = 'results/' + EXPERIMENT + '/STEPS/'

        # Initialize array to stack performances of all agents under consideration.
        step_array = np.zeros((1, episodes), int)

        num_loaded_agents = 0
        for ag in range(0, num_agents):
            # Loaded agent files.
            num_loaded_agents += 1
            file_name = file_name_starts[counter] + '_size_' + str(size) + '_agent_' + str(ag) + '_seed_' + str(ag) + '_nqubits_' + str(n_qubits) + file_name_ends[counter]
            steps = gate_factor * np.load(plots_folder + file_name + '.npy')
            steps = np.expand_dims(steps, axis=0)
            # Add loaded list to the array.

            step_array = np.append(step_array, steps[:episodes], axis=0)


        # Mean over the different agents
        mean_array = np.mean(step_array[1:, :], axis=0)
        std_array = np.std(step_array[1:, :], axis=0)
        perf_mean = sum(mean_array[-perf_avg:]) / perf_avg
        perf_std = sum(std_array[-perf_avg:]) / perf_avg
        average_performance_dict.update({labels[counter]:{'mean':int(np.round(perf_mean,0)), 'std':int(np.round(perf_std,0))}})



def non_det_direct_comparison_table(exp=[], file_end ='', with_N_sin=False, N_sin=[], x_type='action', max_steps=1000,
                                file_name_starts=[], file_name_ends=[], n_qubits=4, size=5, seed=8, num_agents=1,
                                episodes=15000, trial_avg=100, labels=[], perf_avg=10):
        """
        Generates a plot that compares a single agent trained on many states with many agents trained separately on a single state
        Args:
            file_name: Name of the input file.
            num_agents: Number of agents to be averaged.
            episodes: Number of episodes
        """
        if x_type == 'gate':
            gate_factor = 2
            max_steps *= 2
        else:
            gate_factor = 1

        plt.figure(figsize=(8.5, 5.5))
        plots_save_folder = 'results/plots/'
        config = '_qubits=' + str(n_qubits) + '_size=' + str(size)
        plot_name = file_name_starts[0] + config
        mean_list = []
        std_list = []
        average_performance_dict = {}
        start_time = 749000
        for counter in range(0,len(exp)-1):
            EXPERIMENT = str(n_qubits) + 'q_' + str(size) + 't' + exp[counter]+file_end

            pathlib.Path('results/').mkdir(parents=True, exist_ok=True)
            plots_folder = 'results/' + EXPERIMENT + '/STEPS/'

            # Initialize array to stack performances of all agents under consideration.
            test_states = 1000
            step_array = np.zeros((1, test_states), int)

            num_loaded_agents = 0
            for ag in range(0, num_agents):
                # Loaded agent files.
                num_loaded_agents += 1

                file_name = file_name_starts[counter] + '_size_' + str(size) + '_agent_' + str(ag) + '_seed_' + str(
                    ag) + '_nqubits_' + str(n_qubits) + file_name_ends[counter]

                steps = np.load(plots_folder + file_name + '.npy', allow_pickle=True).item()
                steps = gate_factor * steps[14999]


                steps = np.expand_dims(steps, axis=0)
                # Add loaded list to the array.

                step_array = np.append(step_array, steps, axis=0)

            print("Number of loaded agents:", num_loaded_agents)



            # Mean over the different agents

            mean_array = np.mean(step_array[1:, :], axis=0)
            std_array = np.std(step_array[1:, :], axis=0)
            num_results_ag = 50


        for counter in [-1]:
            EXPERIMENT = str(n_qubits) + 'q_' + str(size) + 't' + exp[counter]

            pathlib.Path('results/').mkdir(parents=True, exist_ok=True)
            plots_folder = 'results/' + EXPERIMENT + '/STEPS/'

            # Initialize array to stack performances of all agents under consideration.
            episodes = 1000
            comp_array = np.zeros((1, episodes), int)

            num_loaded_agents = 0
            start_time = 15000
            for ag in range(0, num_results_ag):
                # Loaded agent files.
                num_loaded_agents += 1
                file_name = file_name_starts[counter] + '_size_' + str(size) + '_agent_' + str(ag) + '_seed_' + str(
                    ag) + '_nqubits_' + str(n_qubits) + file_name_ends[counter]
                steps = gate_factor * np.load(plots_folder + file_name + '.npy')
                steps = np.expand_dims(steps[start_time-episodes:start_time], axis=0)
                # Add loaded list to the array.
                comp_array = np.append(comp_array, steps, axis=0)

            comp_array = np.mean(comp_array[1:], axis=1)
            print("Number of loaded agents:", num_loaded_agents)

            # remove entries with unsuccessful trials before taking the mean over different agents
            # step_array = step_array[step_array!=gate_factor*max_steps]


            num_single_agents = 25
            ratio_dict = {'below': np.array([0] * num_single_agents), 'small': np.array([0] * num_single_agents), 'medium': np.array([0] * num_single_agents),'large': np.array([0] * num_single_agents)}

            for target in range(num_single_agents):
                plt.clf()
                ratio_list = []

                for idx, elem in enumerate(step_array[1:][target][:50]):
                    #print(elem, comp_array[idx])
                    ratio = (elem)*100/comp_array[idx]

                    if ratio <= 100.0:
                        ratio_dict['below'][target]+=1
                    elif ratio > 100.0 and ratio <=115.0:
                        ratio_dict['small'][target]+=1
                        #print(ratio_dict['small'])
                    elif ratio > 115.0 and ratio <= 130.0:
                        ratio_dict['medium'][target]+=1
                        #print(ratio_dict['medium'])
                    elif ratio > 130.0:
                        ratio_dict['large'][target]+=1
                        #print(ratio_dict['large'])
                    ratio_list.append(ratio)


            plt.savefig(plots_save_folder + 'Plot_histo_mean.png', dpi=200)

            num_states = 1000

            N_sin_array = np.array([])
            for idx in range(num_states):
                N_sin = calculate_singular_solution(qubit=n_qubits, size=size, x_type=x_type, set_seed_idx=idx, exp='_single')
                N_sin_array = np.append(N_sin_array,N_sin)



            mean_comp_single = np.round(comp_array/N_sin_array[:50]*100,0)



            data_dict = {}
            for percent in [40,50,60,70,80,90,100]:
                counts = np.array([])
                for ag in range(1,25):
                    one_comp = np.round((step_array[ag]/N_sin_array*100), 0)
                    counter = 0
                    for entry in one_comp:
                        if entry < percent:
                            counter += 1

                    counts = np.append(counts, counter)
                avg_count = np.mean(counts)
                std_count = np.std(counts)
                data_dict.update({percent:{'avg_count':avg_count,'std_count':std_count}})


            data_dict = {}
            for percent in [45,55,65,75,85,95,105]:
                counts = np.array([])
                for ag in range(1,25):
                    one_comp = np.round((step_array[ag]/N_sin_array*100), 0)
                    counter = 0
                    for entry in one_comp:
                        if entry < percent:
                            counter += 1
                    counts = np.append(counts, counter)
                avg_count = np.mean(counts)
                std_count = np.std(counts)
                data_dict.update({percent:{'avg_count':avg_count,'std_count':std_count}})
            print(data_dict)







def shortest_vs_average_table(exp=[], with_N_sin=False, N_sin=[], x_type='action', max_steps=1000,
                       file_name_starts=[], file_name_ends=[], n_qubits=4, size=5, seed=8, num_agents=1, episodes=15000, trial_avg=100, labels = [], perf_avg=10, data = []):
    """
    Generates a plot that compares a single agent trained on many states with many agents trained separately on a single state
    Args:
        file_name: Name of the input file.
        num_agents: Number of agents to be averaged.
        episodes: Number of episodes

    """
    if x_type == 'gate':
        gate_factor = 2
        max_steps *= 2
    else:
        gate_factor = 1


    plt.figure(figsize=(8.5, 5.5))


    average_performance_dict = {}
    shortest_len_dict = {}
    for n_qubit in n_qubits:
        config = str(n_qubit) + 'q_' + str(size) + 't'
        EXPERIMENT = config + exp[0]

        pathlib.Path('results/').mkdir(parents=True, exist_ok=True)
        plots_folder = 'results/' + EXPERIMENT + '/STEPS/'

        # Initialize array to stack performances of all agents under consideration.
        step_array = np.zeros((1, episodes), int)

        num_loaded_agents = 0
        for ag in range(0, num_agents):
            # Loaded agent files.
            num_loaded_agents += 1
            file_name = file_name_starts[0] + '_size_' + str(size) + '_agent_' + str(ag) + '_seed_' + str(0) + '_nqubits_' + str(n_qubit) + file_name_ends[0]

            steps = gate_factor * np.load(plots_folder + file_name + '.npy')
            steps = np.expand_dims(steps, axis=0)



            step_array = np.append(step_array, steps[:episodes], axis=0)
        print("Number of loaded agents:", num_loaded_agents)



        # Mean over the different agents
        mean_array = np.mean(step_array[1:, :], axis=0)
        std_array = np.std(step_array[1:, :], axis=0)
        perf_mean = sum(mean_array[-perf_avg:]) / perf_avg
        perf_std = sum(std_array[-perf_avg:]) / perf_avg
        average_performance_dict.update({n_qubit:{'mean':int(np.round(perf_mean,0)), 'std':int(np.round(perf_std,0))}})

        table_path = "../DDQN/results/" + config + "_table/min_num_gates.npy"
        data = np.load(table_path, allow_pickle=True).item()
        len_list = []
        for entry in data[0]:
            if entry[-4:] != 'tail':

                len_list.append(data[0][entry])


        shortest_len_dict.update({n_qubit: {'short': int(np.round(min(len_list), 0))}})

    f = open("../DDQN/results/short_vs_avg_table" + config + ".csv", "+w")

    header_list = ['numstates','len','naivelen']
    numbers = range(len(header_list))
    header_dict = dict(zip(numbers, header_list))
    headers = ";".join(header_list) + '\n'
    f.write(headers)
    for qubit in qubits:
        line_content = str(qubit)+';'
        line_content += str(average_performance_dict[qubit]['mean']) + '\pm' + str(average_performance_dict[qubit]['std']) + ';'
        line_content+=str(shortest_len_dict[qubit]['short'])+';'

        f.write(line_content + '\n')
    f.close()





if __name__ == "__main__":
    NUM_CONFIG = 0
    NUM_AGENTS = 50
    EPISODES = 15000
    MAX_STEPS = 1000

    SEED = 0
    #choose whether to plot the number of actions or the number of gates, these separate by a factor of 2
    X_TYPE = 'gate'

    SINGLE_PLOT = False
    MULTI_PLOT = True
    MULTI_TARGET = False
    TABLE = False
    FAILS = False
    COMPARE = False
    DIRECT_COMPARE = False
    CALC_NAIVE = False
    DIRECT_COMPARE_TABLE = False
    SHORT_VS_AVG = False
    NON_DET = False
    COUNT_STEPS = False
    TABLE_CUTOFF = False

    if SINGLE_PLOT:
        EXP = ''
        file_name_start = 'Time_steps_HSC_DDQN_model'

        learning_curve(file_name_start, exp=EXP, with_N_sin=True, N_sin=[], x_type=X_TYPE, max_steps=MAX_STEPS,
                       qubit_list=4, target_list=[5], seed=SEED, num_agents=NUM_AGENTS, episodes=EPISODES,
                       trial_avg=1)

        file_name_start = 'Test_time_steps_HSC_DDQN_model'

        learning_curve(file_name_start, exp=EXP, with_N_sin=True, N_sin=[], x_type=X_TYPE, max_steps=MAX_STEPS,
                       qubit_list=3, target_list=[4], seed=SEED, num_agents=NUM_AGENTS, episodes=EPISODES,
                       trial_avg=1)

    if FAILS:
        EXP = '_plot'
        N_sin_qubit = []
        for qubit in [7]:
            N_sin_qubit.append(calculate_singular_solution(qubit=qubit, size=8, set_seed_idx=SEED, exp='_plot'))


        file_name_start = ['Time_steps_HSC_DDQN_model','Test_time_steps_HSC_DDQN_model', 'Test_time_steps_HSC_DDQN_model']

        # learning_curve(file_name_start, exp = EXP, with_N_sin = True, N_sin = N_sin_size, x_type = X_TYPE, max_steps= MAX_STEPS, qubit_list = 4, target_list = [8,12,16], seed = SEED, num_agents = NUM_AGENTS, episodes= EPISODES,trial_avg = 100)

        learning_curve_fails_removed(file_name_start, exp=EXP, with_N_sin=True, N_sin=N_sin_qubit, x_type=X_TYPE, max_steps=MAX_STEPS,
                       qubit_list=['epsilon-greedy','deterministic','deterministic - no fails'], target_list=8, seed=SEED, num_agents=NUM_AGENTS, episodes=EPISODES,
                       trial_avg=1000)


    if MULTI_PLOT:
        EXP = '_plot'
        N_sin_qubit = []
        qubit_list = [4,5,7]
        #the x-length of the inset plot
        DIFF = 1000
        for qubit in qubit_list:
            N_sin_qubit.append(calculate_singular_solution(qubit=qubit, size=8, set_seed_idx=SEED, exp='_plot'))

        #file_name_start = 'Time_steps_HSC_DDQN_model'

        #learning_curve_inset(file_name_start, exp=EXP, with_N_sin=True, N_sin=N_sin_qubit, x_type=X_TYPE, max_steps=MAX_STEPS,qubit_list=qubit_list, target_list=8, seed=SEED, num_agents=NUM_AGENTS, episodes=EPISODES, trial_avg=1, diff=DIFF)

        file_name_start = 'Test_time_steps_HSC_DDQN_model'

        #learning_curve(file_name_start, exp = EXP, with_N_sin = True, N_sin = N_sin_size, x_type = X_TYPE, max_steps= MAX_STEPS, qubit_list = 4, target_list = [8,12,16], seed = SEED, num_agents = NUM_AGENTS, episodes= EPISODES,trial_avg = 100)

        learning_curve(file_name_start, exp = EXP, with_N_sin = True, N_sin = N_sin_qubit, x_type = X_TYPE, max_steps= MAX_STEPS, qubit_list = qubit_list, target_list = 8, seed = SEED, num_agents = NUM_AGENTS, episodes= EPISODES,trial_avg = 1)

    if MULTI_TARGET:
        EXP = '_plot_15000'
        N_sin_qubit = []
        qubit = 4
        size_list = [8,12,16]
        EPISODES  = 15000
        #the x-length of the inset plot
        DIFF = 1000
        for size in size_list:
            N_sin_qubit.append(calculate_singular_solution(qubit=qubit, size=size, set_seed_idx=SEED, exp=EXP))


        #file_name_start = 'Time_steps_HSC_DDQN_model'

        #learning_curve(file_name_start, exp=EXP, with_N_sin=True, N_sin=N_sin_qubit, x_type=X_TYPE, max_steps=MAX_STEPS,qubit_list=qubit_list, target_list=8, seed=SEED, num_agents=NUM_AGENTS, episodes=EPISODES, trial_avg=1)

        file_name_start = 'Time_steps_HSC_DDQN_model'

        #learning_curve(file_name_start, exp = EXP, with_N_sin = True, N_sin = N_sin_size, x_type = X_TYPE, max_steps= MAX_STEPS, qubit_list = 4, target_list = [8,12,16], seed = SEED, num_agents = NUM_AGENTS, episodes= EPISODES,trial_avg = 100)

        learning_curve_inset(file_name_start, exp = EXP, with_N_sin = True, N_sin = N_sin_qubit, x_type = X_TYPE, max_steps= MAX_STEPS, qubit_list = qubit, target_list = size_list, seed = SEED, num_agents = NUM_AGENTS, episodes= EPISODES,trial_avg = 1, diff=DIFF)

    if TABLE:
        qubit = 4
        size = 8
        seq_average_over=100
        exp = '_cutoff100'
        CONFIG = str(qubit)+'q_'+str(size)+'t'+exp
        table_path = "../DDQN/results/" + CONFIG + "/min_num_gates.npy"
        data = np.load(table_path, allow_pickle=True).item()
        #LOAD singluar solution and add it to the table:
        N_sin_qubit = []
        for idx in range(5):
            N_sin = calculate_singular_solution(qubit=qubit, size=size, x_type = X_TYPE, set_seed_idx=idx, exp= exp)
            N_sin_qubit.append(N_sin)
            data[idx].update({"naivelenOG":N_sin})
        for idx in range(5):
            N_seq = calculate_sequential_solution(qubit=qubit, size=size, x_type = X_TYPE, set_seed_idx=idx, exp= exp, average_over=seq_average_over)
            data[idx].update({"naivelen":N_seq})
        print(data)
        generate_csv(data, CONFIG)

    if TABLE_CUTOFF:
        qubit = 4
        size = 8
        seq_average_over=100
        exp = '_cutoff100'
        CONFIG = str(qubit)+'q_'+str(size)+'t'+exp
        table_path = "../DDQN/results/" + CONFIG + "/min_num_gates_5000.npy"
        data = np.load(table_path, allow_pickle=True).item()
        #LOAD singluar solution and add it to the table:
        N_sin_qubit = []
        for seed in range(5):
            total_steps = 0
            for ag in range(1):
                    seq_average_over=100
                    exp = '_cutoff100'
                    EXPERIMENT = str(qubit) + 'q_' + str(size) + 't' + exp
                    plots_folder = 'results/' + EXPERIMENT + '/STEPS/'
                    CONFIG = str(qubit)+'q_'+str(size)+'t'+exp
                    file_name_start = "Time_steps_HSC_DDQN_model"
                    file_name = file_name_start + '_size_' + str(size) + '_agent_' + str(ag) + '_seed_' + str(
                        seed) + '_nqubits_' + str(qubit) + '_torch'
                    #print(plots_folder)
                    steps = np.load(plots_folder + file_name + '.npy')
                    num_steps = 0
                    for step in steps[:5000]:
                       # print(step)
                        num_steps += step
                    total_steps+=num_steps
                    print(num_steps)
            data[seed].update({"eval":total_steps})
            print('total', total_steps)
        for idx in range(5):
            N_sin = calculate_singular_solution(qubit=qubit, size=size, x_type = X_TYPE, set_seed_idx=idx, exp= exp)
            N_sin_qubit.append(N_sin)
            data[idx].update({"naivelenOG":N_sin})
        for idx in range(5):
            N_seq = calculate_sequential_solution(qubit=qubit, size=size, x_type = X_TYPE, set_seed_idx=idx, exp= exp, average_over=seq_average_over)
            data[idx].update({"naivelen":N_seq})
        print(data)
        generate_csv_eval(data, CONFIG)


    if COUNT_STEPS:
        qubit = 4
        size = 8
        for seed in range(5):
            total_steps = 0
            for ag in range(3):
                seq_average_over=100
                exp = '_cutoff100'
                EXPERIMENT = str(qubit) + 'q_' + str(size) + 't' + exp
                plots_folder = 'results/' + EXPERIMENT + '/STEPS/'
                CONFIG = str(qubit)+'q_'+str(size)+'t'+exp
                file_name_start = "Time_steps_HSC_DDQN_model"
                file_name = file_name_start + '_size_' + str(size) + '_agent_' + str(ag) + '_seed_' + str(
                    seed) + '_nqubits_' + str(qubit) + '_torch'
                #print(plots_folder)
                steps = np.load(plots_folder + file_name + '.npy')
                num_steps = 0
                for step in steps[:5000]:
                   # print(step)
                    num_steps += step
                total_steps+=num_steps
                print(num_steps)
            print('total', total_steps)

        print(num_steps)


    if COMPARE:
        qubit = 4
        size = 8
        exp = '_single'
        episodes = 10000
        trial_avg = 1
        num_agents = 1

        file_name_start = 'Time_steps_HSC_DDQN_model'
        comparision_plot(n_qubits = qubit, target_size = size, file_name_start=file_name_start, trial_avg=trial_avg, exp=exp, episodes=episodes, num_agents=num_agents)


    if CALC_NAIVE:
        N_sin_qubit = []
        qubit = 4
        size = 8
        CONFIG = str(qubit)+'q_'+str(size)+'t_'+'single'
        num_states = 50
        average_len = np.array([])

        for idx in range(num_states):
             N_sin = calculate_singular_solution(qubit=qubit, size=8, x_type = X_TYPE, set_seed_idx=idx, exp='_single')
             average_len = np.append(average_len,N_sin)
             print(idx,N_sin)
        print(np.mean(average_len),np.std(average_len))
        print('AVERAGE',np.mean(average_len),np.std(average_len))

        #array = np.load('results/'+CONFIG+'/min_num_gates.npy')
        #print(array)


    if DIRECT_COMPARE:
        qubit = 4
        size = 8
        labels = [1000,100,50,1]
        exp = ['_single_'+str(labels[0]),'_single_'+str(labels[1]),'_single_'+str(labels[2]),'_single']

        print(labels)
        file_name_starts = ['Time_steps_HSC_DDQN','Time_steps_HSC_DDQN','Time_steps_HSC_DDQN','Time_steps_HSC_DDQN_model']
        file_name_ends = ['','','','_torch']
        episodes = 5000
        trial_avg = 1
        num_agents = 40
        #display the number of gates
        x_type = 'gate'
        perf_avg = 1000


        direct_comparison_plot(x_type=x_type, n_qubits = qubit, size = size, file_name_starts=file_name_starts, file_name_ends=file_name_ends, trial_avg=trial_avg, exp=exp, episodes=episodes, num_agents=num_agents, labels = labels, perf_avg=perf_avg)


    if DIRECT_COMPARE_TABLE:
        qubit = 4
        size = 8
        labels = [1000,100,50,1]
        exp = ['_single_'+str(labels[0]),'_single_'+str(labels[1]),'_single_'+str(labels[2]),'_single']

        print(labels)
        file_name_starts = ['Time_steps_HSC_DDQN','Time_steps_HSC_DDQN','Time_steps_HSC_DDQN','Time_steps_HSC_DDQN_model']
        file_name_ends = ['','','','_torch']
        episodes = 15000
        trial_avg = 1
        num_agents = 40
        #display the number of gates
        x_type = 'gate'
        perf_avg = 1000


        direct_comparison_table(x_type=x_type, n_qubits = qubit, size = size, file_name_starts=file_name_starts, file_name_ends=file_name_ends, trial_avg=trial_avg, exp=exp, episodes=episodes, num_agents=num_agents, labels = labels, perf_avg=perf_avg)



    if SHORT_VS_AVG:
        qubits = [4,5,7]
        size = 8
        labels = ['']
        exp = ['_plot']

        print(labels)
        file_name_starts = ['Time_steps_HSC_DDQN_model']
        file_name_ends = ['_torch']
        episodes = 15000
        trial_avg = 1
        num_agents = 40
        #display the number of gates
        x_type = 'gate'
        perf_avg = 10


        shortest_vs_average_table(x_type=x_type, n_qubits = qubits, size = size, file_name_starts=file_name_starts, file_name_ends=file_name_ends, trial_avg=trial_avg, exp=exp, episodes=episodes, num_agents=num_agents, labels = labels, perf_avg=perf_avg)



    if NON_DET:
        qubit = 4
        size = 8
        labels = [1000, 1]
        exp = ['_single_'+str(labels[0]),'_single']

        print(labels)
        file_name_starts = ['Test_time_steps_HSC_DDQN','Time_steps_HSC_DDQN_model']
        file_end ='_det'
        file_name_ends = ['','_torch']
        episodes = int(1000)
        trial_avg = 1
        num_agents = 25
        #display the number of gates
        x_type = 'gate'
        perf_avg = 1000


        non_det_direct_comparison_table(x_type=x_type, file_end = file_end, n_qubits=qubit, size=size, file_name_starts=file_name_starts,
                                file_name_ends=file_name_ends, trial_avg=trial_avg, exp=exp, episodes=episodes,
                                num_agents=num_agents, labels=labels, perf_avg=perf_avg)
