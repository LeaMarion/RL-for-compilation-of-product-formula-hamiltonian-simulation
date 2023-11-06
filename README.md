# RL-for-compilation-of-product-formula-hamiltonian-simulation

This code accompanies the paper "Compilation of product-formula Hamiltonian simulation via reinforcement learning". 

In situations where the order isn't fixed by the goal of minimizing approximation errors, we propose selecting the order to enhance compilation for a specific quantum architecture. This introduces a novel compilation problem: "order-agnostic quantum circuit compilation", which we proof to be NP-hard in the worst-case.

Since finding an exact solution is computationally challenging, we turn to heuristic optimization methods. Our focus is on reinforcement learning (RL), in particular, Double Deep Q Learning (DDQN), due to the sequential nature of the compilation process, and we compare its performance to two other methods: simulated annealing (SA) and Monte Carlo tree search (MCTS). Our results reveal that RL, in terms of gate count, demonstrates an improvement of approximately 12% over the second-best method, MCTS, and about 50% over a basic heuristic.
Furthermore, we assess the ability of reinforcement learning to generalize across different instances of the compilation problem and find that a single learner can effectively tackle entire problem families. This study highlights the potential of machine learning techniques in providing valuable assistance for order-agnostic quantum compilation tasks.



## Runfiles

### Learning Performance

To reproduce the results for Fig. 6., run the command line below which automatically loads the configuration file with the corresponding hyperparameters.

```python HSC_DDQN_cluster --par_idx {0..49} --num_states 1 --num_agents 50 --config-file "4q_{8-12-16}t_plot"```

To reproduce the results for Fig 7. and Fig. 9, run the command line below which automatically loads the configuration file with the corresponding hyperparameters.

```python HSC_DDQN_cluster --par_idx {0..49} --num_states 1 --num_agents 50 --config-file "{4-5-7}q_8t_plot"```

To reproduce Fig. 10, run the command line below which automatically loads the configuration file with the corresponding hyperparameters.

```python HSC_DDQN_cluster --par_idx {0..49} --config-file "7q_8t_plot"```


### Shortest Solution

To reproduce the results for Table 1, run the command line below which automatically loads the configuration file with the corresponding hyperparameters.

```python HSC_DDQN_cluster --par_idx {0..14} --config-file "4q_{8-12-16}t_table"```

To reproduce the results for Table 2, run the command line below which automatically loads the configuration file with the corresponding hyperparameters.

```python HSC_DDQN_cluster --par_idx {0..14} --config-file "{4-5-6-7}q_8t_table"```



### Comparison

To reproduce the results for DDQN in Table 3, run the command lines below which automatically loads the configuration file with the corresponding hyperparameters.

```python HSC_DDQN_cluster --par_idx {0..15} --config-file "4q_8t_cutoff100"```


To reproduce the results for Table 3 and Table 7, run the command line below which automatically loads the configuration file with the corresponding hyperparameters.

```python HSC_SA --par_idx {0..20} --config-file "{sa_naiveinit_1Mio,sa_emptyinit_1Mio}"```


To reproduce the results for Table 3 and Table 8, run the command line below which automatically loads the configuration file with the corresponding hyperparameters.

```python MCTSrun --par_idx {0..5}```


### Generalization

To reproduce the results for Fig. 8, as well as, Table 4, 5, and 6, run the command line below which automatically loads the configuration file with the corresponding hyperparameters.

```python HSC_DDQN_cluster --par_idx {0-40} --config-file "config_4q_8t_single_{1-50-100-1000}"```



## Directories

```DDQN```

This directory contains the code and the data analysis files for the DDQN agent and stores the results of the DDQN experiments.  


```JSON```

This directory contains all files with the parameter configuration to reproduce the results.


```MCTS```

This directory contains the code and the data analysis files for MCTS and stores the results of the MCTS experiments.  


```processing```

This directory contains all the code for processing the raw DDQN, MCTS and SA data. 


```SA```

This directory contains the code and the data analysis files for SA and stores the results of the SA experiments.  

```shared```

This directory contains the code for the environment shared between DDDQN, MCTS and SA.   


