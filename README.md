# Neur2SP
Implementation of "Neur2SP: Neural Two-Stage Stochastic Programming" (https://arxiv.org/pdf/2205.12006.pdf)


## How to reproduce the experimental results

To reproduce the results from the paper, one can either do this sequentially or in parallel.  We provide a single file (`runner.py`) that can accommodate both, although the parallel commands may vary depending on the compute infrastructure.  `runner.py` is also able to only run a subset of the experiments, so please see the usage for more detail. 


### Sequential
The easiest way to reproduce the experiments is to run them sequentially.  For example, to reproduce the results for a single problem run the following:

```
python runner.py --problems {PROBLEMS} --run_all 1 --n_cpus {N_CPUS}
```

where `PROBLEMS` can be any subset of `{cflp_10_10, cflp_25_25, cflp_50_50, sslp_5_25, sslp_10_50, sslp_15_45, ip_b_E, ip_b_H, ip_i_E, ip_i_H, pp}`.  This may take several days as the baselines are run on multiple instances for a problem, each of which takes several hours.  As such, the use of parallel computing is advantageous, especially for CFLP and SSLP as they have a larger number of instances to evaluate on.


### Parallel
To run the experiments in parallel, we also provide the functionality to obtain batches of commands that can be run in parallel.  In our experiments, these were run on Compute Canada with META, however, utilization to other computing clusters may require minor modifications to run the set of parallelizable commands.  Each command below generates a `table.dat` file containing a list of commands to execute. For this reason, one must execute a single command below and run the parallel set of jobs, then when all parallel jobs are completed, the next command can be executed.  To generate the batch commands for each step, the following can be done:
```
python runner.py --problems {PROBLEMS} --run_dg_inst 1 --as_dat 1                                         
python runner.py --problems {PROBLEMS} --run_dg_p 1 --run_dg_e 1 --as_dat 1 --n_cpus {N_CPUS}           
python runner.py --problems {PROBLEMS} --train_lr 1 --train_nn_p 1 --train_nn_e 1 --as_dat 1              
python runner.py --problems {PROBLEMS} --get_best_nn_p_model 1 --get_best_nn_e_model 1 --as_dat 1        
python runner.py --problems {PROBLEMS} --eval_lr 1 --eval_nn_p 1 --eval_nn_e 1 --eval_nn_p 1 --as_dat 1 --n_cpus {N_CPUS}
```

### Generating the tables

After `runner.py` is executed, all experimental results can be collected to a single file in `.pkl` file (`data/results.pkl`) with the following command:
```
 python -m nsp.scripts.collect_all_results
```
Once the results are collected the Latex tables from the paper can be generated using `notebook/PaperResults.ipynb`.  This can also be done for a subset of the problems as the values will be populated with `NaN`.



## A note on SSLP instances
To run experiments on SSLP, one must download the SIPLIB instances from https://www2.isye.gatech.edu/~sahmed/siplib/sslp/sslp.zip, unzip them, and move the instances to `data/sslp/sslp_data/`.




## Reference

Please cite our work if you find our code/paper useful to your work.

```
  @article{dumouchelle2022neur2sp,
    title={Neur2{SP}: Neural Two-Stage Stochastic Programming},
    author={Dumouchelle, Justin and Patel, Rahul and Khalil, Elias B and Bodur, Merve},
    journal={Advances in Neural Information Processing Systems},
    volume={35},
    year={2022}
  }
```




