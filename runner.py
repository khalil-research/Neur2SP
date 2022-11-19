import argparse
import pickle as pkl
import shutil
import subprocess

import numpy as np

import sys

def get_nn_p_function_call(args, problem):
    """  Gets training function call for best config 
         as per the 100 random search runs NN-P.  
    """
    from nn_params import nn_p_params
    params = nn_p_params[problem]

    # get hidden dimension as nargs='+'
    hds = list(map(lambda x: str(x), params["hidden_dims"]))
    hds = " ".join(hds)

    cmd =  f'python -m nsp.scripts.train_model --problem {problem} --model nn_p '
    cmd += f'--hidden_dims {hds} '
    cmd += f'--lr {params["lr"]} '
    cmd += f'--dropout {params["dropout"]} '
    cmd += f'--optimizer {params["optimizer_type"]} '
    cmd += f'--batch_size {params["batch_size"]} '
    cmd += f'--loss_fn {params["loss_fn"]} '
    cmd += f'--wt_lasso {params["wt_lasso"]} '
    cmd += f'--wt_ridge {params["wt_ridge"]} '
    cmd += f'--log_freq {params["log_freq"]} '
    cmd += f'--n_epochs {params["n_epochs"]} '
    cmd += f'--use_wandb {params["use_wandb"]} '

    return cmd


def get_nn_e_function_call(args, problem):
    """  Gets training function call for best config 
         as per the 100 random search runs NN-E.  
    """
    from nn_params import nn_e_params
    params = nn_e_params[problem]

    cmd =  f'python -m nsp.scripts.train_model --problem {problem} --model nn_e '
    cmd += f'--embed_hidden_dim {params["embed_hidden_dim"]} '
    cmd += f'--embed_dim1 {params["embed_dim1"]} '
    cmd += f'--embed_dim2 {params["embed_dim2"]} '
    cmd += f'--relu_hidden_dim {params["relu_hidden_dim"]} '
    cmd += f'--agg_type {params["agg_type"]} '
    cmd += f'--lr {params["lr"]} '
    cmd += f'--dropout {params["dropout"]} '
    cmd += f'--optimizer {params["optimizer_type"]} '
    cmd += f'--batch_size {params["batch_size"]} '
    cmd += f'--loss_fn {params["loss_fn"]} '
    cmd += f'--wt_lasso {params["wt_lasso"]} '
    cmd += f'--wt_ridge {params["wt_ridge"]} '
    cmd += f'--log_freq {params["log_freq"]} '
    cmd += f'--n_epochs {params["n_epochs"]} '
    cmd += f'--use_wandb {params["use_wandb"]} '

    return cmd


def get_scenario_and_test_sets(problem):
    """  Gets scenario set sizes and test set indexes to
         reproduce results from the paper.
    """
    if problem == 'cflp_10_10':
        scenarios  = [100, 500, 1000] 
        test_sets = list(range(0,11))
    elif problem == 'cflp_25_25':
        scenarios  = [100, 500, 1000] 
        test_sets = list(range(0,11))
    elif problem == 'cflp_50_50':
        scenarios  = [100, 500, 1000]  
        test_sets = list(range(0,11))

    # Stochastic Server Location Problem
    elif problem == 'sslp_5_25':
        scenarios  = [50, 100] 
        test_sets = list(range(0,11))
        test_sets.append('siplib')
    elif problem == 'sslp_10_50':
        scenarios  = [50, 100, 500, 1000, 2000] 
        test_sets = list(range(0,11))
        test_sets.append('siplib')
    elif problem == 'sslp_15_45':
        scenarios  = [5, 10, 15] 
        test_sets = list(range(0,11))
        test_sets.append('siplib')

    # Investment Problem
    elif problem == 'ip_b_H':
        scenarios  = [4, 9, 36, 121, 441, 1681, 10000] 
        test_sets = [0]
    elif problem == 'ip_i_H':
        scenarios  = [4, 9, 36, 121, 441, 1681, 10000] 
        test_sets = [0]
    elif problem == 'ip_b_E':
        scenarios  = [4, 9, 36, 121, 441, 1681, 10000] 
        test_sets = [0]
    elif problem == 'ip_i_E':
        scenarios  = [4, 9, 36, 121, 441, 1681, 10000] 
        test_sets = [0]

    # Pooling Problem
    elif problem == 'pp':
        scenarios  = [125, 216, 343, 512, 729, 1000] 
        test_sets = [0]

    return scenarios, test_sets



def get_commands(problem, args):
    """ Gets list of commands to reproduce experiements. """
    cmds = []

    # generate instance and dataset function calls.
    if args.run_all or args.run_dg_inst:
        cmds.append(f'python -m nsp.scripts.run_dm --problem {problem} --mode GEN_INSTANCE')
    if args.run_all or args.run_dg_p:
        cmds.append(f'python -m nsp.scripts.run_dm --problem {problem} --mode GEN_DATASET_P --n_procs {args.n_cpus}')
    if args.run_all or args.run_dg_e:
        cmds.append(f'python -m nsp.scripts.run_dm --problem {problem} --mode GEN_DATASET_E --n_procs {args.n_cpus}')

    # generate train_model function calls.
    if args.run_all or args.train_lr:
        cmds.append(f'python -m nsp.scripts.train_model --problem {problem} --model lr ')
    if args.run_all or args.train_nn_p:
        cmds.append(get_nn_p_function_call(args, problem))
    if args.run_all or args.train_nn_e:
        cmds.append(get_nn_e_function_call(args, problem))

    # call get best model
    if args.run_all or args.get_best_nn_p_model:
        cmds.append(f'python -m nsp.scripts.get_best_model --problem {problem} --model nn_p ')
    if args.run_all or args.get_best_nn_e_model:
        cmds.append(f'python -m nsp.scripts.get_best_model --problem {problem} --model nn_e ')

    # call get scenario sets and test sets for problem
    scenarios, test_sets = get_scenario_and_test_sets(problem)

    # evaluate scenarios and test sets for problem
    for scenario in scenarios:
        for test_set in test_sets:
            if args.run_all or args.eval_lr:
                cmds.append(f'python -m nsp.scripts.evaluate_model --problem {problem} --model lr --n_scenarios {scenario} --test_set {test_set} --n_procs {args.n_cpus}')
            if args.run_all or args.eval_nn_p:
                cmds.append(f'python -m nsp.scripts.evaluate_model --problem {problem} --model nn_p --n_scenarios {scenario} --test_set {test_set} --n_procs {args.n_cpus}')
            if args.run_all or args.eval_nn_e:
                cmds.append(f'python -m nsp.scripts.evaluate_model --problem {problem} --model nn_e --n_scenarios {scenario} --test_set {test_set} --n_procs {args.n_cpus}')
            if args.run_all or args.eval_ef:
                cmds.append(f'python -m nsp.scripts.evaluate_extensive --problem {problem} --n_scenarios {scenario} --test_set {test_set} --n_procs {args.n_cpus}')

    return cmds


def main(args):
    """  Get commands and execute them sequentially. """
    sys.path.append(args.data_dir)

    if args.as_dat and args.run_all:
        raise Exception(" Using --run_all and --as_dat may cause issues with parallelization. \n\
            It is strongly recommend to read the comments at the end of runner.py with respect --as_dat usage.  ")

    # global variables for counting index of commant
    cmds = []
    for problem in args.problems:
        cmds += get_commands(problem, args)

    # convert commands to index .dat file
    if args.as_dat:
        for idx in range(len(cmds)):
            cmds[idx] = f"{idx + 1} {cmds[idx]}\n" 

        # write dat file
        with open(args.dat_file, 'w') as f:
            for cmd in cmds:
                f.write(cmd)

    # Otherwise, execute commands sequentially
    # Note that this will take quite a long time.
    else:
        for cmd in cmds:
            cmd_as_list = cmd.split(" ")
            subprocess.call(cmd_as_list)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--problems', type=str, nargs='+', default=['cflp_10_10'],
         description = 'The problem(s) to run.  Must be from the following set of problems: \n\
                            cflp_10_10, cflp_25_25, cflp_50_50, sslp_5_25, sslp_10_50, \n\
                            sslp_15_45, ip_b_E, ip_b_H, ip_i_E, ip_i_H, pp')
    parser.add_argument('--data_dir', type=str, default='./data/',
         description = 'The data directory.  This should be left as the default unless otherwise required.')
    parser.add_argument('--n_cpus', type=int, default=1,
         description = 'Number of CPUs/threads to use.  This is only used in data generation and evaluating first-stage solutions.')

    # run all commands (this overrides all below arguments)
    parser.add_argument('--run_all', type=int, default=0,
         description = 'Runs all commands for a specified problem.  Should only be used in if reproducing experiements sequentially.')

    # commands indicating which parts of experiements to run
    parser.add_argument('--run_dg_inst', type=int, default=0,
         description = 'Runs commands to generate instance')
    parser.add_argument('--run_dg_p', type=int, default=0,
         description = 'Runs commands to generate dataset for NN-P.')
    parser.add_argument('--run_dg_e', type=int, default=0,
         description = 'Runs commands to generate dataset for NN-E.')

    # train models
    parser.add_argument('--train_lr', type=int, default=0,
         description = 'Trains linear regression model.')
    parser.add_argument('--train_nn_p', type=int, default=0,
         description = 'Trains NN-P model.')
    parser.add_argument('--train_nn_e', type=int, default=0,
         description = 'Trains NN-E model.')

    parser.add_argument('--get_best_nn_p_model', type=int, default=0,
         description = 'Recovers best model for NN-P.')
    parser.add_argument('--get_best_nn_e_model', type=int, default=0,
         description = 'Recovers best model for NN-E.')

    # evaluate models
    parser.add_argument('--eval_lr', type=int, default=0,
         description = 'Evaluations opimization model with linear regression predictor.')
    parser.add_argument('--eval_nn_p', type=int, default=0,
         description = 'Evaluations opimization model with NN-P predictor.')
    parser.add_argument('--eval_nn_e', type=int, default=0,
         description = 'Evaluations opimization model with NN-E predictor.')
    parser.add_argument('--eval_ef', type=int, default=0,
         description = 'Evaluations extensive form.')

    # This argument is mostly for those with parallel computing resources.  
    # Parallel computing is not strictly nesseary, but will provide notable in evaluation.  
    # This allows one to write the jobs to a .dat to be executed in parallel with Meta on Compute Canada,
    # however, this may be general to other clusters or can be modified to easily run in parallel.
    # This does require changing some of the default arguments above.
    # An example of how to use this is for a single problem (cflp_10_10) is provided below.  However,
    # it is if one wants to reproduce results for multiple problems, then all problems should be incldued.
    # The cflp_10_10 example is provided below:
    #   - Generate instance: 
    #       python runner.py --problems cflp_10_10 --run_dg_inst 1 --as_dat 1
    #   - Generate datasets: 
    #       python runner.py --problems cflp_10_10 --run_dg_p 1 --run_dg_e 1 --as_dat 1
    #   - Train Models: 
    #       python runner.py --problems cflp_10_10 --train_lr 1 --train_nn_p 1 --train_nn_e 1 --as_dat 1
    #   - Get Best Models: 
    #       python runner.py --problems cflp_10_10 --get_best_nn_p_model 1 --get_best_nn_e_model 1 --as_dat 1
    #   - Evaluate Models and Extensive Form: 
    #       python runner.py --problems cflp_10_10 --eval_lr 1 --eval_nn_p 1 --eval_nn_e 1 --eval_nn_p 1 --as_dat 1
    parser.add_argument('--as_dat', type=int, default=0,
         description = 'Indicator for saving commands to .dat files for use with parallel computing.')
    parser.add_argument('--dat_file', type=str, default='table.dat',
         description = 'File to save batch commands to.')

    args = parser.parse_args()

    main(args)


