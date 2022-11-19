import argparse
import pickle

import numpy as np

import nsp.params as params
from nsp.utils.cflp import get_path as get_path_cflp
from nsp.utils.ip import get_path as get_path_ip
from nsp.utils.pp import get_path as get_path_pp
from nsp.utils.sslp import get_path as get_path_sslp


def get_imports_for_problem_class(args):
    """ Imports correct get_path file based on args.problem.  """
    global get_path

    if 'cflp' in args.problem:
        get_path = get_path_cflp
    elif 'ip' in args.problem:
        get_path = get_path_ip
    elif 'sslp' in args.problem:
        get_path = get_path_sslp
    elif 'pp' in args.problem:
        get_path = get_path_pp
    else:
        raise Exception(f"nsp.utils not defined for problem class {args.problem}")


def get_data_generation_stats(results, cfg, args, expected=False):
    """ Gets info/stats from data geneartion. """
    if expected:
        ml_data_fp = get_path(args.data_dir, cfg, ptype="ml_data_e")
        try:
            data = pickle.load(open(ml_data_fp, 'rb'))
            results['nn_e_data_generation_time'] = data['total_time']
            results['nn_e_n_samples'] = len(data['data'])
            #results['nn_e_n_lps_solved'] = cfg.n_samples_e
            results['nn_e_time_per_sample'] = results['nn_e_data_generation_time'] / results['nn_e_n_samples']
        except:
            print(f"Failed to load data: {ml_data_fp}")
            results['nn_e_data_generation_time'] = np.nan
            results['nn_e_n_samples'] = np.nan
            #results['nn_e_n_lps_solved'] = np.nan
            results['nn_e_time_per_sample'] = np.nan
    else:
        ml_data_fp = get_path(args.data_dir, cfg, ptype="ml_data_p")
        try:
            data = pickle.load(open(ml_data_fp, 'rb'))
            results['nn_p_data_generation_time'] = data['total_time']
            results['nn_p_n_samples'] = len(data['data'])
            results['nn_p_time_per_sample'] = results['nn_p_data_generation_time'] / results['nn_p_n_samples']
        except:
            print(f"  Failed to load data: {ml_data_fp}")
            results['nn_p_data_generation_time'] = np.nan
            results['nn_p_n_samples'] = np.nan
            results['nn_p_time_per_sample'] = np.nan
    return results


def get_model_training_results(results, cfg, args, model_type):
    """ Gets training results for given model_type.  """
    result_fp = get_path(args.data_dir, cfg, ptype=f"{model_type}_results")
    try:
        model_results = pickle.load(open(result_fp, 'rb'))
        results[f'{model_type}_training_time'] = model_results['time']
        results[f'{model_type}_training_stats'] = model_results
    except:
        print(f"  Failed to load training results: {result_fp}")
        results[f'{model_type}_training_time'] = np.nan
        results[f'{model_type}_training_stats'] = np.nan
    return results


def get_embedding_results(results, cfg, args, model_type, n_scenarios, test_set, dict_key="downstream"):
    """ Gets downstream results for given model_type.  """
    problem_str = f"{model_type}_s{n_scenarios}_ts{test_set}"
    result_fp = get_path(args.data_dir, cfg, ptype=f"embedding_results_{problem_str}")

    try:
        embedding_results = pickle.load(open(result_fp, 'rb'))
        results[dict_key][test_set][f'{model_type}_pred_obj'] = embedding_results['predicted_obj']
        results[dict_key][test_set][f'{model_type}_true_obj'] = embedding_results['true_obj']
        results[dict_key][test_set][f'{model_type}_time'] = embedding_results['time']
        results[dict_key][test_set][f'{model_type}_sol'] = embedding_results['sol']
        results[dict_key][test_set][f'{model_type}_solving_results'] = embedding_results['solving_results']
        try:  # in the case of LR this is an LP so no incumbent  will be found
            results[dict_key][test_set][f'{model_type}_time_to_best_sol_grb'] = \
                embedding_results['solving_results']['time'][-1]
        except:
            results[dict_key][test_set][f'{model_type}_time_to_best_sol_grb'] = embedding_results['time']

    except:
        print(f"  Failed to load {problem_str} embedding results: {result_fp}")
        results[dict_key][test_set][f'{model_type}_pred_obj'] = np.nan
        results[dict_key][test_set][f'{model_type}_true_obj'] = np.nan
        results[dict_key][test_set][f'{model_type}_time'] = np.nan
        results[dict_key][test_set][f'{model_type}_sol'] = np.nan
        results[dict_key][test_set][f'{model_type}_solving_results'] = np.nan
        results[dict_key][test_set][f'{model_type}_time_to_best_sol_grb'] = np.nan

    return results


def get_extensive_results(results, cfg, args, n_scenarios, test_set, dict_key="downstream"):
    """ Gets extensive form results.  """
    problem_str = f"s{n_scenarios}_ts{test_set}"
    result_fp = get_path(args.data_dir, cfg, ptype=f"ef_results_{problem_str}")

    try:
        ef_results = pickle.load(open(result_fp, 'rb'))
        results[dict_key][test_set]['ef_obj'] = ef_results['obj']
        results[dict_key][test_set]['ef_obj_fs'] = ef_results['true_obj']
        results[dict_key][test_set]['ef_dual'] = ef_results['dual']
        results[dict_key][test_set]['ef_time'] = ef_results['time_ef_mip']
        results[dict_key][test_set]['ef_solving_results'] = ef_results['solving_results']
        results[dict_key][test_set]['ef_time_to_best_sol_grb'] = ef_results['solving_results']['time'][-1]

    except:
        print(f"  Failed to load extensive results: {result_fp}")
        results[dict_key][test_set]['ef_obj'] = np.nan
        results[dict_key][test_set]['ef_obj_fs'] = np.nan
        results[dict_key][test_set]['ef_dual'] = np.nan
        results[dict_key][test_set]['ef_time'] = np.nan
        results[dict_key][test_set]['ef_solving_results'] = np.nan
        results[dict_key][test_set]['ef_time_to_best_sol_grb'] = np.nan

    return results


def get_saa_results(results, cfg, args, n_scenarios, large_n_scenario, test_set, dict_key="large_scenario"):
    """ Gets extensive form results.  """
    result_fp = get_path(args.data_dir, cfg, ptype=f"ef_saa_results_sfs{n_scenarios}_sev{large_n_scenario}")

    try:
        ef_results = pickle.load(open(result_fp, 'rb'))
        results[dict_key][test_set]['ef_first_stage_sols'] = ef_results['ef_first_stage_sols']
        results[dict_key][test_set]['ef_objs'] = ef_results['ef_objs']
        results[dict_key][test_set]['ef_best_first_stage_sol'] = ef_results['ef_best_first_stage_sol']
        results[dict_key][test_set]['ef_best_obj'] = ef_results['ef_best_obj']
        results[dict_key][test_set]['ef_n_unique'] = ef_results['ef_n_unique']

    except:
        print(f"  Failed to extensive results: {result_fp}")
        results[dict_key][test_set]['ef_first_stage_sols'] = np.nan
        results[dict_key][test_set]['ef_objs'] = np.nan
        results[dict_key][test_set]['ef_best_first_stage_sol'] = np.nan
        results[dict_key][test_set]['ef_best_obj'] = np.nan
        results[dict_key][test_set]['ef_n_unique'] = np.nan

    return results


def main(args):
    print(f"Collecting results for {args.problem} with {args.n_scenarios} scenarios...")

    # load config
    get_imports_for_problem_class(args)
    cfg = getattr(params, args.problem)

    results = {}

    # add results for data generation
    results = get_data_generation_stats(results, cfg, args, expected=False)
    results = get_data_generation_stats(results, cfg, args, expected=True)

    # add results for model training
    results = get_model_training_results(results, cfg, args, 'lr')
    results = get_model_training_results(results, cfg, args, 'nn_p')
    results = get_model_training_results(results, cfg, args, 'nn_e')

    results['downstream'] = {}

    # add results for ef/embeddings
    for test_set in args.test_sets:
        results['downstream'][test_set] = {}
        results = get_embedding_results(results, cfg, args, 'lr', args.n_scenarios, test_set)
        results = get_embedding_results(results, cfg, args, 'nn_p', args.n_scenarios, test_set)
        results = get_embedding_results(results, cfg, args, 'nn_e', args.n_scenarios, test_set)
        results = get_extensive_results(results, cfg, args, args.n_scenarios, test_set)


    # save results
    result_fp = get_path(args.data_dir, cfg, ptype=f"combined_results_s{args.n_scenarios}")
    with open(result_fp, 'wb') as p:
        pickle.dump(results, p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collects results for given problem from the specified data directory.')

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--problem', type=str)
    parser.add_argument('--n_scenarios', type=int)
    parser.add_argument('--test_sets', nargs="+", type=str)

    args = parser.parse_args()

    main(args)
