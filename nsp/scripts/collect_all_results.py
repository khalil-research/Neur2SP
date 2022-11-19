import argparse
import pickle
import subprocess

from nsp.utils import get_large_scenario_size
from nsp.utils import get_problem_from_param
from nsp.utils import get_results_for_problem
from nsp.utils import get_scenario_sets_for_problem
from nsp.utils import get_test_sets


def main(args):
    # Get the names of problems from params.py
    problems = get_problem_from_param()

    combined_results = {}
    for problem in problems:

        print('Problem: ', problem)

        problem_scenario_sets = get_scenario_sets_for_problem(problem)
        problem_large_scenario_size = get_large_scenario_size(problem)

        for n_scenarios in problem_scenario_sets:
            problem_str = f"{problem}_{n_scenarios}"
            problem_test_sets = get_test_sets(problem, n_scenarios)

            cmd_as_list = ["python", "-m", "nsp.scripts.collect_results",
                           "--problem", problem,
                           "--data_dir", args.data_dir,
                           "--n_scenarios", str(n_scenarios),
                           "--test_sets"] + problem_test_sets

            subprocess.call(cmd_as_list)
            combined_results[problem_str] = get_results_for_problem(args, problem, n_scenarios)

    fp_res = args.data_dir + 'results.pkl'
    with open(fp_res, 'wb') as p:
        pickle.dump(combined_results, p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collects are stores all results.')

    parser.add_argument('--data_dir', type=str, default="./data/")
    args = parser.parse_args()

    main(args)
