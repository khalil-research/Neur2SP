import argparse
import pickle
import time
from pathlib import Path

import nsp.params as params
from nsp.two_sp import factory_two_sp
from nsp.utils import factory_get_path
from nsp.utils import factory_sampler
from nsp.utils import load_instance


def main(args):
    print(f"EVALUATING EXTENSIVE: problem = {args.problem}, test={args.test_set}\n")
    problem_str = f"s{args.n_scenarios}_ts{args.test_set}"

    # Get problem specific functions
    global get_path
    get_path = factory_get_path(args)

    # load config
    cfg = getattr(params, args.problem)

    # load instance
    inst = load_instance(args, cfg)

    # load sampler
    sampler = factory_sampler(args, cfg=cfg)

    two_sp = factory_two_sp(args.problem, inst, sampler=sampler)

    # get file locations 
    log_dir = get_path(cfg.data_path, cfg, ptype=f"grb_log_ef_{problem_str}", suffix='.log', as_str=True)
    node_file_dir = f'node_files/{problem_str}'
    node_file_dir = Path(node_file_dir)

    if not node_file_dir.exists():  # make node file directory is needed
        node_file_dir.mkdir(parents=True, exist_ok=True)

    # solve extensive
    ef_time = time.time()
    ef_mip = two_sp.solve_extensive(args.n_scenarios,
                                    gap=args.mip_gap,
                                    time_limit=args.time_limit,
                                    threads=args.mip_threads,
                                    log_dir=log_dir,
                                    node_file_start=args.node_file_start,
                                    node_file_dir=str(node_file_dir),
                                    test_set=args.test_set)
    ef_time = time.time() - ef_time

    # get first stage solution and objective of first stage solution
    first_stage_sol = two_sp.get_first_stage_extensive_solution(ef_mip)
    ef_fs_obj = two_sp.evaluate_first_stage_sol(first_stage_sol,
                                                args.n_scenarios,
                                                verbose=0,
                                                test_set=args.test_set,
                                                n_procs=args.n_procs)

    results = {
        'time_total': ef_time,
        'obj': ef_mip.objVal,
        'dual': ef_mip.ObjBound,
        'sol': first_stage_sol,
        'true_obj': ef_fs_obj,
        'solving_results': two_sp.ef_solving_results,
        'time_ef_mip': ef_mip.Runtime,
    }

    print("Extensive form stats:")
    print(f'  Incumbent:   {ef_mip.objVal}')
    print(f'  Dual bound:  {ef_mip.ObjBound} ')
    print(f'  FS obj:      {ef_fs_obj} ')
    print(f'  Time:        {ef_time}')
    print(f"  x:           {first_stage_sol}")

    fp_results = get_path(cfg.data_path, cfg, ptype=f"ef_results_{problem_str}")

    with open(fp_results, 'wb') as p:
        pickle.dump(results, p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate extensive form.')
    parser.add_argument('--problem', type=str, default="cflp_10_10")
    parser.add_argument('--n_scenarios', type=int, default=100)
    parser.add_argument('--test_set', type=str, default="0",
                        help='Evaluate on a test set (unseen scenarios).')

    # Optimization parameters
    parser.add_argument('--time_limit', type=int, default=3 * 3600, help='Time limit for solver.')
    parser.add_argument('--mip_gap', type=float, default=0.0001, help='Gap limit for solver.')
    parser.add_argument('--node_file_start', type=float, default=0.5, help='Node file amount to avoid out of memory.')
    parser.add_argument('--mip_threads', type=int, default=1, help='Number of threads for MIP solver.')
    parser.add_argument('--n_procs', type=int, default=1)
    args = parser.parse_args()
    main(args)
