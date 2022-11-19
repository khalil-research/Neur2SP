from argparse import ArgumentParser

from nsp.dm import factory_dm
from nsp.utils import DataManagerModes as Modes


def main(args):
    data_manager = factory_dm(args.problem)
    if args.do == Modes.GEN_INSTANCE.value or args.mode == Modes.GEN_INSTANCE.name:
        data_manager.generate_instance()
    elif args.do == Modes.GEN_DATASET_P.value or args.mode == Modes.GEN_DATASET_P.name:
        data_manager.generate_dataset_per_scenario(args.n_procs)
    elif args.do == Modes.GEN_DATASET_E.value or args.mode == Modes.GEN_DATASET_E.name:
        data_manager.generate_dataset_expected(args.n_procs)
    else:
        raise ValueError("Invalid argument for do")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str)
    parser.add_argument('--do', type=int)
    parser.add_argument('--mode', type=str, choices=list([m.name for m in Modes]))
    parser.add_argument('--n_procs', type=int, default=1)
    args = parser.parse_args()
    main(args)
