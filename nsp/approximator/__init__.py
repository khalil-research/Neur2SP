from nsp.model2mip import factory_model2mip


def factory_approximator(args, two_sp, model, model_type):
    mipper = factory_model2mip(model_type)

    if 'cflp' in args.problem:
        from .cflp import FacilityLocationProblemApproximator
        return FacilityLocationProblemApproximator(two_sp, model, model_type, mipper)

    elif 'ip' in args.problem:
        from .ip import InvestmentProblemApproximator
        return InvestmentProblemApproximator(two_sp, model, model_type, mipper)

    elif 'sslp' in args.problem:
        from .sslp import ServerLocationProblemApproximator
        return ServerLocationProblemApproximator(two_sp, model, model_type, mipper)

    elif 'pp' in args.problem:
        from .pp import PoolingProblemApproximator
        return PoolingProblemApproximator(two_sp, model, model_type, mipper)

    else:
        raise Exception(f"nsp.utils not defined for problem class {args.problem}")
