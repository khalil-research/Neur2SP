from .two_sp import TwoStageStocProg


def factory_two_sp(problem, inst, sampler=None):
    if 'cflp' in problem:
        from .cflp import FacilityLocationProblem
        return FacilityLocationProblem(inst)

    elif 'ip' in problem:
        assert sampler is not None
        from .ip import InvestmentProblem

        scenarios = sampler.get_support()
        return InvestmentProblem(inst, scenarios)

    elif 'sslp' in problem:
        from .sslp import SSLP
        return SSLP(inst)

    elif 'pp' in problem:
        assert sampler is not None
        from .pp import PoolingProblem

        scenarios = sampler.get_support()
        return PoolingProblem(inst, scenarios)

    else:
        raise Exception(f"nsp.utils not defined for problem class {problem}")
