import nsp.params as params


def factory_dm(problem):
    cfg = getattr(params, problem)

    if "cflp" in problem:
        print("Loading CFLP data manager...")
        from .cflp import FacilityLocationDataManager
        return FacilityLocationDataManager(cfg)
    if "ip" in problem:
        print("Loading Investment Problem data manager...")
        from .ip import InvestmentProblemDataManager
        return InvestmentProblemDataManager(cfg)
    if "sslp" in problem:
        print("Loading SSLP data manager...")
        from .sslp import SSLPDataManager
        return SSLPDataManager(cfg)
    if "pp" in problem:
        print("Loading PP data manager...")
        from .pp import PoolingProblemDataManager
        return PoolingProblemDataManager(cfg)
    else:
        raise ValueError("Invalid problem type!")
