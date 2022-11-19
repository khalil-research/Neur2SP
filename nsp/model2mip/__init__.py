from .lr2mip import LR2MIP
from .net2mip import Net2MIPPerScenario, Net2MIPExpected


def factory_model2mip(model_type):
    if model_type == 'lr':
        return LR2MIP
    elif model_type == 'nn_p':
        return Net2MIPPerScenario
    elif model_type == 'nn_e':
        return Net2MIPExpected
    else:
        raise ValueError('Invalid model type!')
