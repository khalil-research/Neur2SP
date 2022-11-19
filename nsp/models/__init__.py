from nsp.utils import LearningModelTypes
from .network import ReLUNetworkPerScenario, ReLUNetworkExpected


def factory_learning_model(args, inst):
    if args.model_type == LearningModelTypes.lr.name:
        from .sklearn_model import LinearRegression
        return LinearRegression()

    elif args.model_type == LearningModelTypes.nn_p.name:
        from .pytorch_model import ReLUNetworkPerScenarioModel
        return ReLUNetworkPerScenarioModel(
            args.problem,
            inst,
            args.hidden_dims,
            args.lr,
            args.dropout,
            args.optimizer,
            args.batch_size,
            args.loss_fn,
            args.wt_lasso,
            args.wt_ridge,
            args.log_freq,
            args.n_epochs,
            args.use_wandb
        )

    elif args.model_type == LearningModelTypes.nn_e.name:
        from .pytorch_model import ReLUNetworkExpectedModel
        return ReLUNetworkExpectedModel(
            args.problem,
            inst,
            args.embed_hidden_dim,
            args.embed_dim1,
            args.embed_dim2,
            args.relu_hidden_dim,
            args.lr,
            args.dropout,
            args.optimizer,
            args.batch_size,
            args.loss_fn,
            args.wt_lasso,
            args.wt_ridge,
            args.agg_type,
            args.log_freq,
            args.n_epochs,
            args.use_wandb
        )
