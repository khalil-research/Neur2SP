from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import norm


def get_path(data_path, cfg, ptype="inst", suffix=".pkl", as_str=False):
    p = Path(data_path) / f"pp"
    p.mkdir(parents=True, exist_ok=True)

    p = p / f"{ptype}_n{cfg.n_instances}_sd{cfg.seed}{suffix}"

    if as_str:
        return str(p)

    return p


class Sampler:
    def __init__(self, cfg, samples_per_dist=100, seed=0):
        self.cfg = cfg
        self.samples_per_dist = samples_per_dist
        self.n_support = samples_per_dist ** 3
        self.scenarios = None
        self.rng = np.random.RandomState(seed)

        D_sulfur = self._generate_scenarios(mean=self.cfg.D_sulfur_mean,
                                            dev=self.cfg.D_sulfur_dev,
                                            samples=self.samples_per_dist)
        X_demand = self._generate_scenarios(mean=self.cfg.X_demand_mean,
                                            dev=self.cfg.X_demand_dev,
                                            samples=self.samples_per_dist)
        Y_demand = self._generate_scenarios(mean=self.cfg.Y_demand_mean,
                                            dev=self.cfg.Y_demand_dev,
                                            samples=self.samples_per_dist)

        scenarios = product(D_sulfur, X_demand, Y_demand)
        scenarios = [(scen[0][0], scen[1][0], scen[2][0], scen[0][1] * scen[1][1] * scen[2][1])
                     for scen in scenarios]

        probs = []
        demand = {}
        sulfur = {}
        for sid, scen in enumerate(scenarios):
            probs.append(scen[-1])
            sulfur[('A', sid)] = 3
            sulfur[('B', sid)] = 1
            sulfur[('C', sid)] = 2
            sulfur[('D', sid)] = scen[0]
            demand[('X', sid)] = scen[1]
            demand[('Y', sid)] = scen[2]

        self.scenarios = {
            'scenarios': {
                'demand': demand,
                'sulfur': sulfur
            },
            'probs': probs
        }

    def get_scenarios(self, n_scenarios):
        scenarios_idxs = self.get_scenario_idxs(n_scenarios)

        scenarios = self.scenarios['scenarios']

        probs = []
        sulfur = {}
        demand = {}
        for idx, scenario_idx in enumerate(scenarios_idxs):
            probs.append(self.scenarios['probs'][scenario_idx])
            sulfur[('A', idx)] = 3
            sulfur[('B', idx)] = 1
            sulfur[('C', idx)] = 2
            sulfur[('D', idx)] = scenarios['sulfur']['D', scenario_idx]
            demand[('X', idx)] = scenarios['demand']['X', scenario_idx]
            demand[('Y', idx)] = scenarios['demand']['Y', scenario_idx]

        probs = np.asarray(probs) / np.sum(probs)

        return {'scenarios': {'sulfur': sulfur, 'demand': demand}, 'probs': probs}

    def get_support(self):
        return self.scenarios

    def get_scenario_idxs(self, n_scenarios):
        return self.rng.choice(np.arange(self.n_support),
                               size=n_scenarios,
                               replace=False,
                               p=self.scenarios['probs'])

    @staticmethod
    def _generate_scenarios(mean=2.5, dev=0.8, samples=3):
        scenarios = []
        probs = []
        for k in range(1, samples + 1):
            scenarios.append(-3 * dev + mean + 3 * dev / samples + (k - 1) * 6 * dev / samples)

            if k == 1 and samples == 1:
                probs.append(1)

            elif k == 1 and samples > 1:
                probs.append(norm.cdf(-3 * dev + mean + 6 * dev / samples, loc=mean, scale=dev))

            elif k == samples:
                probs.append(1 - norm.cdf(-3 * dev + mean + 6 * dev * (k - 1) / samples, loc=mean, scale=dev))

            else:
                probs.append(norm.cdf(-3 * dev + mean + 6 * dev * k / samples, loc=mean, scale=dev) - norm.cdf(
                    -3 * dev + mean + 6 * dev * (k - 1) / samples, loc=mean, scale=dev))

        return [(s, p) for s, p in zip(scenarios, probs)]
