import itertools
from pathlib import Path

import numpy as np


def get_path(data_path, cfg, ptype="inst", suffix=".pkl", as_str=False):
    p = Path(data_path) / f"ip"
    p.mkdir(parents=True, exist_ok=True)

    p = p / f"{ptype}_fv{cfg.first_stage_vtype}_sv{cfg.second_stage_vtype}_" \
            f"n{cfg.n_instances}_tmi{cfg.technology_identity}_sd{cfg.seed}{suffix}"

    if as_str:
        return str(p)

    return p


class Sampler:
    def __init__(self, samples_per_dist=1000):
        self.samples_per_dist = np.linspace(5, 15, samples_per_dist)
        self.xi_support = list(itertools.product(self.samples_per_dist,
                                                 self.samples_per_dist))
        self.xi_support_np = np.asarray(self.xi_support)
        self.xi_support_idx = np.arange(len(self.xi_support))
        self.xi_prob = [1 / len(self.xi_support)] * len(self.xi_support)
        self.rng = np.random.RandomState()

    def get_scenarios(self, n_scenarios):
        return self.xi_support_np[self.rng.choice(self.xi_support_idx,
                                                  size=n_scenarios,
                                                  replace=False,
                                                  p=self.xi_prob)]

    def get_support(self):
        return self.xi_support_np
