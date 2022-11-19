from pathlib import Path


def get_path(data_path, cfg, ptype="inst", suffix=".pkl", as_str=False):
    p = Path(data_path) / "sslp"
    p.mkdir(parents=True, exist_ok=True)

    p = p / f"{ptype}_l{cfg.n_locations}_c{cfg.n_clients}_nsp{cfg.n_samples_p}_nse{cfg.n_samples_e}_sd{cfg.seed}{suffix}"

    if as_str:
        return str(p)
    return p
