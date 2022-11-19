from pathlib import Path


def get_path(data_path, cfg, ptype="inst", suffix=".pkl", as_str=False):
    p = Path(data_path) / "cflp"
    p.mkdir(parents=True, exist_ok=True)

    p = p / f"{ptype}_f{cfg.n_facilities}_c{cfg.n_customers}_" \
            f"r{cfg.ratio}_iss{1 if cfg.flag_integer_second_stage else 0}" \
            f"_bt{1 if cfg.flag_bound_tightening else 0}_nsp{cfg.n_samples_p}_nse{cfg.n_samples_e}_sd{cfg.seed}{suffix}"
            
    if as_str:
        return str(p)
    return p
