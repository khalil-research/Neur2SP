from types import SimpleNamespace

# -----------------------------------------#
#   Capacited Facility Location Problem   #
# -----------------------------------------#

cflp_10_10 = SimpleNamespace(
    n_facilities=10,
    n_customers=10,
    ratio=2.0,
    flag_integer_second_stage=True,
    flag_bound_tightening=True,
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=10,      # NN-P specific data generation
    n_samples_e=5000,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=60,                  # data generation
    mip_gap=0.01,                   # data generation
    tr_split=0.80,                  # data generation
    verbose=0,                      # data generation
    seed=7,
    data_path='./data'
)

cflp_25_25 = SimpleNamespace(
    n_facilities=25,
    n_customers=25,
    ratio=2.0,
    flag_integer_second_stage=True,
    flag_bound_tightening=True,
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=10,      # NN-P specific data generation
    n_samples_e=5000,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=60,                  # data generation
    mip_gap=0.05,                   # data generation
    tr_split=0.80,                  # data generation
    verbose=0,                      # data generation
    seed=7,
    data_path='./data'
)

cflp_50_50 = SimpleNamespace(
    n_facilities=50,
    n_customers=50,
    ratio=2.0,
    flag_integer_second_stage=True,
    flag_bound_tightening=True,
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=10,      # NN-P specific data generation
    n_samples_e=5000,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=60,                  # data generation
    mip_gap=0.05,                   # data generation
    tr_split=0.80,                  # data generation
    verbose=0,                      # data generation
    seed=7,
    data_path='./data'
)

# ----------------------------------------#
#   Stochastic Server Location Problem    #
# ----------------------------------------#

sslp_5_25 = SimpleNamespace(
    n_locations=5,
    n_clients=25,
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=10,      # NN-P specific data generation
    n_samples_e=5000,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=60,                  # data generation
    mip_gap=0.01,                   # data generation
    tr_split=0.80,                  # data generation
    verbose=0,                      # data generation
    seed=7,
    siplib_instance_names=["sslp_5_25_50", "sslp_5_25_100"],
    data_path='./data'
)

sslp_10_50 = SimpleNamespace(
    n_locations=10,
    n_clients=50,
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=10,      # NN-P specific data generation
    n_samples_e=5000,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=60,                  # data generation
    mip_gap=0.01,                   # data generation
    tr_split=0.80,                  # data generation
    verbose=0,                      # data generation
    seed=7,
    siplib_instance_names=["sslp_10_50_50", "sslp_10_50_100", "sslp_10_50_500",
                           "sslp_10_50_1000", "sslp_10_50_2000"],
    data_path='./data'
)

sslp_15_45 = SimpleNamespace(
    n_locations=15,
    n_clients=45,
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=10,      # NN-P specific data generation
    n_samples_e=5000,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=60,                  # data generation
    mip_gap=0.01,                   # data generation
    tr_split=0.80,                  # data generation
    verbose=0,                      # data generation
    seed=7,
    siplib_instance_names=["sslp_15_45_5", "sslp_15_45_10", "sslp_15_45_15"],
    data_path='./data'
)

# -------------------------#
#   Investment Problem    #
# -------------------------#

# binary second stage
ip_b_H = SimpleNamespace(
    n_instances=1,
    technology_identity=0,
    first_stage_vtype="C",
    second_stage_vtype="B",
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=10,      # NN-P specific data generation
    n_samples_e=5000,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=60,                  # for data generation
    mip_gap=0.01,                   # for data generation
    verbose=0,                      # for data generation
    seed=777,
    tr_split=0.80,
    data_path='./data'
)

# integer second stage
ip_i_H = SimpleNamespace(
    n_instances=1,
    technology_identity=0,
    first_stage_vtype="C",
    second_stage_vtype="I",
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=10,      # NN-P specific data generation
    n_samples_e=5000,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=60,                  # for data generation
    mip_gap=0.01,                   # for data generation
    verbose=0,                      # for data generation
    seed=777,
    tr_split=0.80,
    data_path='./data'
)

# binary second stage, technology identity
ip_b_E = SimpleNamespace(
    n_instances=1,
    technology_identity=1,
    first_stage_vtype="C",
    second_stage_vtype="B",
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=10,      # NN-P specific data generation
    n_samples_e=5000,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=60,                  # for data generation
    mip_gap=0.01,                   # for data generation
    verbose=0,                      # for data generation
    seed=777,
    tr_split=0.80,
    data_path='./data'
)

# integer second stage, technology identity
ip_i_E = SimpleNamespace(
    n_instances=1,
    technology_identity=1,
    first_stage_vtype="C",
    second_stage_vtype="I",
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=10,      # NN-P specific data generation
    n_samples_e=5000,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=60,                  # for data generation
    mip_gap=0.01,                   # for data generation
    verbose=0,                      # for data generation
    seed=777,
    tr_split=0.80,
    data_path='./data'
)

# ---------------------#
#   Pooling Problem    #
# ---------------------#

pp = SimpleNamespace(
    n_instances=1,
    D_sulfur_mean=2.5,
    D_sulfur_dev=0.8,
    X_demand_mean=180,
    X_demand_dev=10,
    Y_demand_mean=200,
    Y_demand_dev=10,
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=10,      # NN-P specific data generation
    n_samples_e=5000,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=60,                  # for data generation
    mip_gap=0.01,                   # for data generation
    verbose=0,                      # for data generation
    cutoff_time_limit=3 * 3600,     # total time for data generation
    tr_split=0.80,
    seed=7,
    data_path='./data'
)
