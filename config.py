import itertools

RESULT_DIR = 'result' # online SAC results
OFF_RESULT_DIR = 'off_result' # offline methods results
DATA_DIR = 'data' # dataset from online policies
OFF_DATA_DIR = 'off_data' # dataset from offline policies
PLOT_DIR = 'plot' # plots from online SAC
OFF_PLOT_DIR = 'off_plot' # plots from offline methods
CORR_DIR = 'correlation' # correlation plot of ours
VIDEO_DIR = 'video' # videos from offline methods
VIDEO_COMPARE_DIR = 'video_comp' # videos from offline methods, compared with offline data.
REWARD_DIR = 'reward' # plots of per-step rewards
PER_STEP_DIST_DIR = 'ps_dist' # plots of per-step distances
INIT_ACT_DIR = 'init_act' # plots of init actions

VARIANT = 'variant.json'
PROGRESS = 'progress.csv'
DATA = 'data.pkl'

D4RL_DIR = '/data_large/readonly/d4rl/datasets'

GRID_TIMEOUT = 100
UMAZE_TIMEOUT = 200

ACTION_MIN = -1
ACTION_MAX = 1

OBS_MIN = 0
OBS_MAX = 6

STATE_MIN = 0

BUFFER_ESSENTIALS = [
    'observations',
    'actions',
    'next_observations',
    'rewards',
    'rtgs',
    'dones',
]

# Trajectory paths
RANDOM = [
]

MEDIUM = [
]

EXPERT = [
]

MEDIUM_EXPERT = MEDIUM + EXPERT

# ours
TARGET_ENTROPY_THRES = -100

NUM_WORKERS = 10
TOP_K = 100
MIXED_ROLLOUT_TIMESTEPS = list(range(0, 110, 10))


BC_MODELS = [
    "BC_{}_plr{:.5f}".format(distance, plr)
    for distance in ['distance_conti']
    # for plr in [0.00003, 0.0001]
    for plr in [0.00003]
]

CQL_MODELS = [
    "CQL_plr{:.5f}_minqw{:.1f}_player{}_qlayer{}_ls{}".format(plr, minqw, pl, ql, ls)
    # "CQL_plr{:.5f}_minqw{:.1f}".format(plr, minqw)
    # for plr in [1e-4, 3e-4]
    # for minqw in [1.0, 10.0]
    for plr in [1e-4]
    for minqw in [10.0]
    for pl in [3]
    for ql in [3]
    for ls in [256]
]

IQL_MODELS = [
    "IQL_plr{:.5f}_b{}_q{}_cs100.0".format(pl, b, q)
    for pl in [0.0001]
    for b in [0.1]
    for q in [0.4]
]

OURS_MODELS = [
    # "OURS_contrast7_distance_conti4_plr0.00030_player2_local_gaussian_1.00_0.0001_bs16_ts16_tp1.00"
    # "OURS_{}_{}_plr{:.5f}_player{}_ls{}_bs{}_ts{}_tp{:.2f}".format(
    "OURS_{}_{}_plr{:.5f}_player{}_ls{}_bs{}_ts{}_tp{:.2f}".format(
        s, d, plr, pl, ls, bs, ts, tp
    )
    for s in ["contrast9"]
    for d in ["distance_conti4"]
    for plr in [3e-4]
    for pl in [3]
    for ls in [256]
    # for log in ["", "_log"]
    # for ent in ["", "_entropy-1000.0"]
    # for log in ["_log"]
    # for ent in ["_entropy-1000.0"]
    # for local_weight in [1.0]
    # for local_scale in [0.01]
    # for local_scale in [0.0001, 0.001, 0.01]
    # for local_weight in [10.0, 1.0, 0.1]
    for bs in [16]
    for ts in [16]
    # for ts in [-1]
    # for tr in ["", *["_tr{:.2f}".format(r) for r in [0.75, 0.5, 0.25]]]
    # for tp in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    # for tp in [1.0, 0.5, 0.2, 0.1]
    # for tp in [10, 5, 2, 1]
    # for tp in [10.0, 1.0, 0.1]
    # for tp in [0.10, 1.00, 10.00]
    for tp in [0.5]
]

SAC_MODELS = [
    'SAC_ls256_npl2_plr0.0003_bs256_mpl1000_te-1000.00_ter0.98_tup1',
]

MODELS = CQL_MODELS + IQL_MODELS
MODELS = OURS_MODELS + BC_MODELS + CQL_MODELS
MODELS = BC_MODELS + CQL_MODELS
MODELS = BC_MODELS + CQL_MODELS + IQL_MODELS + OURS_MODELS
MODELS = BC_MODELS + CQL_MODELS + IQL_MODELS + OURS_MODELS + SAC_MODELS

ENVS = [
    'halfcheetah-random-v2', 'halfcheetah-medium-v2', 'halfcheetah-expert-v2', 'halfcheetah-medium-expert-v2',
    'hopper-random-v2'     , 'hopper-medium-v2'     , 'hopper-expert-v2'     , 'hopper-medium-expert-v2'     ,
    'walker2d-random-v2'   , 'walker2d-medium-v2'   , 'walker2d-expert-v2'   , 'walker2d-medium-expert-v2'   ,
]
CORE_ENVS = [
    'halfcheetah-medium-expert-v2', 'halfcheetah-medium-v2',
    'hopper-medium-expert-v2'     , 'hopper-medium-v2'     ,
    'walker2d-medium-expert-v2'   , 'walker2d-medium-v2'   ,
]
ALTER_ENVS = [
    'halfcheetah-random-v2', 'halfcheetah-expert-v2',
    'hopper-random-v2'     , 'hopper-expert-v2'     ,
    'walker2d-random-v2'   , 'walker2d-expert-v2'   ,
]
ETC_ENVS = [
    'halfcheetah-medium-replay-v2', 'halfcheetah-full-replay-v2',
    'hopper-medium-replay-v2'     , 'hopper-full-replay-v2'     ,
    'walker2d-medium-replay-v2'   , 'walker2d-full-replay-v2'   ,
]

OPE_BC_MODELS = [
    "BC_{}_plr{:.5f}".format(distance, plr, tsr)
    for distance in ['distance_conti']
    for plr in [0.00003]
    for tsr in [0.1]
]

OPE_CQL_MODELS = [
    "CQL_plr{:.5f}_minqw{:.1f}_player{}_qlayer{}_ls{}".format(plr, minqw, pl, ql, ls, tsr)
    for plr in [1e-4]
    for minqw in [10.0]
    for pl in [3]
    for ql in [3]
    for ls in [256]
    for tsr in [0.1]
]

OPE_IQL_MODELS = [
    "IQL_plr{:.5f}_b{}_q{}_cs100.0".format(pl, b, q, tsr)
    for pl in [0.0001]
    for b in [0.1]
    for q in [0.4]
    for tsr in [0.1]
]

OPE_OURS_MODELS = [
    "OURS_{}_{}_plr{:.5f}_player{}_ls{}_bs{}_ts{}_tp{:.2f}".format(
        s, d, plr, pl, ls, bs, ts, tp, tsr
    )
    for s in ["contrast9"]
    for d in ["distance_conti4"]
    for plr in [3e-4]
    for pl in [3]
    for ls in [256]
    for bs in [16]
    for ts in [16]
    for tp in [0.5]
    for tsr in [0.1]
]

OPE_SAC_MODELS = [
    'SAC_ls256_npl2_plr0.0003_bs256_mpl1000_te-1000.00_ter0.98_tup1',
]

OPE_BC_MODELS_SWEEP = [
    "BC_{}_plr{:.5f}".format(distance, plr)
    for distance in ['distance_conti']
    for plr in [0.00003, 0.0001]
]

OPE_CQL_MODELS_SWEEP = [
    "CQL_plr{:.5f}_minqw{:.1f}_player{}_qlayer{}_ls{}".format(plr, minqw, pl, ql, ls)
    for minqw in [1.0, 10.0]
    for plr in [1e-4, 3e-4]
    for pl in [3]
    for ql in [3]
    for ls in [256]
]

OPE_IQL_MODELS_SWEEP = [
    "IQL_plr{:.5f}_b{}_q{}_cs100.0".format(pl, b, q)
    for pl in [0.0001, 0.0003]
    for b in [0.1]
    for q in [0.4, 0.6]
]

DATASET_TYPE_ALL = [
    'expert',
    'medium-expert',
    'medium',
    'random',
]

DATASET_TYPE_CORE = [
    'medium-expert',
    'medium',
]

DATASET_TYPE_ALTER = [
    'expert',
    'random',
]

model_dataset_bc = [OPE_BC_MODELS, DATASET_TYPE_CORE, ['offline']]
model_dataset_cql = [OPE_CQL_MODELS, DATASET_TYPE_CORE, ['offline']]
model_dataset_iql = [OPE_IQL_MODELS, DATASET_TYPE_CORE, ['offline']]
model_dataset_ours = [OPE_OURS_MODELS, DATASET_TYPE_CORE, ['offline']]
model_dataset_sac = [OPE_SAC_MODELS, ['expert'], ['online']]

model_dataset_bc_sweep = [OPE_BC_MODELS_SWEEP, DATASET_TYPE_ALL]
model_dataset_cql_sweep = [OPE_CQL_MODELS_SWEEP, DATASET_TYPE_ALL]
model_dataset_iql_sweep = [OPE_IQL_MODELS_SWEEP, DATASET_TYPE_ALL]

POLICIES = []
POLICIES += list(itertools.product(*model_dataset_bc))
POLICIES += list(itertools.product(*model_dataset_cql))
POLICIES += list(itertools.product(*model_dataset_iql))
POLICIES += list(itertools.product(*model_dataset_ours))
POLICIES += list(itertools.product(*model_dataset_sac))

# POLICIES += list(itertools.product(*model_dataset_bc_sweep))
# POLICIES += list(itertools.product(*model_dataset_cql_sweep))
# POLICIES += list(itertools.product(*model_dataset_iql_sweep))


if __name__ == '__main__':
    print("\n".join(list(map(str, POLICIES))))
    print(len(POLICIES))