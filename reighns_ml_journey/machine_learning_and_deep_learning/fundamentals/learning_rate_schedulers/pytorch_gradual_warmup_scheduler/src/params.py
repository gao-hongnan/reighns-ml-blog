from typing import *

# Parameter settings for different Schedulers.

scheduler_step_lr: Dict = {"step_size": 10,
                           "gamma": 0.1, "last_epoch": -1, "verbose": False}

scheduler_cosine_annealing_warm_restart: Dict = {"T_0": 200,
                                                 "T_mult": 1,
                                                 "eta_min": 0.001,
                                                 "last_epoch": -1,
                                                 "verbose": False}
initial_lr: float = 10
num_epochs: int = 30
warmup_epochs: int = 10
step_size: float = 10
gamma: float = 0.1
multiplier: int = 1
last_epoch: int = -1
is_plot: bool = False
