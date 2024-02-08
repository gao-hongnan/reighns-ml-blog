import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.optim.sgd import SGD

import config
import cosine_annealing_warmup
import gradual_warmup
import params


def get_lr(optimizer: torch.optim) -> float:
    """[summary]

    Args:
        optimizer (torch.optim): [description]

    Returns:
        float: [description]
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run_normal_scheduler(run_name: str, num_epochs: int, is_plot: bool, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler) -> None:
    """Normal StepLR scheduler

    Example:
        Assuming the current settings that base lr is 10.
        lr = 10 if 1 <= epoch <= 10
        lr = 1  if 11 <= epoch <= 20
        lr = 0.1 if 21 <= epoch <= 30
        etc

    Args:
        run_name (str): the name of the scheduler I am using
        num_epochs (int): [description]
        optimizer (torch.optim): [description]
        scheduler (torch.optim.lr_scheduler): [description]
    """

    lrs: List = []

    for epoch in range(1, num_epochs+1):
        optimizer.zero_grad()
        curr_epoch_lr = optimizer.param_groups[0]['lr']

        assert np.isclose(a=curr_epoch_lr, b=get_lr(optimizer),
                          rtol=1e-8, atol=1e-8, equal_nan=False) == True,\
            np.isclose(a=curr_epoch_lr, b=scheduler.get_last_lr()[
                0], rtol=1e-8, atol=1e-8, equal_nan=False) == True

        # TODO: DO NOT CALL scheduler.get_lr()[0] FOR CURRENT LR.

        lrs.append(curr_epoch_lr)

        config.logger_normal_scheduler.info(f"Epoch: {int(epoch)} | "
                                            f"LR: {curr_epoch_lr}")

        optimizer.step()
        scheduler.step()

    if is_plot:
        fig = go.Figure(data=go.Scatter(x=np.arange(1, num_epochs+1), y=lrs))
        fig.write_image(os.path.join(config.PLOTS_DIR, f"{run_name}.jpg"))
        # fig.show()


def run_warmup_scheduler(run_name: str, num_epochs: int, is_plot: bool, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler) -> None:

    lrs: List = []

    for epoch in range(1, num_epochs+1):
        optimizer.zero_grad()
        curr_epoch_lr = optimizer.param_groups[0]['lr']
        lrs.append(curr_epoch_lr)

        config.logger_warmup_scheduler.info(f"Epoch: {int(epoch)} | "
                                            f"LR: {curr_epoch_lr}")

        optimizer.step()
        scheduler.step()

        if epoch == scheduler.total_epoch:
            config.logger_warmup_scheduler.info(
                "Workaround by stepping twice. This is to ensure that right after the warmup epcohs end, we can already start the scheduler, see documentation")
            scheduler.step()

    if is_plot:
        fig = go.Figure(data=go.Scatter(x=np.arange(1, num_epochs+1), y=lrs))
        fig.write_image(os.path.join(config.PLOTS_DIR, f"{run_name}.jpg"))
        # fig.show()


if __name__ == '__main__':
    # create a naive model
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]

    initial_lr: float = 10
    num_epochs: int = 30
    warmup_epochs: int = 10
    multiplier: int = 1

    is_plot: bool = True

    # normal_steplr, normal_cosine_annealing_warm_restart , warmup_steplr, warmup_cosine_annealing_warm_restart
    run: str = 'warmup_cosine_annealing_warm_restart'

    optimizer = SGD(model, lr=initial_lr)

    if run == 'normal_steplr':

        scheduler_steplr = StepLR(optimizer, **params.scheduler_step_lr)

        run_normal_scheduler(run_name=run, num_epochs=num_epochs, is_plot=is_plot,
                             optimizer=optimizer, scheduler=scheduler_steplr)

    elif run == 'warmup_steplr':
        # scheduler_warmup is chained with schduler_steplr, so cannot call together if not got wrong LR
        scheduler_steplr = StepLR(optimizer, **params.scheduler_step_lr)
        scheduler_steplr_warmup = gradual_warmup.GradualWarmupScheduler(
            optimizer, multiplier=multiplier, total_epoch=warmup_epochs, after_scheduler=scheduler_steplr)
        run_warmup_scheduler(run_name=run, num_epochs=num_epochs, is_plot=is_plot,
                             optimizer=optimizer, scheduler=scheduler_steplr_warmup)

    elif run == 'normal_cosine_annealing_warm_restart':
        # set T_0 = num_epochs as we do not wish to "restart" -> read OneCycleLR
        # set last_epoch=0 as we want to be in sync with our epoch tracking, i.e., if you start epoch from 1, then based on the source code
        # in pytorch, you need set last_epoch=0 to have correct tracking.

        # Using the settings, the second epoch LR = 0.001+0.5(10-0.001)(1+cos((1/200)pi)) = 9.999383224
        scheduler_cosine = getattr(torch.optim.lr_scheduler, "CosineAnnealingWarmRestarts")(
            optimizer=optimizer, **params.scheduler_cosine_annealing_warm_restart)

        run_normal_scheduler(run_name=run, num_epochs=num_epochs, is_plot=is_plot,
                             optimizer=optimizer, scheduler=scheduler_cosine)

    elif run == 'warmup_cosine_annealing_warm_restart':
        # Epoch: 11 | LR: 9.999383224092062
        # This is correct since we have 10 warmup epochs and therefore at 10th epoch we reach our target of base lr = 10
        # Then 11th epoch should be 0.99938322... according to our exact setup here. Hand calculated to be correct using formula from cosine annealing warm restart.

        print(
            "Both runs should give you same value for the same params, else might be a bug!")
        scheduler_cosine = getattr(torch.optim.lr_scheduler, "CosineAnnealingWarmRestarts")(
            optimizer=optimizer, **params.scheduler_cosine_annealing_warm_restart)
        scheduler_cosine_warmup = gradual_warmup.GradualWarmupScheduler(
            optimizer, multiplier=multiplier, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

        run_warmup_scheduler(run_name=run, num_epochs=num_epochs, is_plot=is_plot,
                             optimizer=optimizer, scheduler=scheduler_cosine_warmup)

        # scheduler_cosine_annealing_restart = cosine_annealing_warmup.CosineAnnealingWarmupRestarts(optimizer,
        #                                                                                            first_cycle_steps=200,
        #                                                                                            cycle_mult=1.0,
        #                                                                                            max_lr=10,
        #                                                                                            min_lr=0.001,
        #                                                                                            warmup_steps=10,
        #                                                                                            gamma=1.0)
        # run_warmup_scheduler(num_epochs=num_epochs, is_plot=is_plot,
        #                      optimizer=optimizer, scheduler=scheduler_cosine_annealing_restart)
