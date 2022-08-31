## gradual warmup 

### Paper Implementation

Here are some parameters in the paper.

1. n = minibatch size
2. k = multiplier of minibatch size
3. For eg, if n = 32, and k = 2, then the new batch size is 64.
4. Intuition: In the first few steps, the LR is set to lower than the base LR, and increased gradually (linearly) to approach it
in a number of epochs.
5. Example:
           - base_lr = 1
           - total_epoch (Warmup Epochs) = 10
           - [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
           - [1 * (0/10) for [all base_lrs in which case here is just 1]] -> [0],[0.1], [0.2] etc
---

**Important**

- Currently, if your warmup epochs is 10, then at both 10 and 11th epoch, the LR will be the target/base learning rate of the optimizer. Ideally, maybe we should start the 11th epoch with the "stepped" scheduler already. This is on my To-Do list.
- To read these known "issues and confusions": [here](https://discuss.pytorch.org/t/get-current-lr-of-optimizer-with-adaptive-lr/24851/5), [here](https://github.com/pytorch/pytorch/pull/26423) and [here](https://github.com/pytorch/pytorch/issues/22107).
- Intuition of gradual warmup is in the paper, to summarize, instead of going to the base learning rate set in the optimizer, we first have a linear warmup period. Say if your base/initial learning rate is 10, then you want to warmup 10 epochs, so during these first 10 epochs, it will be increasing in a linear fashion from 1, 2, 3... to 10. (I am unsure if we should start from 0 though).
- bug workaround: if epoch == scheduler.total_epoch: we step the scheduler once more. This is because if we use the settings in this code, then Epoch 10 and 11 will both be having LR = 10, what we want is at epoch 10 it is LR=10, then epoch 11 it already start to use the scheduler's decay, instead of repeating twice.

## Params

1. `last_epoch`: This means the previous epoch and not the "last epoch | not end of training".

## Methods

1. `scheduler.get_last_lr()[0]` to get the current learning rate before stepping.
2. `scheduler.get_lr()[0]` not to be used to call the current learning rate. See [here](https://github.com/pytorch/pytorch/pull/26423) and [here](https://github.com/pytorch/pytorch/issues/22107).
3. The above two will cause some confusions, but won't impact the training process, only log file may be confusing.

4. `get_lr` for StepLR:
    ```
        def get_lr(self):
            if not self._get_lr_called_within_step:
                warnings.warn("To get the last learning rate computed by the scheduler, "
                            "please use `get_last_lr()`.", UserWarning)

            if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
                return [group['lr'] for group in self.optimizer.param_groups]
            return [group['lr'] * self.gamma
                    for group in self.optimizer.param_groups]
    ```

    This illustrates the issue mentioned [here](https://stackoverflow.com/questions/59599603/about-pytorch-learning-rate-scheduler).

    As an example, since our `last_epoch = -1` as default, 
    
    ```
    1 10 10
    2 10 10
    3 10 10
    4 10 10
    5 10 10
    6 10 10
    7 10 10
    8 10 10
    9 10 10
    10 10 10
    11 0.1 1.0 # See the problem? https://stackoverflow.com/questions/59599603/about-pytorch-learning-rate-scheduler
    12 1.0 1.0
    13 1.0 1.0
    14 1.0 1.0
    15 1.0 1.0
    16 1.0 1.0
    17 1.0 1.0
    18 1.0 1.0
    19 1.0 1.0
    20 1.0 1.0
    21 0.01 0.1 
    22 0.1 0.1
    23 0.1 0.1
    24 0.1 0.1
    25 0.1 0.1
    26 0.1 0.1
    27 0.1 0.1
    28 0.1 0.1
    29 0.1 0.1
    30 0.1 0.1
    ```
