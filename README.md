# Shot Selection Optimization via Reinforcement Learning

- [Junhyeok Lee](https://github.com/MountGuy)
- [Wookje Han](https://github.com/wookjeHan)
- [Woohyeon Baek](https://github.com/baneling100)

## Overview
One of the important things to be considered in few-shot classification with discrete text prompting is, **which shots should be selected** to yield a good performance. There are some studies to figure out which (e.g. random selection, [closest selection by top-k similarities]()), but these have some problems to be resolved. Motivated by [RLPrompt](https://arxiv.org/abs/2205.12548) that attaches MLP module to the gradient-fixed language model and trains it with reinforcement learning, we propose a way to select the shots by a policy constructed by RL.

## How to train
Just execute `train.py` with some arguments. You will find these definitions in `ArgumentParser` at the bottom of the codes. Here is an example.
```
python train.py --dataset super_glue,cb --shot_num 2 --batch_size 4 --lr 1e-3 --weight_decay 0.0 --topk 5 --tv_split_ratio 0.0
```
If you want to test it, execute `test.py`. For instance,
```
python test.py --dataset super_glue,cb --batch_size 4
```

## Results
