# Shot Selection Optimization via Reinforcement Learning

- [Junhyeok Lee](https://github.com/MountGuy)
- [Wookje Han](https://github.com/wookjeHan)
- [Woohyeon Baek](https://github.com/baneling100)

## Overview
One of the important things to be considered in few-shot classification with discrete text prompting is, **which shots should be selected** to yield a good performance. There are some studies to figure out which (e.g. random selection, [closest selection by top-k similarities](https://arxiv.org/abs/2101.06804)), but these have some problems to be resolved. Motivated by [RLPrompt](https://arxiv.org/abs/2205.12548) that attaches MLP module to the gradient-fixed language model and trains it with reinforcement learning, we propose a way to select the shots by a policy constructed by RL.

## How to train and test
Just execute `train.py` with some arguments. You will find these definitions in `ArgumentParser` at the bottom of the codes. Here is an example.
```
python train.py --dataset super_glue,cb --shot_num 2 --batch_size 4 --lr 1e-3 --weight_decay 0.0 --topk 5 --tv_split_ratio 0.0
```
If you want to test it, execute `test.py`. For instance,
```
python test.py --dataset super_glue,cb --batch_size 4
```

## Results
The test set accuracy of random and ours is measured with the setting which showed the best performance on the validation set.

| cb      | random | closest  | ours     |
|---------|--------|----------|----------|
| 2 shots | 32.1   | 46.4     | **64.3** |
| 3 shots | 35.7   | 53.6     | **75**   |
| 4 shots | 53.6   | **57.1** | 46.4     |

| boolq   | random | closest | ours     |
|---------|--------|---------|----------|
| 2 shots | 48.3   | 53.3    | **55**   |
| 3 shots | 51.7   | 53.8    | **60.4** |
| 4 shots | 57.1   | 54.3    | **60.2** |

| sst2    | random | closest | ours     |
|---------|--------|---------|----------|
| 2 shots | 69.7   | 77.3    | **82.8** |
| 3 shots | 73.4   | 75.9    | **82.6** |
| 4 shots | 76.4   | 73.2    | **81.2** |
