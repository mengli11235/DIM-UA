# State Representation Learning Using an Unbalanced Atlas

This project contains the code for the paper [State Representation Learning Using an Unbalanced Atlas](openreview.net/forum?id=cWdAYDLmPa), based on the code from the benchmark and techniques introduced in the paper [Unsupervised State Representation Learning in Atari](https://arxiv.org/abs/1906.08226). Please visit https://github.com/mila-iqia/atari-representation-learning for detailed instructions on the benchmark.

To run the script:

```bash
python run_probe.py
```

An example of running DIM-UA and setting the environment to Video Pinball, 4 heads and 512 units each, seed 2:

```bash
python run_probe.py --env-name VideoPinballNoFrameskip-v4 --n-head 4 --feature-size 512 --qv --seed 2
```

An example of running ST-DIM and setting the environment to Video Pinball, 512 units, seed 2:

```bash
python run_probe.py --env-name VideoPinballNoFrameskip-v4 --n-head 1 --feature-size 512 --seed 2
```

Running '-UA' described in the paper, and setting the environment to Video Pinball, 4 heads and 512 units each, seed 2:

```bash
python run_probe.py --env-name VideoPinballNoFrameskip-v4 --n-head 4 --feature-size 512 --seed 2
```

Running '+MMD' described in the paper, and setting the environment to Video Pinball, 4 heads and 512 units each, seed 2:

```bash
python run_probe.py --env-name VideoPinballNoFrameskip-v4 --n-head 4 --feature-size 512 --mmd --seed 2
```

A detailed list of parameter setup is in [atariari/methods/utils.py]
