# Third-party code and data

This repository builds on the following works:

## FJSP-DRL / DANIEL (Wang et al., 2023)

The Dual Attention Network encoder (`cdqac/network/attention_layer.py`,
`cdqac/network/sub_layers.py`, parts of `cdqac/network/main_model.py`), the
FJSP simulation environments and instance utilities (`env/`), and the
benchmark instance files (`data/`) are adapted from the official
implementation of:

> Runqing Wang, Gang Wang, Jian Sun, Fang Deng, Jie Chen.
> "Flexible Job Shop Scheduling via Dual Attention Network-Based
> Reinforcement Learning." IEEE Transactions on Neural Networks and Learning
> Systems, 2023. https://doi.org/10.1109/TNNLS.2023.3306421
> https://github.com/wrqccc/FJSP-DRL

## Job Shop Scheduling Benchmark (Reijnen et al., 2023)

GA training trajectories are generated with the genetic algorithm from:

> Robbert Reijnen, Kjell van Straaten, Zaharah Bukhsh, Yingqian Zhang.
> "Job Shop Scheduling Benchmark: Environments and Instances for Learning and
> Non-learning Methods." arXiv:2308.12794, 2023. MIT license.
> https://github.com/ai-for-decision-making-tue/Job_Shop_Scheduling_Benchmark_Environments_and_Instances

`data_generation/run_GA_custom.py` is designed to be run inside a clone of
that repository.

## Benchmark instances

The `data/` directory contains the standard benchmark instance sets of
Brandimarte (1993), Hurink et al. (1994), Taillard (1993), and Demirkol et
al. (1998), together with best-known reference solutions collected from the
literature.
