# rl-ticket
This is the repository for my bachelor's thesis named "Pruning Networks Used in Reinforcement Learning"
**Disclaimer:** this is an addition to Ilya Kostrikov's brilliant [implementation](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) of a bunch of algorithms.
From these I only use PPO with GAE.

# Adittions
While the algorithms are from Ilya's repo, I added some functionality:
- some minor hyperparameters options such as weight decay and 
- better (for the scope of my thesis) logging and saving utilities.
- some utilities for searching directories and loading networks with the best avg value (if the value is the last number in the filename)
- the whole pruning functionality, only using a part of the original "main" function
