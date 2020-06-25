import glob
import os
import json
import re
import torch.nn.utils.prune
import argparse
import torch
import torch.nn as nn

from a2c_ppo_acktr.envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def experiment_number(log_dir):
    try:
        return len([name for name in os.listdir(log_dir)])
    except OSError:
        os.makedirs(log_dir)
        return 0


def default_log_init(log_dir, env_name):
    if log_dir is not None:
        if os.path.exists(log_dir):
            return log_dir
        os.makedirs(log_dir)
        return log_dir
    else:
        log_dir = "experiments/" + env_name + '/'
        exp_nr = experiment_number(log_dir)
        log_dir = log_dir + str(exp_nr) + "/"
        os.makedirs(log_dir)
        return log_dir


def default_save_init(log_dir, save_dir, pruning=False):
    if save_dir is not None:
        save_dir = save_dir
    else:
        save_dir = os.path.join(log_dir, "nets/")
    os.makedirs(save_dir)
    return save_dir


def default_args_init(log_dir, args):
    args_file = ("pruning_" if args.pruning else "") + "args.json"
    with open(log_dir + args_file, 'w')  as file:
        json.dump(vars(args), file, indent=4, sort_keys=True)
    return log_dir + args_file


def detect_best_run(game_dir):
    number_finder = re.compile('([-+]?\d*[.,]?\d)')
    best_avg = -1e8
    best_path = None
    best_seed = 0
    for seed_name in os.listdir(game_dir):
        save_dir = os.path.join(game_dir, seed_name, "nets")
        for filename in os.listdir(save_dir):
            basename, extention = os.path.splitext(filename)
            if extention != ".pth":
                continue
            numbers = (number_finder.findall(basename))
            avg = float(numbers[-1])
            it = int(numbers[0])
            if avg > best_avg:
                best_avg = avg
                #                 print(filename)
                #                 print(numbers)
                best_path = os.path.join(save_dir, "it_{}_log.json".format(it))
                best_seed = int(seed_name)
    #                 print(best_path)
    assert (best_path is not None)
    return str(best_seed)


def detect_best_path(save_dir):
    number_finder = re.compile('([-+]?\d*[.,]?\d)')
    best_avg = -1e8
    best_path = None
    for filename in os.listdir(save_dir):
        basename, extention = os.path.splitext(filename)
        if extention != ".pth":
            continue
        avg = float(number_finder.findall(basename)[-1])

        if avg > best_avg:
            best_avg = avg
            best_path = filename
    assert (best_path is not None)
    return best_avg, os.path.join(save_dir, best_path)


def prune_folders_init(prune_args):
    base_dir = prune_args.init_dir
    args_file = os.path.join(base_dir, 'args.json')
    with open(args_file, "r") as fp:
        args_dict = argparse.Namespace(**json.load(fp))
    log_dir = os.path.join(base_dir, "pruning")
    log_dir = os.path.join(log_dir, str(experiment_number(log_dir)))
    os.makedirs(log_dir)

    prune_args_file = os.path.join(log_dir, "prune_args.json")
    with open(prune_args_file, 'w') as fp:
        json.dump(vars(prune_args), fp, indent=4, sort_keys=True)

    save_dir = os.path.join(log_dir, "nets/")
    os.makedirs(save_dir)
    threads_dir = os.path.join(log_dir, "logs/")
    os.makedirs(threads_dir)
    eval_dir = os.path.join(log_dir, 'eval/')
    os.makedirs(eval_dir)
    return args_dict, log_dir, save_dir, threads_dir, eval_dir


def prune_net(actor_critic, prune_args, prune_round, num_prune_rounds):
    if prune_args.prune_together:
        actor_critic.prune_actor([prune_args.ratio] * 3, [0, 1, 1])
        actor_critic.prune_critic([prune_args.ratio] * 3, [0, 1, 1])
    else:
        if prune_round == 0:
            actor_critic.prune_actor([prune_args.ratio] * 3, [0, 1, 1])
            print("pruned actor")
        if prune_round == 1:
            actor_critic.prune_critic([prune_args.ratio] * 3, [0, 1, 1])
            print("pruned critic")
