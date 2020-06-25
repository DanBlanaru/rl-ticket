import copy
import glob
import os
import time
from collections import deque
import json
import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
from baselines import logger

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, MLPBase
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
import argparse
import re

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--init-dir',
    help='dir to start up from'
)
parser.add_argument(
    '--multirun',
    action = 'store_true',
    default = False,
    help = "if we have multiple runs in the directory"
)

parser.add_argument(
    '--prune-together',
    action='store_true',
    default=False,
    help='if actor and critic should be pruned at the same time'

)
parser.add_argument(
    '--prune-convergence-its',
    default=30,
    type=int,
    help='iterations of no improvement to prune'
)
parser.add_argument(
    '--ratio',
    default=0.1,
    type=float,
    help='percentage of weights to prune every step'
)
parser.add_argument(
    '--max-epochs',
    default = 10**8,
    type=int,
    help='number of maximum epochs to train'
)
parser.add_argument(
    '--threshold',
    default = -1e8,
    type = float,
    help = 'minumum reward to stop pruning if reached'
)
prune_args = parser.parse_args()

def detect_best_run(game_dir):
    number_finder = re.compile('([-+]?\d*[.,]?\d)')
    best_avg = -1e8
    best_path = None
    best_seed = 0
    for seed_name in os.listdir(game_dir):
        save_dir = os.path.join(game_dir,seed_name,"nets")
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
                best_path = os.path.join(save_dir,"it_{}_log.json".format(it))
                best_seed = int(seed_name)
#                 print(best_path)
    assert(best_path is not None)
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


def init(prune_args):
    base_dir = prune_args.init_dir
    args_file = os.path.join(base_dir, 'args.json')
    with open(args_file, "r") as fp:
        args_dict = argparse.Namespace(**json.load(fp))
    log_dir = os.path.join(base_dir, "pruning")
    log_dir = os.path.join(log_dir, str(utils.experiment_number(log_dir)))
    os.makedirs(log_dir)

    prune_args_file = os.path.join(log_dir,"prune_args.json")
    with open(prune_args_file,'w') as fp:
        json.dump(vars(prune_args),fp,indent=4,sort_keys = True)

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




if prune_args.multirun:
    prune_args.init_dir = os.path.join(prune_args.init_dir,detect_best_run(prune_args.init_dir))
print(prune_args.init_dir)

args, log_dir, save_dir, threads_dir, eval_log_dir = init(prune_args)
print(log_dir)
print(save_dir)
print(threads_dir)
model_avg, model_path = detect_best_path(os.path.join(prune_args.init_dir, "nets/"))
print(model_path)

# set seeds and devices
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.set_num_threads(1)
device = torch.device("cuda:0" if args.cuda else "cpu")

# if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

logger.configure(log_dir)

envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                     args.gamma, threads_dir, device, False)


def custom_loader(model_good, model_old):
    # for
    for lg, lo in zip(model_good.named_parameters(), model_old.named_parameters()):
        lg[1].data = lo[1].data.clone()


def load_alg():
    actor_critic_2, ob_rms = \
        torch.load(os.path.join(model_path))
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy}).to(device)
    # print([l[0] for l in actor_critic_2.named_parameters()])
    # print([l[0] for l in actor_critic_2.named_parameters()])
    custom_loader(actor_critic, actor_critic_2)
    # print(actor_critic.dist.linear.weight)
    # print(actor_critic_2.dist.fc_mean.weight)
    del actor_critic_2

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    return actor_critic, ob_rms, agent, rollouts


actor_critic, ob_rms, agent, rollouts = load_alg()
vec_norm = utils.get_vec_normalize(envs)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms
print(actor_critic)

# init observations and rollouts
obs = envs.reset()
rollouts.obs[0].copy_(obs)
rollouts.to(device)

# init train loggers(not useful for the actual training, but for analysis)
episode_rewards = deque(maxlen=args.average_over)
start_time = time.time()
abs_start = start_time
min_rewards = []
max_rewards = []
mean_rewards = []
median_rewards = []
nr_episodes = []
times = []
num_total_steps = []
prune_ratios = []
converged_scores = []
log_dict = {"min_rewards": min_rewards, "max_rewards": max_rewards,
            "mean_rewards": mean_rewards, "median_rewards": median_rewards,
            "nr_episodes": nr_episodes, "times": times, "num_total_steps": num_total_steps,
            "prune_ratio": prune_ratios, "converged_score": converged_scores}

# init convergence checks and other useful variables
action_sample = envs.action_space.sample()
prunable_params = actor_critic.pruned_number()[0]
best_avg = -1e6
best_med = -1e6
since_improve = 0
solved = 0
epochs = int(args.num_env_steps) // args.num_steps // args.num_processes
j = 0

# pruning parameters
prune_round = 0  # 0 for actor, 1 for critic, 3 for base for iterative pruning
if actor_critic.base.__class__ == MLPBase:
    num_prune_rounds = 2  # only actor and critic
else:
    num_prune_rounds = 3  # actor, critic and base
prune_net(actor_critic, prune_args, prune_round, num_prune_rounds)

prune_round += 1
current_pruned = actor_critic.pruned_number()[1].cpu().item()
prune_ratios.append(current_pruned / prunable_params)
converged_scores.append(model_avg)

while True:
    j += 1
    if args.use_linear_lr_decay:
        # decrease learning rate linearly
        utils.update_linear_schedule(
            agent.optimizer, j, epochs,
            agent.optimizer.lr if args.algo == "acktr" else args.lr)

    for step in range(args.num_steps):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                rollouts.masks[step])

        # Obser reward and next obs

        
        if type(action_sample) is int:
            obs, reward, done, infos = envs.step(action.squeeze())
        else:
            obs, reward, done, infos = envs.step(action)

        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

        # If done then clean the history of observations.
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0]
             for info in infos])
        rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks)

    with torch.no_grad():
        next_value = actor_critic.get_value(
            rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()

    rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                             args.gae_lambda, args.use_proper_time_limits)

    value_loss, action_loss, dist_entropy = agent.update(rollouts)

    rollouts.after_update()

    if j % args.log_interval == 0 and len(episode_rewards) > 1:
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        end_time = time.time()
        s_total = end_time - abs_start
        print(
            "Updates(epochs) {}, num timesteps {}, pruned ratio {:.3f} elapsed {:01}:{:02}:{:02.2f} epoch seconds {:.3f} \n"
            "Last {} training episodes: mean/median reward {:.1f}/{:.1f},min/max reward {:.1f}/{:.1f}\n "
                .format(j, total_num_steps, current_pruned / prunable_params,
                        int(s_total // 3600), int(s_total % 3600 // 60), s_total % 60,
                        end_time - start_time, len(episode_rewards),
                        np.mean(episode_rewards), np.median(episode_rewards),
                        np.min(episode_rewards), np.max(episode_rewards),
                        dist_entropy, value_loss,
                        action_loss), flush=True)
        min_rewards.append(np.min(episode_rewards))
        max_rewards.append(np.max(episode_rewards))
        mean_rewards.append(np.mean(episode_rewards))
        median_rewards.append(np.median(episode_rewards))
        nr_episodes.append(total_num_steps)
        times.append(end_time - start_time)
        num_total_steps.append(total_num_steps)
        # print(log_dict)
        start_time = end_time

    # if (j % args.save_interval == 0 or j == epochs - 1) and save_dir != "":
    #     save_path = "{}it{}_val{:.1f}.pth".format(save_dir, j, np.mean(episode_rewards))
    #     torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'ob_rms', None)], save_path)
    #     print("-------Saved at path {}-------\n".format(save_path))
    #     # print(save_path+"it_{}_log.json")
    #     with open(save_dir + "it_{}_log.json".format(j), "w") as file:
    #         json.dump(log_dict, file)

    worse = True
    if best_avg < np.mean(episode_rewards):
        best_avg = np.mean(episode_rewards)
        since_improve = 0
        worse = False
    if best_med < np.median(episode_rewards):
        best_med = np.median(episode_rewards)
        since_improve = 0
        worse = False

    if worse:
        since_improve += 1
        if since_improve > prune_args.prune_convergence_its:
            prune_ratios.append(current_pruned)
            converged_scores.append(np.mean(episode_rewards))
            print("Net density: {} out of {} prunable converged".format(current_pruned, prunable_params))
            print("No improvements in {} iterations, best average is {}, best median is {}, pruning"
                  .format(since_improve, best_avg, best_med))
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            pre_prune_eval = evaluate(actor_critic, ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                                      device, deterministic=False, action_sample = action_sample)

            save_path = "{}it{}_pruned{:.3f}_val{:.1f}_pre.pth".format(save_dir, j, current_pruned / prunable_params,
                                                                       pre_prune_eval)
            print("Saved pruned model at pre pruning at {}".format(save_path))
            torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'ob_rms', None)], save_path)
            with open(save_dir + "it_{}_log_pre.json".format(j), "w") as file:
                json.dump(log_dict, file)

            prune_net(actor_critic, prune_args, prune_round, num_prune_rounds)
            prune_round = (prune_round + 1) % num_prune_rounds

            post_prune_eval = evaluate(actor_critic, ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                                       device, deterministic=False,action_sample = action_sample)
            save_path = "{}it{}_pruned{:.3f}_val{:.1f}_post.pth".format(save_dir, j, current_pruned / prunable_params,
                                                                        post_prune_eval)
            print("Saved pruned model at post pruning at {}".format(save_path))
            torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'ob_rms', None)], save_path)
            with open(save_dir + "it_{}_log_pre.json".format(j), "w") as file:
                json.dump(log_dict, file)


            if np.mean(episode_rewards) < prune_args.threshold:
                print("Reward under threshold")
                quit()


            since_improve = 0
            best_avg = -1e6
            best_med = -1e6
            episode_rewards.clear()
            if actor_critic.pruned_number()[1].cpu().item() == current_pruned:
                print("Cant prune further with this ratio")
                prune_args.ratio *= 2
            current_pruned = actor_critic.pruned_number()[1].cpu().item()
        if current_pruned / prunable_params > 0.99 or j > prune_args.max_epochs:
            pre_prune_eval = evaluate(actor_critic, ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                                      device, deterministic=False, action_sample = action_sample)

            save_path = "{}it{}_pruned{:.3f}_val{:.1f}_final.pth".format(save_dir, j, current_pruned / prunable_params,
                                                                       pre_prune_eval)
            print("Saved pruned model at pre pruning at {}".format(save_path))
            torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'ob_rms', None)], save_path)
            with open(save_dir + "it_{}_final.json".format(j), "w") as file:
                json.dump(log_dict, file)
            quit()
            
    # if (args.eval_interval is not None and len(episode_rewards) > 1
    #         and j % args.eval_interval == 0):
    #     ob_rms = utils.get_vec_normalize(envs).ob_rms
    #     evaluate(actor_critic, ob_rms, args.env_name, args.seed,
    #              args.num_processes, eval_log_dir, device)
