import copy
import glob
import os
import time
from collections import deque
import json

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
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():
    args = get_args()

    # set seeds and devices
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

    # log_dir = os.path.expanduser(args.log_dir)
    log_dir = utils.default_log_init(args.log_dir, args.env_name)
    save_dir = utils.default_save_init(log_dir, args.save_dir)
    args_file = utils.default_args_init(log_dir, args)

    threads_dir = log_dir + "threads/"
    os.makedirs(threads_dir)
    logger.configure(log_dir)

    print(log_dir)

    # utils.cleanup_log_dir(log_dir)
    eval_log_dir = log_dir + "_eval"
    # utils.cleanup_log_dir(eval_log_dir)
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, threads_dir, device, False)
    action_sample = envs.action_space.sample()
    def init_alg():
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy}).to(device)
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
        return actor_critic, agent, rollouts

    actor_critic, agent, rollouts = init_alg()
    print(actor_critic)

    print(actor_critic.num_params)

    # if args.algo == 'a2c':
    #     agent = algo.A2C_ACKTR(
    #         actor_critic,
    #         args.value_loss_coef,
    #         args.entropy_coef,
    #         lr=args.lr,
    #         eps=args.eps,
    #         alpha=args.alpha,
    #         max_grad_norm=args.max_grad_norm)
    # elif args.algo == 'ppo':

    # elif args.algo == 'acktr':
    #     agent = algo.A2C_ACKTR(
    #         actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)
    # if args.gail:
    #     assert len(envs.observation_space.shape) == 1
    #     discr = gail.Discriminator(
    #         envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
    #         device)
    #     file_name = os.path.join(
    #         args.gail_experts_dir, "trajs_{}.pt".format(
    #             args.env_name.split('-')[0].lower()))
    #
    #     expert_dataset = gail.ExpertDataset(
    #         file_name, num_trajectories=4, subsample_frequency=20)
    #     drop_last = len(expert_dataset) > args.gail_batch_size
    #     gail_train_loader = torch.utils.data.DataLoader(
    #         dataset=expert_dataset,
    #         batch_size=args.gail_batch_size,
    #         shuffle=True,
    #         drop_last=drop_last)

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
    log_dict = {"min_rewards": min_rewards, "max_rewards": max_rewards,
                "mean_rewards": mean_rewards, "median_rewards": median_rewards,
                "nr_episodes": nr_episodes, "times": times, "num_total_steps": num_total_steps}

    # init convergence checks and other useful variables
    best_avg = -1e6
    best_med = -1e6
    since_improve = 0
    solved = 0
    epochs = int(args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(1, epochs + 1):

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

        # if args.gail:
        #     if j >= 10:
        #         envs.venv.eval()
        #
        #     gail_epoch = args.gail_epoch
        #     if j < 10:
        #         gail_epoch = 100  # Warm up
        #     for _ in range(gail_epoch):
        #         discr.update(gail_train_loader, rollouts,
        #                      utils.get_vec_normalize(envs)._obfilt)
        #
        #     for step in range(args.num_steps):
        #         rollouts.rewards[step] = discr.predict_reward(
        #             rollouts.obs[step], rollouts.actions[step], args.gamma,
        #             rollouts.masks[step])

        # training iteration
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        # if (j % args.save_interval == 0 or j == epochs- 1) and save_dir != "":
        #     save_path = os.path.join(save_dir, args.algo)
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass
        #
        #     torch.save([actor_critic,
        #                 getattr(utils.get_vec_normalize(envs), 'ob_rms', None)],
        #                os.path.join(save_path, args.env_name + ".pt"))
        #



        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end_time = time.time()
            s_total = end_time - abs_start
            print(
                "Updates(epochs) {}, num timesteps {}, elapsed {:01}:{:02}:{:02.2f} epoch seconds {} \n Last {} training episodes: "
                "mean/median reward {:.1f}/{:.1f},min/max reward {:.1f}/{:.1f}\n "
                    .format(j, total_num_steps,
                            int(s_total//3600), int(s_total%3600//60), s_total%60,
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

        if (j % args.save_interval == 0 or j == epochs - 1) and save_dir != "":
            save_path = "{}it{}_val{:.1f}.pth".format(save_dir, j, np.mean(episode_rewards))
            torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'ob_rms', None)], save_path)
            print("-------Saved at path {}-------\n".format(save_path))
            # print(save_path+"it_{}_log.json")
            with open(save_dir+"it_{}_log.json".format(j),"w") as file:
                json.dump(log_dict,file)

        if args.convergence_its != 0:
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
                if since_improve > args.convergence_its:
                    print("No improvements in {} iterations, best average is {}, best median is {}, stopping training"
                          .format(since_improve, best_avg, best_med))
                    save_path = "{}it{}_val{:.1f}_c.pth".format(save_dir, j, np.mean(episode_rewards))
                    print("Saved final model at {}".format(save_path))
                    torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'ob_rms', None)], save_path)
                    return

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()



