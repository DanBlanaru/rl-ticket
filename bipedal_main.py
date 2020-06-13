import gym
import random
import torch
import torch.nn.utils.prune as prune
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from collections import deque
import time
from model import Policy
from ppo import ppo_agent
from storage import RolloutStorage
from utils import get_render_func, get_vec_normalize
from envs import make_vec_envs
from parallelEnv import parallelEnv
import matplotlib.pyplot as plt

print('gym version: ', gym.__version__)
print('torch version: ', torch.__version__)
# prune.ln_structured(policy.base.actor[2], name = "weight", amount = .1, n =1, dim = 0)

hyperparams = {
    "seed": 0,
    "num_processes": 16,
    "device": "cuda:0"
}

bipedal_params = {
    "name": "BipedalWalker-v3",
    "discrete": False,
    "pixel": False
}
continuous_car_params = {
    "name": "MountainCarContinuous-v0",
    "discrete": False,
    "pixel": False,
}
discrete_car_params = {
    "name": "MountainCar-v0",
    "discrete": True,
    "pixel": False
}
cartpole_params = {
    "name": "CartPole-v0",
    "discrete": True,
    "pixel": False
}

env_params = continuous_car_params
p_hyperparams = {
    "hidden_base_size": 100,
    "ppo_epoch": 16,
    "num_mini_batch": 16,
    "lr": 0.001,
    "eps": 1e-5,
    "max_grad_norm": 0.5,
    "weight_decay": 0.002,
    "base_kwargs": {"recurrent": False}
}
t_hyperparams = {
    "num_updates": 1000000,
    "tau": 0.95,
    "save_interval": 30,
    "log_interval": 1,
    "gamma": .99,
}
logger = {}

torch.manual_seed(hyperparams["seed"])
torch.cuda.manual_seed(hyperparams["seed"])
np.random.seed(hyperparams["seed"])
device = torch.device(hyperparams["device"])
print('device: ', device)
print('seed:', hyperparams["seed"])

envs = parallelEnv(env_params["name"], n=hyperparams["num_processes"], seed=hyperparams["seed"])
max_steps = envs.max_steps
print('max_steps: ', max_steps)


def init_algorithm():
    policy = Policy(envs.observation_space.shape, envs.action_space, env_params["discrete"],
                    hidden_base_size= p_hyperparams["hidden_base_size"],base_kwargs=p_hyperparams["base_kwargs"]).to(device)
    agent = ppo_agent(actor_critic=policy, ppo_epoch=p_hyperparams["ppo_epoch"],
                      num_mini_batch=p_hyperparams["num_mini_batch"],
                      lr=p_hyperparams["eps"], eps=p_hyperparams["eps"], max_grad_norm=p_hyperparams["max_grad_norm"],
                      weight_decay=p_hyperparams["weight_decay"])
    rollouts = RolloutStorage(num_steps=max_steps, num_processes=hyperparams["num_processes"],
                              obs_shape=envs.observation_space.shape, action_space=envs.action_space,
                              recurrent_hidden_state_size=policy.recurrent_hidden_state_size)
    return policy, agent, rollouts


policy, agent, rollouts = init_algorithm()
print(policy)

print(policy.recurrent_hidden_state_size)

obs = envs.reset()
print('type obs: ', type(obs), ', shape obs: ', obs.shape)
obs_t = torch.tensor(obs)
print('type obs_t: ', type(obs_t), ', shape obs_t: ', obs_t.shape)

rollouts.obs[0].copy_(obs_t)
rollouts.to(device)


# def save(model, directory, filename, suffix):
#     torch.save(model.base.actor.state_dict(), '%s/%s_actor_%s.pth' % (directory, filename, suffix))
#     torch.save(model.base.critic.state_dict(), '%s/%s_critic_%s.pth' % (directory, filename, suffix))
#     torch.save(model.base.critic_linear.state_dict(), '%s/%s_critic_linear_%s.pth' % (directory, filename, suffix))
def save(model, directory, filename, suffix, object="model"):
    if object == "model":
        print("saved model at :", '%s/%s_network_%s.pth' % (directory, filename, suffix))
        torch.save(model.state_dict(), '%s/%s_network_%s.pth' % (directory, filename, suffix))


limits = [-300, -160, -100, -70, -50, 0, 20, 30, 40, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


def return_suffix(j):
    suf = '0'
    for i in range(len(limits) - 1):
        if j > limits[i] and j < limits[i + 1]:
            suf = str(limits[i + 1])
            break

        i_last = len(limits) - 1
        if j > limits[i_last]:
            suf = str(limits[i_last])
            break
    return suf


def ppo_vec_env_train(envs, agent, policy, num_processes, num_steps, rollouts):
    time_start = time.time()
    envs.reset()

    num_updates = t_hyperparams["num_updates"]
    gamma = t_hyperparams["gamma"]
    tau = t_hyperparams["tau"]
    save_interval = t_hyperparams["save_interval"]
    log_interval = t_hyperparams['log_interval']

    # start all parallel agents
    print('Number of agents: ', num_processes)
    envs.step([envs.sample()] * num_processes)
    s = 0
    solved_reward = envs.env_fns[0].spec.reward_threshold
    solved = 0
    print("Solved at reward:", solved_reward)

    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    for i_episode in range(num_updates):

        total_reward = np.zeros(num_processes)
        timestep = 0

        for timestep in range(num_steps):
            with torch.no_grad():
                value, actions, action_log_prob, recurrent_hidden_states = \
                    policy.act(
                        rollouts.obs[timestep],
                        rollouts.recurrent_hidden_states[timestep],
                        rollouts.masks[timestep])

            np_actions = actions.cpu().detach().numpy()
            if env_params['discrete']:  # discrete environments expect numbers, continuous expect arrays
                np_actions = np_actions.squeeze()

            # print(actions.cpu().detach().numpy().squeeze())
            obs, rewards, done, infos = envs.step(np_actions)

            total_reward += rewards  ## this is the list by agents

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            obs_t = torch.tensor(obs)
            ## Add one dimnesion to tensor, otherwise does not work
            ## This is (unsqueeze(1)) solution for:
            ## RuntimeError: The expanded size of the tensor (1) must match the existing size...
            rewards_t = torch.tensor(rewards).unsqueeze(1)
            rollouts.insert(obs_t, recurrent_hidden_states, actions, action_log_prob, \
                            value, rewards_t, masks)

            # if done.any():
            #     envs.reset()


        avg_total_reward = np.mean(total_reward)
        scores_deque.append(avg_total_reward)
        scores_array.append(avg_total_reward)

        with torch.no_grad():
            next_value = policy.get_value(rollouts.obs[-1],
                                          rollouts.recurrent_hidden_states[-1],
                                          rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, gamma, tau)

        agent.update(rollouts)

        rollouts.after_update()

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        if i_episode > 0 and i_episode % save_interval == 0:
            print('Saving model, i_episode: ', i_episode, '\n')
            suf = return_suffix(avg_score)
            save(policy, 'dir_save', 'we0', suf)
            # save_venv(policy, 'dir_save_VecEnv', 'final')

        if i_episode % log_interval == 0 and len(scores_deque) > 1:
            prev_s = s
            s = (int)(time.time() - time_start)
            t_del = s - prev_s
            print(
                'Ep. {}, Timesteps {}, Score.Agents: {:.2f}, Avg.Score: {:.2f}, Time: {:02}:{:02}:{:02},Interval: {:02}:{:02}'
                    .format(i_episode, timestep + 1, avg_total_reward, avg_score, s // 3600, s % 3600 // 60, s % 60,
                            t_del % 3600 // 60,
                            t_del % 60))

        if len(scores_deque) == 100 and np.mean(scores_deque) > solved_reward:
            if solved == 0:
                print('Environment solved with Average Score: ', np.mean(scores_deque))
                solved = 1
            break
    # envs.close()
    final_time = time.time()
    return scores_array, avg_scores_array

scores, avg_scores = ppo_vec_env_train(envs, agent, policy, hyperparams["num_processes"], max_steps, rollouts)
# scores, avg_scores = ppo_vec_env_train(envs, agent, policy, 1, max_steps, rollouts)
