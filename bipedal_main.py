import gym
import random
import torch
import torch.nn.utils.prune as prune
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


from  collections  import deque
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

seed = 0
gamma=0.99
num_processes=3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

envs = parallelEnv('BipedalWalker-v3', n=num_processes, seed=seed)

## make_vec_envs -cannot find context for 'forkserver'
## forkserver is only available in Python 3.4+ and only on some Unix platforms (not on Windows).
## envs = make_vec_envs('BipedalWalker-v2', \
##                    seed + 1000, num_processes,
##                    None, None, False, device='cpu', allow_early_resets=False)

max_steps = envs.max_steps
print('max_steps: ', max_steps)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)




## model Policy uses MLPBase
policy = Policy(envs.observation_space.shape, envs.action_space,False,\
        base_kwargs={'recurrent': False})
print(policy)
# prune.ln_structured(policy.base.actor[2], name = "weight", amount = .1, n =1, dim = 0)
policy.to(device)
agent = ppo_agent(actor_critic=policy, ppo_epoch=16, num_mini_batch=16,\
                lr=0.001, eps=1e-5, max_grad_norm=0.5)

rollouts = RolloutStorage(num_steps=max_steps, num_processes=num_processes, \
                        obs_shape=envs.observation_space.shape, action_space=envs.action_space, \
                        recurrent_hidden_state_size=policy.recurrent_hidden_state_size)

obs = envs.reset()
print('type obs: ', type(obs), ', shape obs: ', obs.shape)
obs_t = torch.tensor(obs)
print('type obs_t: ', type(obs_t), ', shape obs_t: ', obs_t.shape)

rollouts.obs[0].copy_(obs_t)
rollouts.to(device)


def save(model, directory, filename, suffix):
    torch.save(model.base.actor.state_dict(), '%s/%s_actor_%s.pth' % (directory, filename, suffix))
    torch.save(model.base.critic.state_dict(), '%s/%s_critic_%s.pth' % (directory, filename, suffix))
    torch.save(model.base.critic_linear.state_dict(), '%s/%s_critic_linear_%s.pth' % (directory, filename, suffix))


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


num_updates = 1000000
gamma = 0.99
tau = 0.95
save_interval = 30
log_interval = 1


def ppo_vec_env_train(envs, agent, policy, num_processes, num_steps, rollouts):
    time_start = time.time()

    n = len(envs.ps)
    envs.reset()

    # start all parallel agents
    print('Number of agents: ', n)
    # envs.step([[1] * 4] * n)

    envs.step([envs.sample()] * n)
    indices = []
    for i in range(n):
        indices.append(i)

    s = 0

    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    for i_episode in range(num_updates):

        total_reward = np.zeros(n)
        timestep = 0

        for timestep in range(num_steps):
            with torch.no_grad():
                value, actions, action_log_prob, recurrent_hidden_states = \
                    policy.act(
                        rollouts.obs[timestep],
                        rollouts.recurrent_hidden_states[timestep],
                        rollouts.masks[timestep])
            # print(actions.cpu().detach().numpy().s)
            obs, rewards, done, infos = envs.step(actions.cpu().detach().numpy().squeeze())

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
            print('Ep. {}, Timesteps {}, Score.Agents: {:.2f}, Avg.Score: {:.2f}, Time: {:02}:{:02}:{:02}, \
Interval: {:02}:{:02}' \
                  .format(i_episode, timestep + 1, \
                          avg_total_reward, avg_score, s // 3600, s % 3600 // 60, s % 60, t_del % 3600 // 60,
                          t_del % 60))

        if len(scores_deque) == 100 and np.mean(scores_deque) > 300.5:
            # if np.mean(scores_deque) > 20:
            print('Environment solved with Average Score: ', np.mean(scores_deque))
            break

    return scores_array, avg_scores_array

scores, avg_scores = ppo_vec_env_train(envs, agent, policy, num_processes, max_steps, rollouts)
