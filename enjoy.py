import argparse
import os
# workaround to unpickle olf model files
import sys
import re
import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--detect-path',
    default = None,
    help='detect the best policy from a dir by its pathname'
)
args = parser.parse_args()

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
            print(filename)
            best_path = filename
    print("Best path is:" + filename)
    assert(best_path is not None)
    return os.path.join(save_dir,best_path)

        



args.det = not args.non_det

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
if args.detect_path:
    load_dir = detect_best_path(args.detect_path)
else:
    load_dir = args.load_dir

actor_critic, ob_rms = \
            torch.load(os.path.join(load_dir))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

if render_func is not None:
    render_func('human')

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i
actor_critic.to(torch.device("cpu"))
total_reward = 0
rews = []
while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    # rews.append(reward)
    if done:
        env.reset()
        print("Env done with reward%f"%total_reward)
        # print("len: %f, mean: %f\n"%(len(rews), np.mean(rews)))
        total_reward = 0
    masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')
