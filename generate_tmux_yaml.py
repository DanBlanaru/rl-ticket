import argparse

# import yaml
import json

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--num-seeds',
    type=int,
    default=4,
    help='number of random seeds to generate')
parser.add_argument(
    '--env-names',
    default="PongNoFrameskip-v4",
    help='environment name separated by semicolons')
parser.add_argument(
    '--weight-decay',
    type = float,
    default = 0,
    help = "weight decay used for experiments"
)
args = parser.parse_args()

# "Reacher-v2;HalfCheetah-v2;Walker2d-v2;Hopper-v2"
# "CartPole-v0;MountainCarContinuous-v0;Acrobot-v1;Pendulum-v0"
# ppo_mujoco_template = "python main.py --env-name {0} --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-env-steps 1000000 --use-linear-lr-decay --no-cuda --log-dir /tmp/gym/{1}/{1}-{2} --seed {2} --use-proper-time-limits"
ppo_mujoco_template = "python main.py --env-name {0} --algo ppo --use-gae --log-interval 1 --num-env-steps 1000000  --num-steps 2048 --num-processes 1 --weight-decay 0.0005 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --use-linear-lr-decay --seed {2} --use-proper-time-limits --save-interval 10"
# ppo_classic_template = "python main.py --env-name {0} --algo ppo --use-gae --log-interval 1 --num-env-steps 1000000  --num-steps 512 --num-processes 8 --weight-decay 0.00 --lr 3e-4 --entropy-coef 0.01 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --use-linear-lr-decay --seed {2} --save-interval 10"
ppo_classic_template = "python main.py --env-name {0} --algo ppo --use-gae  --num-env-steps 2000000 --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 512 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --save-interval 10 --weight-decay 0.0 --seed {2}"

# ppo_atari_template = "env CUDA_VISIBLE_DEVICES={2} python main.py --env-name {0} --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --log-dir /tmp/gym/{1}/{1}-{2} --seed {2}"
ppo_atari_template = "python main.py --env-name {0} --algo ppo --use-gae --lr 2.5e-4 --weight-decay 0.001 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --seed {2} --save-interval 40"
# png = 'python main.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01'
# 'python main.py --env-name "BattleZone-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01'

template = ppo_classic_template

script_list = []

config = {"session_name": "run-all", "windows": []}

for i in range(args.num_seeds):
    panes_list = []
    for env_name in args.env_names.split(';'):
        # panes_list.append(
        #     template.format(env_name,
        #                     env_name.split('-')[0].lower(), i))
        script_list.append(template.format(env_name, env_name.split('-')[0].lower(), i))

    # config["windows"].append({
    #     "window_name": "seed-{}".format(i),
    #     "panes": panes_list
    # })

# yaml.dump(config, open("run_all2.yaml", "w"), default_flow_style=False)
json.dump(script_list, open("run_all_reg2.json","w"),indent=4)