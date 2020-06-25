import os

template = "python prune.py --init-dir experiments/{}/{}/ " \
           "--multirun --ratio {} --threshold {} --max-epochs 1200 --prune-convergence-its {} {}"
base = "classic_001"
# envs = ["CartPole-v0","Acrobot-v1","Pendulum-v0","MountainCarContinuous-v0"]
# threshold = [10.0,-499.0,-1500.0,0.0]
base = "mujoco_001"
envs = ["HalfCheetah-v2", "Hopper-v2", "Reacher-v2", "Walker2d-v2"]
threshold = [100.0, 100.0, -120.0, 100.0]
# envs = ["Pendulum-v0"]
# threshold = [-2500.0]
# ratio = 0.05
# together = "--prune-together"

# ratio = 0.1
# together = ""

its = 50

exec_list = []

for env, thresh in zip(envs, threshold):
    exec_list.append(template.format(base, env, 0.1, thresh, its, ""))
    exec_list.append(template.format(base, env, 0.05, thresh, its, "--prune-together"))
    print(exec_list[-2])
    print(exec_list[-1])

for command in exec_list:
    print('-' * 50, '\n\n\n\n')
    print(command)
    os.system(command)
