# python main.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --log-interval 1 --num-steps 2048
# --num-processes 4 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5
# --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95
# --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits  --save-interval 10
# usage: main.py [-h] [--algo ALGO] [--gail]
#                [--gail-experts-dir GAIL_EXPERTS_DIR]
#                [--gail-batch-size GAIL_BATCH_SIZE] [--gail-epoch GAIL_EPOCH]
#                [--lr LR] [--eps EPS] [--weight-decay WEIGHT_DECAY]
#                [--alpha ALPHA] [--gamma GAMMA] [--use-gae]
#                [--gae-lambda GAE_LAMBDA] [--entropy-coef ENTROPY_COEF]
#                [--value-loss-coef VALUE_LOSS_COEF]
#                [--max-grad-norm MAX_GRAD_NORM] [--seed SEED]
#                [--cuda-deterministic] [--num-processes NUM_PROCESSES]
#                [--num-steps NUM_STEPS] [--ppo-epoch PPO_EPOCH]
#                [--num-mini-batch NUM_MINI_BATCH] [--clip-param CLIP_PARAM]
#                [--log-interval LOG_INTERVAL] [--save-interval SAVE_INTERVAL]
#                [--eval-interval EVAL_INTERVAL] [--num-env-steps NUM_ENV_STEPS]
#                [--env-name ENV_NAME] [--log-dir LOG_DIR] [--save-dir SAVE_DIR]
#                [--no-cuda] [--use-proper-time-limits] [--recurrent-policy]
#                [--use-linear-lr-decay] [--discrete]



arg_dict = {
    "env-name": "PongNoFrameskip-v4",
    "discrete": False,
    "algo": "ppo",
    "num-processes": 16,
    "seed": 1,
    "lr": 3e-4,
    # "eps": 1e5,
    "weight-decay": 0.003,
    "num-env-steps": 10 ** 6,
    "num-steps": 2048,
    # "log-interval": 10,
    "save-interval": 10,
    # "log-dir": "saves/logs",
    "save-dir": "trained/",
    # "gamma":0.99,
    "use-gae": True,
    "gae-lambda": 0.95,
    "entropy-coef": 0.0,
    # "value-loss-coef": 0.5,
    # "max-grad-norm": 0.5,
    "ppo-epoch": 10,
    # "num-mini-batch":32,
    # "clip-param":0.2,
    "use-proper-time-limits": True,
    # "use-linear-lr-decay":False,
    "convergence_its": 100,
}

attributes = [key +' '+ str(value) if type(value) is not bool else key for key,value in arg_dict.items()]
args = ' --'.join([' ']+attributes)
invocation = "python main.py" + args