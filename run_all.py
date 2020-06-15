import json
import os

exec_list = json.load(open("run_all_reg2.json", "r"))

for command in exec_list:
    os.system(command)



# import torch.nn as nn
# import torch.nn.utils.prune as prune
# class Network(nn.Module):
#     def __init__(self, num_in, num_out):
#         super(Network,self).__init__()
#         self.l1 = nn.Linear(num_in,100)
#         self.l2 = nn.Linear(100,num_out)
# net = Network(10,1)
# prune.ln_structured(net.l1, name = "weight", amount = 0.2, n=1, dim = 0)
