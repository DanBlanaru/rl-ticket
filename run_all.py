import json
import os

exec_list = json.load(open("run_all.json", "r"))

for command in exec_list:
    os.system(command)

