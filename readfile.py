import sys

sys.path.append('sim/')
sys.path.append('../sim/')
sys.path.append('../')
# from important_ops_simulator import ImportantOpsSimulator
# from grouper.group_pruner import neighbor_merge_pruner
import pickle
import networkx as nx
from sim.tf_placement_sim.tf_pl_simulator import ImportantOpsSimulator
import random


class ReadPickleFile(object):
    def __init__(self, pickled_inp_file, gcluster=None, reset_devs=False, devices_num=1):
        with open(pickled_inp_file, 'rb') as f:
            j = pickle.load(f)
            mg, G, ungroup_map = j['optim_mg'], j['G'], j['ungrouped_mapping']

        # if reset_devs and gcluster:
        #     available_devices = [device.name for device in gcluster.ListDevices()]
        #     for node in mg.graph_def.node:
        #         if node.device is not None:
        #             # node.device = available_devices[random.randint(0, devices_num)]
        #             node.device = available_devices[random.randint(0, devices_num-1)]
        # else:
        #     for node in mg.graph_def.node:
        #         if node.device == '/job:localhost/replica:0/task:0/device:GPU:1' or node.device == '/job:localhost/replica:0/task:0/device:GPU:2' or node.device == '/job:localhost/replica:0/task:0/device:GPU:3':
        #             node.device = '/job:localhost/replica:0/task:0/device:GPU:0'
        self.mg, self.G, self.ungroup_map = mg, G, ungroup_map
        self.j = j

    def getMg(self):
        return self.mg

    def get_triple(self):
        return self.mg, self.G, self.ungroup_map

    def get_stats(self):
        return self.j['step_stats']

    def get_outperform(self):
        return self.j['op_perf'], self.j['step_stats']