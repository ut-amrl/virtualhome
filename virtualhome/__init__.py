import glob
import sys
from sys import platform

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
simulation_path = os.path.join(current_dir, 'simulation')
sys.path.append(simulation_path)
from unity_simulator.comm_unity import UnityCommunication
from unity_simulator import utils_viz

current_dir = os.path.dirname(os.path.abspath(__file__))
starbench_path = os.path.join(current_dir, 'starbench')
sys.path.append(starbench_path)
from ros_utils import *
from utils_demo import *
from graph_utils import *