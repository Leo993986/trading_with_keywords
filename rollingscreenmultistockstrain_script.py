import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from multiprocessing import Pool
from rollingscreenmultistockstrain import rollingscreenmultistockstrain
from openpyxl import Workbook

algorithms = ['PPO2']
policies = ['MlpPolicy']

process_pool = Pool(processes=6, maxtasksperchild=2)
wb = Workbook()

for algorithm in algorithms:
    for policy in policies:
            process_pool.apply_async(rollingscreenmultistockstrain, ('all', algorithm, policy))

process_pool.close()
process_pool.join()
