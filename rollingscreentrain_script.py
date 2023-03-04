import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
from openpyxl import Workbook
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from multiprocessing import Pool
from rollingscreentrain import rollingscreentrain

algorithms = ['PPO2']
policies = ['MlpPolicy']

process_pool = Pool(processes=6, maxtasksperchild=2)
wb = Workbook

for algorithm in algorithms:
    for policy in policies:
        # process_pool.apply_async(rollingscreentrain, ('VT', algorithm, policy))
        # process_pool.apply_async(rollingscreentrain, ('QQQ', algorithm, policy))
        process_pool.apply_async(rollingscreentrain, ('AXP', algorithm, policy))


process_pool.close()
process_pool.join()
