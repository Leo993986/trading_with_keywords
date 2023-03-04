import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from multiprocessing import Pool
from rollingscreenmultistocksevaluation import rollingscreenmultistocksevaluation
from openpyxl import Workbook


algorithms = ['PPO2']
policies = ['MlpPolicy']

process_pool = Pool(processes=6, maxtasksperchild=2)
wb = Workbook()

for algorithm in algorithms:
    for policy in policies:
        ws = wb.create_sheet(algorithm + '-' + policy)

        result = [
                    process_pool.apply_async(rollingscreenmultistocksevaluation, ('all', algorithm, policy))
                ]

        result_array = result[0].get()

        ans1 = [''] 
        title1 = ['']
        ans2 = [''] 
        title2 = [''] 
        for i in range(16*12):
            title1 += ['trading_' + str(i)]
            ans1 += [result_array[i*2+1]]
            title2 += ['trading_brnchmark_' + str(i)]
            ans2 += [result_array[i*2+2]]

        ws.append(title1)
        ws.append(ans1)
        ws.append(title2)
        ws.append(ans2)
        ws.append(['sharpe4years'])
        ws.append([result_array[33]])

        wb.save('fintechmultiassetpercentagemomentum.xlsx')

process_pool.close()
process_pool.join()
