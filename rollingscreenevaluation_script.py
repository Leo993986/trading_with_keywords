import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from multiprocessing import Pool
from rollingscreenevaluation import rollingscreenevaluation
from openpyxl import Workbook


algorithms = ['PPO2']
policies = ['MlpPolicy']

process_pool = Pool(processes=6, maxtasksperchild=2)
wb = Workbook()

for algorithm in algorithms:
    for policy in policies:
        ws = wb.create_sheet(algorithm + '-' + policy)

        result = [
                    # process_pool.apply_async(rollingscreenevaluation, ('VT', algorithm, policy)),
                    # process_pool.apply_async(rollingscreenevaluation, ('QQQ', algorithm, policy)),
                    process_pool.apply_async(rollingscreenevaluation, ('AXP', algorithm, policy))
                ]
        print(result)
        result_array = result[0].get()
        print(result_array[18])
        print(result_array)
        print(len(result_array))
        
        ans1 = ['trading'] 
        title1 = ['']
        #ans2 = ['trading_brnchmark'] 
        #title2 = [''] 
        for i in  range(16*10):
          title1 += ['trading_' + str(i)]
          ans1 += [result_array[i+1]]
          #title2 += ['trading_brnchmark_' + str(i)]
          #ans2 += [result_array[i*2+2]]

        ws.append(title1)
        ws.append(ans1)
        #ws.append(title2)
        #ws.append(ans2)
        ws.append(['sharpe4years'])
        ws.append([result_array[18]])
        
        wb.save('fintechVTI.xlsx')

process_pool.close()
process_pool.join()
