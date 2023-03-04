# fintech
## build env
### install python3.6

> ```
> sudo add-apt-repository ppa:deadsnakes/ppa
> sudo apt update
> sudo apt install python3.6
> sudo apt install virtualenv 
> sudo apt install libgl1-mesa-glx
> ```

### build_python_env.sh

> ```
> ~/my/fintech/scripts/build_python_env.sh
> ```

## scripts
### run_venv.sh

```
move to fintech directory:
cd ~/my/fintech/
```
> #### train data
> ```
> ./scripts/run_venv.sh python train.py
> ./scripts/run_venv.sh python train_script.py
> ```

> #### evalution model
> ```
> ./scripts/run_venv.sh python evaluation_script.py
> ```

> #### tensorboard
> ```
> # Default port: 6006
> ./scripts/run_venv.sh tensorboard --logdir=./tensorboard/['PPO2', 'A2C', 'ACKTR-PPO', 'ACKTR-A2C']/MlpPolicy/[Data_Name]/ --host=0.0.0.0
> example :
> ./scripts/run_venv.sh tensorboard --logdir=./tensorboard/PPO2/MlpPolicy/DIA/ --host=0.0.0.0
> ./scripts/run_venv.sh tensorboard --logdir=./tensorboard/PPO2/MlpPolicy/VTI2022/15/ --host=0.0.0.0
> ```

> #### open jupyter notebook
> ```
> ./scripts/run_venv.sh jupyter notebook
> ```

> #### clean
> ```
> # delete train results
> ./scripts/clean.sh
> ```

## document

- OpenAI use stable-baselines
  - https://github.com/hill-a/stable-baselines
- reward set in ./env/FintechEnv.py
- data place in ./data
- train_script.py put data into train.py
- train.py 
  - split data into train data,evaluation data and test data 
  - training data  
  - generate model and tensorboard in ./model ./tensorboard
- evaluation_script.py read ./model and write evaluation result in  xml
- evaluation.py evaluation model
- ./util some tools
- test.py just a test