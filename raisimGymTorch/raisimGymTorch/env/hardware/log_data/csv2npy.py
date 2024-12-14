import pandas as pd
import numpy as np
import os

PATH = '/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/log_data'
files = os.listdir(PATH+'/csv')
csv_files = [file for file in files if file.lower().endswith('.csv')]
CSV_NUM = len(csv_files)

for cnt in range(CSV_NUM):
    df = pd.read_csv(f'{PATH}/csv/{cnt}.csv', skiprows=1, header=None)

    tarPos_idx = []
    realPos_idx = []
    simPos_idx = []
    realVel_idx = []
    simVel_idx = []

    for i in range(22):
        tarPos_idx.append(0 + i * 4)
        realPos_idx.append(2 + i * 4)
        #simPos_idx.append(2 + i * 4)
        realVel_idx.append(3 + i * 4)
        #simVel_idx.append(4 + i * 4)

    tarPos = np.array(df[tarPos_idx].values)
    realPos = np.array(df[realPos_idx].values)
    simPos = np.array(0.0)
    realVel = np.array(df[realVel_idx].values)
    simVel = np.array(0.0)


    data = {
        'command':tarPos,
        'real_gcActualq':realPos,
        'real_gcActualv':realVel,
        'sim_gcActualq':simPos,
        'sim_gcActualv':simVel,
    }

    np.savez(f'{PATH}/npz/{cnt}.npz', **data)