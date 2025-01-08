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
    realVel_idx = []
    realEff_idx = []

    for i in range(22):
        tarPos_idx.append(0 + i * 5)
        realPos_idx.append(1 + i * 5)
        realVel_idx.append(2 + i * 5)
        realEff_idx.append(3 + i * 5)

    tarPos = np.array(df[tarPos_idx].values)
    realPos = np.array(df[realPos_idx].values)
    realVel = np.array(df[realVel_idx].values)
    realEff = np.array(df[realEff_idx].values)


    data = {
        'command':tarPos,
        'real_gcActualq':realPos,
        'real_gcActualv':realVel,
        'real_gcActualeff':realEff,
    }

    np.savez(f'{PATH}/npz/{cnt}.npz', **data)