import pandas as pd
import numpy as np
import os
import csv

PATH = '/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/log_data/csv'
df_real = pd.read_csv(f'{PATH}/reconFalse-sim.csv', skiprows=1, header=None)
df_sim = pd.read_csv(f'{PATH}/reconFalse-real.csv', skiprows=1, header=None)
result_diff = df_real - df_sim
result_pre = (df_real - df_sim) / df_real
result_diff.to_csv(f'{PATH}/reconFalse-diff.csv', index=False)

