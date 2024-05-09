import pandas as pd
import glob

# Use this script to find which of the pulsars in PSRPi, PSRPi II, and MSPPi are also in
# the NANOGrav 15 yr dataset

psr_list = pd.read_csv('./data/pulsars_list.csv').iloc[:, 0]
par_dir: str = '/media/sophia/Sophia_s Drive/NANOGrav15yr_PulsarTiming/narrowband/par'
res_list = []

for PSR in psr_list:
    if len(glob.glob(par_dir + f"/{PSR}*")) >= 1:
        res_list.append(PSR)

res_list.sort()
print(res_list)
print(len(res_list))
