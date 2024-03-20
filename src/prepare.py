import pandas as pd
from io import StringIO
import sys
import logging

from pandas.core.tools import numeric

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

from imblearn.under_sampling import RandomUnderSampler

logging.info('Fetching data...')

df=pd.read_csv("./Data/winequality-red.csv")

logging.info('modificacion de data...')

df.replace({(1,2,3,4,5): 0, (6,7,8,9): 1},inplace=True)

logging.info('separando de data...')

X=df.drop("quality",axis=1)
y=df.quality

logging.info('Reduccion de data...')

undersample = RandomUnderSampler(random_state=42)
X_over , y_over = undersample.fit_resample(X,y)

logging.info('Uniendo de data...')

df_dea = X_over
df_dea['quality'] = y_over

logging.info('almacenando la data...')

df_dea.to_csv("./Data/full_data.csv",index=False)


