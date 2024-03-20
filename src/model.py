
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib 
from sklearn.linear_model import LogisticRegression
import logging
import sys
import numpy as np
import pandas as pd

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logger.info('Loading Data...')
df = pd.read_csv('./Data/full_data.csv')

logger.info('Loading model...')


X_over=df.drop("quality",axis=1)
y_over=df.quality

logger.info('Separando Modelo...')
X_train, X_test, y_train , y_test = train_test_split(X_over, y_over, random_state=42, shuffle=True, test_size=0.3)

logger.info('Haciendo modelo...')

model = LogisticRegression(random_state=42, solver= 'liblinear', multi_class= 'ovr', C=1 )
model.fit(X_train, y_train)

logger.info('Loading model...')
joblib.dump(model,"./model/best_model.pkl")

logger.info('Training Finished')
