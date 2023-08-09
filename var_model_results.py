import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

for filename in os.listdir('data'):
    print('#' * 80)
    print(filename)
    df = pd.read_csv('data/'+filename)
    model = VAR(df)
    model_fitted = model.fit(1)

    # Fazer previsões para o próximo período
    forecast = model_fitted.forecast(df.values, steps=1)

    # Calcular os erros relativos
    relative_errors = np.abs((forecast - df.values[-1]) / df.values[-1])

    # Calcular o MMRE
    mmre = np.mean(relative_errors)

    print("Mean Magnitude of Relative Errors (MMRE):", mmre)
    print('#' * 80)
