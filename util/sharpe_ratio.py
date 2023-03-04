import numpy as np
import pandas as pd

def sharpe_ratio(df, years):
    pct_change_Close = df['Close'].pct_change()
    pct_change_TNX = df['TNX'].pct_change()   
    pct_change = pct_change_Close - pct_change_TNX
    mean = pct_change.mean()
    standard_deviation = pct_change.std()
    variance = pct_change.var()
    # (2 ** 0.5)代表testing的總年數為2
    sharpe = (mean / standard_deviation) * (years ** 0.5)


    return sharpe, mean, variance , pct_change_Close
