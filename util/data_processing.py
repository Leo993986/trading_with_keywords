import numpy as np

def data_stationary(df, columns):
    '''
    數據平穩化-使用價差比例取代價格
    我們的時間序列包含明顯的趨勢和季節性，這兩者都會影響我們的算法準確預測時間序列的能力。我們可以通過使用差分和變換技術來解決這個問題，從我們現有的時間序列中產生更正常的分佈。
    差分是從該時間步的值減去每個時間步的導數（收益率）的過程。這有消除我們案例趨勢的預期結果，但是，數據仍然具有明確的季節性。我們可以嘗試通過在差分之前的每個時間步長取對數來移除它，這產生最終的靜止時間序列。
    '''
    transformed_df = df.copy()

    for column in columns:
        transformed_df[column] = (transformed_df[column] - transformed_df[column].shift(1))/transformed_df[column].shift(1) * 100

    transformed_df = transformed_df.fillna(method='bfill')

    return transformed_df