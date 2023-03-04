import ta


def create_indicators(df):
    df['MFI'] = ta.money_flow_index(
        df["High"], df["Low"], df["Close"], df["Volume"])
    df['TSI'] = ta.tsi(df["Close"])
    df['Trix'] = ta.trix(df["Close"])
    df['DPO'] = ta.dpo(df["Close"])
    df['BBLI'] = ta.bollinger_lband_indicator(
        df["Close"])
    df['DCHI'] = ta.donchian_channel_hband_indicator(df["Close"])
    df['ADI'] = ta.acc_dist_index(df["High"],
                                  df["Low"],
                                  df["Close"],
                                  df["Volume"])
    df['FI'] = ta.force_index(df["Close"],
                            df["Volume"])
    df['DR'] = ta.daily_return(df["Close"])
            
    # momentum
    # df['RSI'] = ta.rsi(df["Close"])
    # df['MACD_diff'] = ta.macd_diff(df["Close"])
    # df['CCI'] = ta.cci(df["High"], df["Low"], df["Close"])

    # df['UO'] = ta.uo(df["High"], df["Low"], df["Close"])
    # df['AO'] = ta.ao(df["High"], df["Low"])

    # df['Vortex_pos'] = ta.vortex_indicator_pos(
    #     df["High"], df["Low"], df["Close"])
    # df['Vortex_neg'] = ta.vortex_indicator_neg(
    #     df["High"], df["Low"], df["Close"])
    # df['Vortex_diff'] = abs(
    #     df['Vortex_pos'] -
    #     df['Vortex_neg'])

    # df['Mass_index'] = ta.mass_index(df["High"], df["Low"])
  
    # df['KST'] = ta.kst(df["Close"])
    # df['KST_sig'] = ta.kst_sig(df["Close"])
    # df['KST_diff'] = (
    #     df['KST'] -
    #     df['KST_sig'])
    # df['Aroon_up'] = ta.aroon_up(df["Close"])
    # df['Aroon_down'] = ta.aroon_down(df["Close"])
    # df['Aroon_ind'] = (
    #     df['Aroon_up'] -
    #     df['Aroon_down']
    # )

    # df['BBH'] = ta.bollinger_hband(df["Close"])
    # df['BBL'] = ta.bollinger_lband(df["Close"])
    # df['BBM'] = ta.bollinger_mavg(df["Close"])
    # df['BBHI'] = ta.bollinger_hband_indicator(
    #     df["Close"])

    # df['KCHI'] = ta.keltner_channel_hband_indicator(df["High"],
    #                                                 df["Low"],
    #                                                 df["Close"])
    # df['KCLI'] = ta.keltner_channel_lband_indicator(df["High"],
    #                                                 df["Low"],
    #                                                 df["Close"])

    # df['DCLI'] = ta.donchian_channel_lband_indicator(df["Close"])

    
    # df['OBV'] = ta.on_balance_volume(df["Close"],
                                    #  df["Volume"])
    # df['CMF'] = ta.chaikin_money_flow(df["High"],
    #                                   df["Low"],
    #                                   df["Close"],
    #                                   df["Volume"])

    # df['EM'] = ta.ease_of_movement(df["High"],
    #                                df["Low"],
    #                                df["Close"],
    #                                df["Volume"])
    # df['VPT'] = ta.volume_price_trend(df["Close"],
    #                                   df["Volume"])
    # df['NVI'] = ta.negative_volume_index(df["Close"],
    #                                      df["Volume"])

    # df['DLR'] = ta.daily_log_return(df["Close"])

    # df.fillna(method='bfill', inplace=True)

    return df

def create_indicators_with_momentum(df):
    df['MFI'] = ta.money_flow_index(
        df["High"], df["Low"], df["Close"], df["Volume"])
    df['TSI'] = ta.tsi(df["Close"])
    df['Trix'] = ta.trix(df["Close"])
    df['DPO'] = ta.dpo(df["Close"])
    df['BBLI'] = ta.bollinger_lband_indicator(
        df["Close"])
    df['DCHI'] = ta.donchian_channel_hband_indicator(df["Close"])
    df['ADI'] = ta.acc_dist_index(df["High"],
                                  df["Low"],
                                  df["Close"],
                                  df["Volume"])
    df['FI'] = ta.force_index(df["Close"],
                            df["Volume"])
    df['DR'] = ta.daily_return(df["Close"])
            
    # momentum
    df['RSI'] = ta.rsi(df["Close"])
    df['MACD_diff'] = ta.macd_diff(df["Close"])
    df['CCI'] = ta.cci(df["High"], df["Low"], df["Close"])

    return df