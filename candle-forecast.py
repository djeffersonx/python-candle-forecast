import pandas as pd

def get_candle_upper_shadow_ratio(row):
  if row['candle_body_size'] > 0:
    return row['high'] / row['close']
  else:
    return row['high'] / row['open'] 

def get_candle_lower_shadow_ratio(row):
  if row['candle_body_size'] > 0:
    return row['open'] / row['low']
  else:
    return row['close'] / row['low'] 
    
def get_candle_color(row):
  if row['candle_body_size'] == 0:
    return 0
  elif row['candle_body_size'] > 0: # verde
    return 1
  else: # vermelho
    return -1

def moving_average(df, n):
    ma = pd.Series(df['close'].rolling(n, min_periods=n).mean(), name='ma_' + str(n))
    df = df.join(ma)
    return df

def prepare_data(df):
  df.rename(columns={'Gmt time':'date_time', 'Volume':'volume', 'Open':'open', 'Close':'close', 'High':'high', 'Low': 'low'},  inplace=True)
  df['date_time'] = pd.to_datetime(df['date_time'], format='%d.%m.%Y %H:%M:%S.%f', )
 
  df = df.drop(df[df['volume'] == 0].index)
  
  df['candle_volatility_ratio'] = df['high']/df['low']  
  df['candle_body_ratio'] = df['close']/df['open']
  df['candle_body_size'] = df['close']-df['open']
  
  df = moving_average(df, 20)
  df = moving_average(df, 100)

  df['ma_20_100_ratio'] = df['ma_20'] / df['ma_100']

  df['candle_upper_shadow_ratio'] = df.apply(get_candle_upper_shadow_ratio, axis=1)
  df['candle_lower_shadow_ratio'] = df.apply(get_candle_lower_shadow_ratio, axis=1)

  df['candle_color'] = df.apply(get_candle_color, axis=1)
  df['next_candle_color'] = df['candle_color'].shift(-1)
  
  return df

df = prepare_data(pd.read_csv('data/eur_usd_h1_2017_2019.csv'))
df_test = prepare_data(pd.read_csv('data/eur_usd_h1_2020.csv'))

df.dropna(inplace=True)
df_test.dropna(inplace=True)

y_train = df['next_candle_color']
x_train = df.loc[:, ['candle_volatility_ratio', 'candle_body_ratio', 'ma_20_100_ratio', 'candle_upper_shadow_ratio', 'candle_lower_shadow_ratio']]

y_test = df_test['next_candle_color']
x_test = df_test.loc[:, ['candle_volatility_ratio', 'candle_body_ratio', 'ma_20_100_ratio', 'candle_upper_shadow_ratio', 'candle_lower_shadow_ratio']]

from sklearn.neighbors import KNeighborsClassifier

ml = KNeighborsClassifier(n_neighbors=1, weights='uniform', leaf_size=10)

ml.fit(x_train, y_train)

pred = ml.predict(x_test)

df_trade = pd.DataFrame(x_test)
df_trade['next_candle_color']  = y_test
df_trade['prediction'] = pred
df_trade['win'] = df_trade['next_candle_color'] == df_trade['prediction']

# filtrando probabilidades maiores que 53%
#df_trade = df_trade.merge(pd.DataFrame(ml.predict_proba(x_test)), left_index=True, right_index=True)
#df_trade = df_trade[((df_trade['prediction'] == 1)&(df_trade[1]>0.53))|(df_trade['prediction'] == 0)&(df_trade[0] > 0.53)]

print('Score: %f' % ml.score(x_train, y_train))
print('Real Rate: %f' % (df_trade[df_trade['win'] == True]['win'].count()/df_trade['win'].count()))