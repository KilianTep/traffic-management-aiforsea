import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import geohash


def get_exp_sample_weight(demand):
    '''
    Since we have an imbalanced dataset, we will give more weights to the higher volume of the demand
    with this function
    :param demand:
    :return:
    '''
    return np.exp(np.round(demand*10, 1))


def load_and_process_training_set(file_path):
    '''
    load training set and apply some preprocessing
    :param file_path:
    :return:
    '''

    print('Loading and transforming dataset')
    scaler = MinMaxScaler()
    training_set = pd.read_csv(file_path)
    training_set['lat_lon'] = training_set['geohash6'].apply(lambda x: geohash.decode(x))
    training_set['lat'] = training_set['lat_lon'].apply(lambda x: x[0])
    training_set['lon'] = training_set['lat_lon'].apply(lambda x: x[1])
    training_set['hour'] = training_set['timestamp'].apply(lambda x: x.split(':')[0]).astype('int')
    training_set['timestamp_hour'] = training_set.apply(get_timestamp_hour, axis=1)
    training_set['lat_scaled'] = scaler.fit_transform(training_set['lat'].values.reshape(-1, 1))
    training_set['lon_scaled'] = scaler.fit_transform(training_set['lon'].values.reshape(-1, 1))

    training_set['d_t'] = training_set['demand']
    training_set['sample_weight'] = training_set['demand'].apply(get_exp_sample_weight)
    training_set['timestamp_decimal'] = training_set['timestamp'].apply(get_timestamp_decimal)
    training_set['timestamp_decimal_scaled'] = scaler.fit_transform(
        training_set['timestamp_decimal'].values.reshape(-1, 1))
    training_set = training_set.sort_values(by=['geohash6', 'timestamp_hour']).reset_index(drop=True)
    training_set = training_set.drop(columns='lat_lon')

    print('Loading and transformation complete')
    return training_set


def get_timestamp_hour(row):
    '''
    Function to be used with pd.Series.apply during preprocessing
    :param row:
    :return: day and hour converted to decimal
    '''
    timestamp_to_convert = row.timestamp.split(':')
    hour = float(timestamp_to_convert[0])
    minutes = float(timestamp_to_convert[1]) / 60
    return row.day * 24 + hour + minutes


def get_timestamp_decimal(timestamp):
    '''
    Function to be used with pd.Series.apply during preprocessing
    :param timestamp:
    :return: timestamp converted to decimal
    '''
    timestamp_to_convert = timestamp.split(':')
    hour = float(timestamp_to_convert[0])
    minutes = float(timestamp_to_convert[1])
    return hour + minutes / 60


def create_ts_decimal_lag(ts, lag):
    '''
    Function to be used with pd.Series.apply during preprocessing
    :param timestamp decimal created above
    :return: decimal timestamp to create the relevant demand
    '''

    if (ts - lag * 0.25) < 0:
        return ts - lag * 0.25 + 24.0
    elif (ts - lag * 0.25) > 23.75:
        return ts - lag * 0.25 - 24.0
    else:
        return ts - lag * 0.25


def replace_mistmatching_demand(row, lag):
    '''
    Function to be used with pd.Series.apply during preprocessing
    Since we have missing timestamps for particular hours, we assume that if the previous
    demand does not have the corresponding timestamp, we replace it with 0
    :param row:
    :param lag:
    :return: fixed mismatching demand
    '''
    if lag > 0:
        if pd.isnull(row['d_t_plus_{}'.format(lag)]):
            return np.nan
        elif row['tdelta_plus_{}'.format(lag)] != 0.25 * lag:
            return 0
        else:
            return row['d_t_plus_{}'.format(lag)]
    else:
        if pd.isnull(row['d_t_minus_{}'.format(-lag)]):
            return np.nan
        elif row['tdelta_minus_{}'.format(-lag)] != 0.25 * lag:
            return 0
        else:
            return row['d_t_minus_{}'.format(-lag)]


def get_time_lags(df):
    '''
    We need to create time_lags for each different geohash so we can build a training set
    :param df:
    :return: transformed_df with relevant time lags
    '''
    print('Getting time lags for our dataset')
    unique_geohash = df['geohash6'].unique()
    temp = []
    for idx, gh in enumerate(unique_geohash):
        if (idx + 1) % 50 == 0:
            print('{}/{} geohash processed'.format(idx + 1, len(unique_geohash)))
        elif (idx + 1) % len(unique_geohash) == 0:
            print('{}/{} geohash processed'.format(idx + 1, len(unique_geohash)))

        rel_gh = df.loc[df.geohash6 == gh].copy()
        for t in range(1, 6):
            rel_gh['ts_plus_{}'.format(t)] = rel_gh['timestamp_hour'].shift(-t)
            rel_gh['tdelta_plus_{}'.format(t)] = rel_gh['ts_plus_{}'.format(t)] - rel_gh['timestamp_hour']
            rel_gh['d_t_plus_{}'.format(t)] = rel_gh['demand'].shift(-t)
            rel_gh['d_t_plus_{}'.format(t)] = rel_gh.apply(lambda x: replace_mistmatching_demand(x, t), axis=1)

            rel_gh['ts_minus_{}'.format(t)] = rel_gh['timestamp_hour'].shift(t)
            rel_gh['tdelta_minus_{}'.format(t)] = rel_gh['ts_minus_{}'.format(t)] - rel_gh['timestamp_hour']
            rel_gh['d_t_minus_{}'.format(t)] = rel_gh['demand'].shift(t)
            rel_gh['d_t_minus_{}'.format(t)] = rel_gh.apply(lambda x: replace_mistmatching_demand(x, -t), axis=1)

        temp.append(rel_gh)

    train_df = pd.concat(temp)

    scaler = MinMaxScaler()
    for lag in range(1, 6):
        train_df['ts_d_minus_{}'.format(lag)] = train_df['timestamp_decimal'] \
            .apply(lambda x: create_ts_decimal_lag(x, lag))
        train_df['ts_d_minus_{}_scaled'.format(lag)] = scaler \
            .fit_transform(train_df['ts_d_minus_{}'.format(lag)].values.reshape(-1, 1))
    train_df = train_df[sorted(train_df.columns)].dropna().reset_index(drop=True)
    print('Finished getting time lags. Find parquet file into output folder')
    return train_df


