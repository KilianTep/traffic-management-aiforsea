import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Flatten, Input, concatenate, Reshape
from keras.constraints import MinMaxNorm
from keras.losses import mean_squared_error
import keras.backend as K


# dictionary to keep track of features used from transformed training set
features = {
    'demand_features': [
        'd_t_minus_5', 'd_t_minus_4',
        'd_t_minus_3', 'd_t_minus_2',
        'd_t_minus_1', 'd_t'
    ],

    'ts_features': [
        'lat_scaled', 'lon_scaled',
        'ts_d_minus_5_scaled', 'ts_d_minus_4_scaled',
        'ts_d_minus_3_scaled', 'ts_d_minus_2_scaled',
        'ts_d_minus_1_scaled', 'timestamp_decimal_scaled'
    ],

    'target_features': ['d_t_plus_1', 'd_t_plus_2', 'd_t_plus_3', 'd_t_plus_4', 'd_t_plus_5']
}


def rmse(y_true, y_pred):

    '''
    Keras custom metric to monitor RMSE during training
    :param y_true:
    :param y_pred:
    :return: rmse score
    '''

    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def lstm_block(dense_input):
    layer_dim = dense_input.shape[1].value
    dense_input = Reshape((layer_dim, 1))(dense_input)
    lstm_layer1 = LSTM(units=100, activation='tanh', return_sequences=True)(dense_input)
    dropout_1 = Dropout(0.5)(lstm_layer1)
    lstm_layer2 = LSTM(units=50, activation='tanh', return_sequences=True)(dropout_1)
    dropout_2 = Dropout(0.5)(lstm_layer2)
    lstm_layer3 = LSTM(units=25, activation='tanh', return_sequences=True)(dropout_2)
    dropout_3 = Dropout(0.2)(lstm_layer3)
    lstm_layer4 = LSTM(units=10, activation='tanh', return_sequences=True)(dropout_3)
    flatten_lstm3 = Flatten()(lstm_layer4)
    return flatten_lstm3


def dense_block(flattened_lstm):
    dense_1 = Dense(75, activation='relu')(flattened_lstm)
    dense_2 = Dense(25, activation='relu')(dense_1)
    dense_3 = Dense(10, activation='relu')(dense_2)
    dense_4 = Dense(5, activation='relu')(dense_3)
    return dense_4


def append_demand_input(demand_input, new_pred):
    new_pred = Reshape((1, 1))(new_pred)
    new_demand_input = concatenate([demand_input, new_pred], axis=1)
    return new_demand_input


def window_lstm(step_back, ts_shape, lr=0.001):
    demand_predictions = []  # array that will contain all predictions
    demand_input = Input(shape=(step_back, 1))
    flatten_lstm_block_1 = lstm_block(demand_input)

    # adding time space and input
    time_space_input = Input(shape=(ts_shape,))
    dense_ts = Dense(64, name='dense_ts')(time_space_input)

    merge_ts_lstm = concatenate([flatten_lstm_block_1, dense_ts])
    dense_block_1 = dense_block(merge_ts_lstm)

    # generating d_t+1
    d_t_plus_1 = Dense(1, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0), name='d_t_plus_1')(dense_block_1)
    demand_predictions.append(d_t_plus_1)

    demand_input_2 = append_demand_input(demand_input, d_t_plus_1)
    flatten_lstm_block_2 = lstm_block(demand_input_2)
    merge_ts_lstm_2 = concatenate([flatten_lstm_block_2, dense_ts])
    dense_block_2 = dense_block(merge_ts_lstm_2)

    # generating d_t+2
    d_t_plus_2 = Dense(1, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0), name='d_t_plus_2')(dense_block_2)
    demand_predictions.append(d_t_plus_2)

    # using d_t+2 prediction
    demand_input_3 = append_demand_input(demand_input_2, d_t_plus_2)
    flatten_lstm_block_3 = lstm_block(demand_input_3)
    merge_ts_lstm_3 = concatenate([flatten_lstm_block_3, dense_ts])
    dense_block_3 = dense_block(merge_ts_lstm_3)

    # generating d_t+3
    d_t_plus_3 = Dense(1, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0), name='d_t_plus_3')(dense_block_3)
    demand_predictions.append(d_t_plus_3)

    # using d_t+3 prediction
    demand_input_4 = append_demand_input(demand_input_3, d_t_plus_3)
    flatten_lstm_block_4 = lstm_block(demand_input_4)
    merge_ts_lstm_4 = concatenate([flatten_lstm_block_4, dense_ts])
    dense_block_4 = dense_block(merge_ts_lstm_4)

    # generating d_t+4
    d_t_plus_4 = Dense(1, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0), name='d_t_plus_4')(dense_block_4)
    demand_predictions.append(d_t_plus_4)

    # using d_t+4 prediction
    demand_input_5 = append_demand_input(demand_input_4, d_t_plus_4)
    flatten_lstm_block_5 = lstm_block(demand_input_5)
    merge_ts_lstm_5 = concatenate([flatten_lstm_block_5, dense_ts])
    dense_block_5 = dense_block(merge_ts_lstm_5)

    # generating d_t+5
    d_t_plus_5 = Dense(1, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0), name='d_t_plus_5')(dense_block_5)
    demand_predictions.append(d_t_plus_5)

    model = Model(inputs=[demand_input, time_space_input], outputs=demand_predictions)
    adam = Adam(lr=lr)
    model.compile(
        optimizer=adam,
        loss=mean_squared_error,
        metrics=[rmse]
    )
    return model


def prepare_window_model_inputs(train_df, demand_features, ts_features, target_features):
    sample_weight = train_df.loc[train_df.day <= 47]['sample_weight'].values
    weight_dict = {}
    for i in range(1, 6):
        weight_dict['d_t_plus_{}'.format(i)] = sample_weight

    step_back = len(demand_features)

    X_demand_train = train_df.loc[train_df.day <= 47][demand_features].values
    X_demand_train = np.reshape([x for x in X_demand_train], (-1, step_back, 1))
    print('X_demand_train.shape', X_demand_train.shape)

    X_demand_val = train_df.loc[(train_df.day > 47) & (train_df.demand >= 0.5)][demand_features].values
    X_demand_val = np.reshape([x for x in X_demand_val], (-1, step_back, 1))
    print('X_demand_val.shape', X_demand_val.shape)

    ts_shape = len(ts_features)
    X_ts_train = train_df.loc[train_df.day <= 47][ts_features].values
    X_ts_train = np.reshape([x for x in X_ts_train], (-1, ts_shape))
    print('X_ts_train.shape', X_ts_train.shape)

    X_ts_val = train_df.loc[(train_df.day > 47) & (train_df.demand >= 0.5)][ts_features].values
    X_ts_val = np.reshape([x for x in X_ts_val], (-1, ts_shape))
    print('X_ts_val.shape', X_ts_val.shape)

    Y_train = []
    Y_val = []
    for tf in target_features:
        Y_train.append(train_df.loc[train_df.day <= 47][tf].values)
        Y_val.append(train_df.loc[(train_df.day > 47) & (train_df.demand >= 0.5)][tf].values)

    print('Y_train.shape', Y_train[0])
    print('Y_val.shape', Y_val[0])

    return X_demand_train, X_ts_train, X_demand_val, X_ts_val, \
        Y_train, Y_val, sample_weight, step_back, ts_shape, weight_dict
