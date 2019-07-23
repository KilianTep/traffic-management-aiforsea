import numpy as np
from sklearn import metrics
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


def prepare_model_inputs(transformed_df, demand_features, ts_features, target_features):
    '''
    This function does all the necessary reshaping to pass the features into the model
    :param transformed_df: training.csv transformed with ./src/preprocess_dataset.py
    :param demand_features:
    :param ts_features:
    :param target_features:
    :return: model inputs
    '''
    sample_weight = transformed_df.loc[transformed_df.day <= 47]['sample_weight'].values

    step_back = len(demand_features)
    ts_shape = len(ts_features)
    y_shape = len(target_features)

    X_demand_train = transformed_df.loc[transformed_df.day <= 47][demand_features].values
    X_demand_train = np.reshape([x for x in X_demand_train], (-1, step_back, 1))
    print('X_demand_train.shape', X_demand_train.shape)

    X_demand_test = transformed_df.loc[transformed_df.day > 47][demand_features].values
    X_demand_test = np.reshape([x for x in X_demand_test], (-1, step_back, 1))
    print('X_demand_test.shape', X_demand_test.shape)

    X_ts_train = transformed_df.loc[transformed_df.day <= 47][ts_features].values
    X_ts_train = np.reshape([x for x in X_ts_train], (-1, ts_shape))
    print('X_ts_train.shape', X_ts_train.shape)

    X_ts_test = transformed_df.loc[transformed_df.day > 47][ts_features].values
    X_ts_test = np.reshape([x for x in X_ts_test], (-1, ts_shape))
    print('X_ts_test.shape', X_ts_test.shape)

    Y_train = transformed_df.loc[transformed_df.day <= 47][target_features].values
    print('Y_train.shape', Y_train.shape)
    Y_test = transformed_df.loc[transformed_df.day > 47][target_features].values
    print('Y_test.shape', Y_test.shape)

    return X_demand_train, X_ts_train, X_demand_test, X_ts_test, \
            Y_train, Y_test, sample_weight, step_back, ts_shape, y_shape


def demand_lstm(step_back, ts_shape, y_shape):
    '''
    Architecture for LSTM model
    :param step_back:  number step back in time for demand
    :param ts_shape: shape of time time-space vector
    :param y_shape: shape of target vector
    :return: model
    '''
    demand_input = Input(shape=(step_back, 1))
    lstm_layer = LSTM(units=100, activation='tanh', return_sequences=True)(demand_input)
    dropout = Dropout(0.5)(lstm_layer)
    lstm_layer1 = LSTM(units=50, activation='tanh', return_sequences=True)(dropout)
    dropout_1 = Dropout(0.5)(lstm_layer1)
    lstm_layer2 = LSTM(units=25, activation='tanh', return_sequences=True)(dropout_1)
    dropout_2 = Dropout(0.2)(lstm_layer2)
    lstm_layer3 = LSTM(units=10, activation='tanh', return_sequences=True)(dropout_2)
    flatten_lstm3 = Flatten()(lstm_layer3)

    time_space_input = Input(shape=(ts_shape,))
    dense_ts = Dense(64)(time_space_input)

    merge_ts_lstm = concatenate([flatten_lstm3, dense_ts])
    dense_1 = Dense(75)(merge_ts_lstm)
    dense_2 = Dense(25)(dense_1)
    output_dense = Dense(y_shape, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0))(dense_2)

    model = Model(inputs=[demand_input, time_space_input], outputs=output_dense)
    model.compile(optimizer='adam', loss=mean_squared_error, metrics=[rmse])
    print(model.summary())
    return model


def evaluate_t_plus_1_performance(transformed_test_df, model, demand_features, ts_features, target_features):
    '''
    Function to evaluate performance at T+1
    :param transformed_test_df:
    :param model:
    :param demand_features:
    :param ts_features:
    :param target_features:
    :return: print performance scores (MSE, RMSE)
    '''
    step_back = len(demand_features)
    ts_shape = len(ts_features)

    X_demand = transformed_test_df[demand_features].values
    X_demand = np.reshape([x for x in X_demand], (-1, step_back, 1))

    X_ts = transformed_test_df[ts_features].values
    X_ts = np.reshape([x for x in X_ts], (-1, ts_shape))

    Y = transformed_test_df[target_features].values
    evaluation = model.evaluate([X_demand, X_ts], Y)
    mse_score = evaluation[0]
    rmse_score = evaluation[1]
    print('--------------------------------------------------')
    print('Performance at t+1 only:')
    print('MSE: {}'.format(mse_score))
    print('RMSE: {}'.format(rmse_score))
    print('--------------------------------------------------')


def evaluate_t_plus_5_performance(transform_test_df, model, ts_norm_to_scaled, ts_scaled_to_norm,
                                  demand_features, ts_features):
    '''
    Function to evalute at T+1,..,T+5 performance
    :param transform_test_df:
    :param model:
    :param ts_norm_to_scaled:
    :param ts_scaled_to_norm:
    :return: print performance scores (MSE, RMSE)
    '''

    step_back = len(demand_features)
    ts_shape = len(ts_features)
    d_truth = []
    d_preds = []
    print('Evaluating t+1..t+5 performance for {} predictions.'.format(transform_test_df.shape[0]))
    print('This may take some time if your test set is very big.')
    print('--------------------------------------------------')
    temp_rmse = None
    for idx, row in transform_test_df.iterrows():
        if (idx + 1) % 1000 == 0:
            print('{} / {} predictions evaluated'.format(idx + 1, transform_test_df.shape[0]))
            print('t+1, ..., t+5 RMSE so far: {}'.format(temp_rmse))
            print('--------------------------------------------------')
        elif (idx + 1) % transform_test_df.shape[0] == 0:
            print('{} / {} predictions evaluated'.format(idx + 1, transform_test_df.shape[0]))

        d_feat_vector = row[demand_features].values.reshape(1, step_back, 1)
        ts_vector = row[ts_features].values.reshape(1, ts_shape)
        lat_lon_scaled = ts_vector[0, :2]

        # keeping track of the non-scaled time values so we can map the next steps
        # to their scaled values
        time_vector = [ts_scaled_to_norm[ts] for ts in ts_vector[0, 2:]]
        d_truth.append(row[['d_t_plus_{}'.format(i) for i in range(1, 6)]].values)

        preds = []
        for step in range(1, 6):
            pred_next_step = model.predict([d_feat_vector, ts_vector])
            preds.append(pred_next_step.flatten()[0])

            time_shift = lambda t: t + step * 0.25 if (t + step * 0.25) <= 23.75 else t + step * 0.25 - 24

            # shifting the time idx of vector
            d_feat_vector = np.append(d_feat_vector[:, 1:, :], pred_next_step)
            d_feat_vector = d_feat_vector.reshape(1, step_back, 1)

            # shifting time vector
            time_vector_shift = list(map(time_shift, time_vector))
            time_shift_scaled = [ts_norm_to_scaled[shift] for shift in time_vector_shift]

            # updating time vector scaled
            ts_vector = np.append(lat_lon_scaled, time_shift_scaled)
            ts_vector = ts_vector.reshape(1, ts_shape)

        d_preds.append(preds)
        temp_rmse = np.sqrt(metrics.mean_squared_error(d_truth, d_preds))

    mse_score = metrics.mean_squared_error(d_truth, d_preds)
    rmse_score = np.sqrt(metrics.mean_squared_error(d_truth, d_preds))

    print('--------------------------------------------------')
    print('Performance at t+1 til t+5:')
    print('MSE: {}'.format(mse_score))
    print('RMSE: {}'.format(rmse_score))
    print('--------------------------------------------------')


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


def window_lstm(step_back, ts_shape):
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
    model.compile(
        optimizer='adam',
        loss=mean_squared_error,
        metrics=[rmse]
    )
    return model




# def window_lstm(step_back, ts_shape):
#     demand_predictions = []  # array that will contain all predictions
#     demand_input = Input(shape=(step_back, 1))
#     flatten_lstm_block_1 = lstm_block(demand_input)
#
#     # adding time space and input
#     time_space_input = Input(shape=(ts_shape,))
#     dense_ts = Dense(64, name='dense_ts')(time_space_input)
#
#     merge_ts_lstm = concatenate([flatten_lstm_block_1, dense_ts])
#     dense_block_1 = dense_block(merge_ts_lstm, step_back)
#
#     # generating d_t+1
#     d_t_plus_1 = Dense(1, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0), name='d_t_plus_1')(dense_block_1)
#     demand_predictions.append(d_t_plus_1)
#
#     # using dense_block_1 from d_t+1 prediction
#     flatten_lstm_block_2 = lstm_block(dense_block_1, step_back)
#     merge_ts_lstm_2 = concatenate([flatten_lstm_block_2, dense_ts])
#     dense_block_2 = dense_block(merge_ts_lstm_2, step_back)
#
#     # generating d_t+2
#     d_t_plus_2 = Dense(1, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0), name='d_t_plus_2')(dense_block_2)
#     demand_predictions.append(d_t_plus_2)
#
#     # using dense_block_2 from d_t+2 prediction
#     flatten_lstm_block_3 = lstm_block(dense_block_2, step_back)
#     merge_ts_lstm_3 = concatenate([flatten_lstm_block_3, dense_ts])
#     dense_block_3 = dense_block(merge_ts_lstm_3, step_back)
#
#     # generating d_t+3
#     d_t_plus_3 = Dense(1, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0), name='d_t_plus_3')(dense_block_3)
#     demand_predictions.append(d_t_plus_3)
#
#     # using dense_block_3 from d_t+3 prediction
#     flatten_lstm_block_4 = lstm_block(dense_block_3, step_back)
#     merge_ts_lstm_4 = concatenate([flatten_lstm_block_4, dense_ts])
#     dense_block_4 = dense_block(merge_ts_lstm_4, step_back)
#
#     # generating d_t+4
#     d_t_plus_4 = Dense(1, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0), name='d_t_plus_4')(dense_block_4)
#     demand_predictions.append(d_t_plus_4)
#
#     # using dense_block_4 from d_t+4 prediction
#     flatten_lstm_block_5 = lstm_block(dense_block_4, step_back)
#     merge_ts_lstm_5 = concatenate([flatten_lstm_block_5, dense_ts])
#     dense_block_5 = dense_block(merge_ts_lstm_5, step_back)
#
#     # generating d_t+5
#     d_t_plus_5 = Dense(1, kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0), name='d_t_plus_5')(dense_block_5)
#     demand_predictions.append(d_t_plus_5)
#
#     model = Model(inputs=[demand_input, time_space_input], outputs=demand_predictions)
#     model.compile(
#         optimizer='adam',
#         loss=mean_squared_error,
#         metrics=[rmse]
#     )
#     return model

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
