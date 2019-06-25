import numpy as np
from sklearn import metrics
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Flatten, Input, concatenate
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
        'ts_d_minus_5', 'ts_d_minus_4',
        'ts_d_minus_3', 'ts_d_minus_2',
        'ts_d_minus_1', 'timestamp_decimal'
    ],

    'target_features': ['d_t_plus_1']
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
    for idx, row in transform_test_df.iterrows():
        if (idx + 1) % 1000 == 0:
            print('{} / {} predictions evaluated'.format(idx + 1, transform_test_df.shape[0]))
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
            d_feat_vector = np.append(pred_next_step, d_feat_vector[:, :-1, :])
            d_feat_vector = d_feat_vector.reshape(1, step_back, 1)

            # shifting time vector
            time_vector_shift = list(map(time_shift, time_vector))
            time_shift_scaled = [ts_norm_to_scaled[shift] for shift in time_vector_shift]

            # updating time vector scaled
            ts_vector = np.append(lat_lon_scaled, time_shift_scaled)
            ts_vector = ts_vector.reshape(1, ts_shape)

        d_preds.append(preds)

    mse_score = metrics.mean_squared_error(d_truth, d_preds)
    rmse_score = np.sqrt(metrics.mean_squared_error(d_truth, d_preds))

    print('--------------------------------------------------')
    print('Performance at t+1 til t+5:')
    print('MSE: {}'.format(mse_score))
    print('RMSE: {}'.format(rmse_score))
    print('--------------------------------------------------')


