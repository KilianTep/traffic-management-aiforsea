import argparse
import datetime
import pandas as pd
from src.model import prepare_model_inputs, demand_lstm, features
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

DATE_STR = datetime.date.today()

parser = argparse.ArgumentParser(description='Train Demand LSTM Model')
parser.add_argument('--transformed_train_path', metavar='TRANSFORMED_TRAIN_PATH', type=str,
                    default='./dataset/training.csv_transformed.snappy.parquet',
                    help='Path of parquet transformed training set. Default value is transformed training ')
parser.add_argument('--output_model_path', metavar='OUTPUT_MODEL_PATH', type=str, default='./models',
                    help='Output path for LSTM mode. Default is ./models')
parser.add_argument('--epochs', metavar='EPOCHS', type=int, default=30,
                    help='Number of epochs for the model to train. Default is 30')
parser.add_argument('--batch_size', metavar='BATCH_SIZE', type=int, default=128,
                    help='Batch size for training. Default is 128.')
parser.add_argument('--log_path', metavar='LOG_PATH', type=str, default='./src/logs',
                    help='Path to store logs during training. Default is ./src/logs directory')

if __name__ == '__main__':
    ts_features = features['ts_features']
    demand_features = features['demand_features']
    target_features = features['target_features']

    args = parser.parse_args()
    transformed_train_path = args.transformed_train_path
    output_model_path = args.output_model_path
    epochs = args.epochs
    log_path = args.log_path
    batch_size = args.batch_size

    if output_model_path[-1] == '/':
        output_model_path = output_model_path[:-1]

    if log_path[-1] == '/':
        log_path = log_path[:-1]

    model_filepath = '{}/trained_model_{}'.format(output_model_path, DATE_STR)
    log_filepath = '{}/model_{}.log'.format(log_path, DATE_STR)

    transformed_df = pd.read_parquet(transformed_train_path)

    X_demand_train, X_ts_train, X_demand_test, X_ts_test, Y_train, Y_test, sample_weight, step_back, \
        ts_shape, y_shape = prepare_model_inputs(transformed_df, demand_features, ts_features, target_features)

    checkpointer = ModelCheckpoint(
        filepath='{}_best_check_point'.format(model_filepath),
        verbose=1,
        monitor='val_loss',
        save_best_only=True)

    csv_logger = CSVLogger(
        filename=log_filepath,
        separator=',',
        append=True)

    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.2,
        patience=2,
        min_lr=0.000008)

    model = demand_lstm(step_back, ts_shape, y_shape)
    model.fit(
        x=[X_demand_train, X_ts_train],
        y=Y_train,
        validation_data=([X_demand_test, X_ts_test], Y_test),
        sample_weight=sample_weight,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpointer, csv_logger, reduce_lr]
    )

    print('Saving model at {}...'.format(model_filepath))
    model.save(model_filepath)
    print('Model saved.')

