import argparse
from src.model import prepare_test_inputs, features, window_lstm
import pandas as pd
from pprint import pprint

parser = argparse.ArgumentParser(description='Load trained model and evaluate')
parser.add_argument('--model_path', metavar='MODEL_PATH', type=str,
                    default='./models/trained_model_2019-07-25_best_check_point',
                    help='Path of trained model. Default value corresponds to previously trained model')
parser.add_argument('--transformed_test_df_path', metavar='TRANSFORMED_TEST_DF_PATH', type=str,
                    default='./dataset/test.csv_transformed.snappy.parquet',
                    help='Path of transformed test df. Test df initially needs to be passed in preprocess_dataset.py',
                    )


if __name__ == "__main__":
    args = parser.parse_args()

    demand_features = features['demand_features']
    ts_features = features['ts_features']
    target_features = features['target_features']

    step_back = len(demand_features)
    ts_shape = len(ts_features)
    y_shape = len(target_features)

    transformed_test_df = pd.read_parquet(args.transformed_test_df_path)
    model = window_lstm(step_back, ts_shape)
    model.load_weights(args.model_path)

    X_demand, X_ts, Y = prepare_test_inputs(transformed_test_df, demand_features, ts_features, target_features)
    evaluation = model.evaluate([X_demand, X_ts], Y)
    eval_dict = {'d_t_plus_{}'.format(t): rmse for t, rmse in zip(range(1, 6), evaluation[-5:])}
    pprint(eval_dict)