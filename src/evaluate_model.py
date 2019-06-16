from src.model import evaluate_t_plus_5_performance, evaluate_t_plus_1_performance
from src.model import ts_norm_to_scaled, ts_scaled_to_norm, features
import argparse
from src.model import demand_lstm
import pandas as pd


parser = argparse.ArgumentParser(description='Load trained model and evaluate')
parser.add_argument('--model_path', metavar='MODEL_PATH', type=str,
                    default='./models/best_lstm_model',
                    help='Path of trained model. Default value corresponds to previously trained model')
parser.add_argument('--transformed_test_df_path', metavar='TRANSFORMED_TEST_DF_PATH', type=str,
                    default='NOPATH',
                    help='Path of transformed test df. Test df initially needs to be passed in preprocess_dataset.py'
                    )

if __name__ == '__main__':
    print('Evaluating performance at t+1 only, followed by performance at t+1...t+5')
    demand_features = features['demand_features']
    ts_features = features['ts_features']
    target_features = features['target_features']

    step_back = len(demand_features)
    ts_shape = len(ts_features)
    y_shape = len(target_features)

    args = parser.parse_args()
    model_path = args.model_path
    transformed_test_df_path = args.transformed_test_df_path

    transformed_test_df = pd.read_parquet(transformed_test_df_path)

    model = demand_lstm(step_back, ts_shape, y_shape)
    model.load_weights(model_path)

    # evaluate performance at t + 1
    evaluate_t_plus_1_performance(transformed_test_df, model, demand_features, ts_features, target_features)

    # evalute performance at t + 5
    evaluate_t_plus_5_performance(transformed_test_df, model, ts_norm_to_scaled, ts_scaled_to_norm,
                                  demand_features, ts_features)
    print('Performance evaluation over.')
