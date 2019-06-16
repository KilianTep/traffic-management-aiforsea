import argparse
from src.util import load_and_process_training_set, get_time_lags


parser = argparse.ArgumentParser(description='Preprocess dataset and do feature engineering')
parser.add_argument('--csv_path', metavar='CSV_PATH', type=str, default='NOPATH',
                    help='CSV path of training or test set. Needs to have same schema as ./dataset/training.csv')
parser.add_argument('--output_path', metavar='OUTPUT_PATH', type=str, default='./dataset',
                    help='OUTPUT path to store preprocessed set into parquet')

if __name__ == '__main__':
    print('Preprocessing for CSV files having the same schema as training.csv.')
    args = parser.parse_args()
    csv_path = args.csv_path
    output_path = args.output_path

    if output_path[-1] == '/':
        output_path = output_path[:-1]

    file_name = csv_path.split('/')[-1]
    output_file = '{}/{}_transformed.snappy.parquet'.format(output_path, file_name)

    input_df = load_and_process_training_set(csv_path)
    input_df_with_lags = get_time_lags(input_df)
    input_df_with_lags.to_parquet(output_file, compression='snappy', index=False)
    print('Preprocessing complete. Look at your output folder.')



