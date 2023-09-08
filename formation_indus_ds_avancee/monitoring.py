import os

import pandas as pd
from sqlalchemy import create_engine


def monitor_with_io(predictions_folder: str, db_con_str: str, monitoring_table_name: str) -> None:
    latest_predictions_path = os.path.join(predictions_folder, 'latest.csv')
    latest_predictions = pd.read_csv(latest_predictions_path,
                                     usecols=['predictions_time', 'predictions'],
                                     parse_dates=['predictions_time'],
                                     date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d-%H%M%S'))

    monitoring_df = monitor(latest_predictions)

    engine = create_engine(db_con_str)
    db_conn = engine.connect()
    monitoring_df.to_sql(name=monitoring_table_name, con=db_conn, if_exists='append', index=False)
    db_conn.close()


def monitor(latest_predictions: pd.DataFrame) -> pd.DataFrame:
    # Start filling function
    monitoring_df = latest_predictions.groupby("predictions_time").agg(mean_predictions=("predictions", "mean")).reset_index()
    # End filling function
    return monitoring_df



def monitor_training(evaluation_df: pd.DataFrame, db_con_str: str, training_table_name: str):
    engine = create_engine(db_con_str)
    db_conn = engine.connect()
    evaluation_df.to_sql(name=training_table_name, con=db_conn, if_exists='append', index=False)
    db_conn.close()

