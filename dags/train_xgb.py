import os
import sys
from datetime import timedelta
import pandas as pd

from airflow.decorators import dag, task
from sqlalchemy_utils.types.enriched_datetime.pendulum_date import pendulum

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))  # So that airflow can find config files

from dags.config import TRAIN_DATA_PATH, MODEL_REGISTRY_FOLDER, DATA_FOLDER
from formation_indus_ds_avancee.feature_engineering import prepare_features_with_io
from formation_indus_ds_avancee.train_and_predict import train_model_with_io
from formation_indus_ds_avancee.monitoring import monitor_training


@dag(default_args={'owner': 'airflow'}, schedule=timedelta(weeks=4),
     start_date=pendulum.today('UTC').add(hours=-1))
def train_xgb_model():
    @task
    def prepare_features_task() -> str:
        feature_train_path: str = os.path.join(DATA_FOLDER, 'prepared_features_train.parquet')
        prepare_features_with_io(data_path=TRAIN_DATA_PATH, features_path=feature_train_path, training_mode=True)
        return feature_train_path

    @task
    def train_model_task(feature_path: str) -> None:
        eval_df = train_model_with_io(features_path=feature_path, model_registry_folder=MODEL_REGISTRY_FOLDER, xgb=True)
        return eval_df

    @task
    def monitor_training_task(eval_df: pd.DataFrame):
        monitor_training(eval_df,
                        training_table_name="evaluation_theo",
                        db_con_str='postgresql://postgres:postgres@postgres:5432/postgres')


    feature_path = prepare_features_task()
    eval_df = train_model_task(feature_path)
    monitor_training_task(eval_df)



train_model_dag = train_xgb_model()
