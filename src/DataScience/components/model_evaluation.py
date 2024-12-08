import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn  
import joblib
from src.DataScience.config.configuration import ModelEvaluationConfig
from pathlib import Path
from src.DataScience.utils.common import save_json
import os



# os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/ammarvohra92/Data-Science-Project.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"] = "ammarvohra92"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "51d801c289d7a96534cfbed1f8a9f4f64bfbc6c4"



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_score = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)


            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            if tracking_url_type_score != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticNetModel")

            else:
                mlflow.sklearn.log_model(model, "model")






