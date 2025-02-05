import mlflow, os
from abc import ABC, abstractmethod

class MLFlowExp(ABC):
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)
        self.mlflow = mlflow
        # self.mlflow.autolog()
        print(f'-> Experiment : {self.experiment_name} started..')
        # print(f'-> Tracking folder : {self.mlflow.get_tracking_uri()}')
        print(f'-> Registry folder : {self.mlflow.get_registry_uri()}')

    @abstractmethod
    def run(self):
        pass

    def get_exp_class_folder_path(self):
        f_ = os.path.abspath(__file__)
        return os.path.dirname(f_)
    
    def get_root_folder_path(self):
        current_folder = self.get_exp_class_folder_path()
        return os.path.abspath(os.path.join(current_folder, '..', '..'))