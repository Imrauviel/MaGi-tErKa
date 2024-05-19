import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from problems.problem import Problem


class ExampleProblemXGBoost(Problem):
    def __init__(self, problem_size=3):
        diabetes_data = pd.read_csv("example/diabetes.csv")
        label = np.array(diabetes_data["Outcome"])
        features = np.array(diabetes_data.iloc[:, :-1])
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(features, label, test_size=0.3,
                                                                                stratify=label, random_state=0)
        self.problem_size = problem_size

    def reward_function(self, param1, param2, param3=1, param4=1, param5=0, param6=0):
        parameter_range = {'max_depth': [3, 10],
                           'n_estimators': [100, 1000],
                           'learning_rate': [0.01, 0.3],
                           'colsample_bytree': [0.5, 1.0]}


        max_depth = int(
            param1 * (parameter_range['max_depth'][1] - parameter_range['max_depth'][0]) + parameter_range['max_depth'][
                0])
        n_estimators = int(param2 * (parameter_range['n_estimators'][1] - parameter_range['n_estimators'][0]) +
                           parameter_range['n_estimators'][0])

        learning_rate = param3 * (parameter_range['learning_rate'][1] - parameter_range['learning_rate'][0]) + \
                        parameter_range['learning_rate'][0]

        colsample_bytree = param4 * (parameter_range['colsample_bytree'][1] - parameter_range['colsample_bytree'][0]) + \
                           parameter_range['colsample_bytree'][0]

        classifier = XGBClassifier(learning_rate=learning_rate,
                                   max_depth=max_depth,
                                   n_estimators=n_estimators,
                                   colsample_bytree=colsample_bytree,
                                   use_label_encoder=False, eval_metric='logloss')

        classifier.fit(self.X_train, self.Y_train)
        pred = classifier.predict(self.X_test)
        acc = np.count_nonzero(pred == self.Y_test) / len(self.Y_test)
        return acc

    def get_bounds(self):
        if self.problem_size == 2:
            return {'param1': (0, 1),
                    'param2': (0, 1)}
        if self.problem_size == 3:
            return {'param1': (0, 1),
                    'param2': (0, 1),
                    'param3': (0, 1)}
        elif self.problem_size == 4:
            return {'param1': (0, 1),
                    'param2': (0, 1),
                    'param3': (0, 1),
                    'param4': (0, 1)}
        else:
            raise ValueError("Problem size not supported")
