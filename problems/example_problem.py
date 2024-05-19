import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

from problems.problem import Problem


class ExampleProblem(Problem):
    def __init__(self):
        diabetes_data = pd.read_csv("example/diabetes.csv")
        label = np.array(diabetes_data["Outcome"])
        features = np.array(diabetes_data.iloc[:, :-1])
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(features, label, test_size=0.3,
                                                                                stratify=label, random_state=0)

    def reward_function(self, param1, param2, param3=0, param4=0, param5=0, param6=0):
        parameter_range = [[1e-4, 1.0], [1e-4, 1.0]]

        C = param1 * (parameter_range[0][1] - parameter_range[0][0]) + parameter_range[0][0]
        gam = param2 * (parameter_range[1][1] - parameter_range[1][0]) + parameter_range[1][0]
        classifier = svm.SVC(kernel="rbf", C=C, gamma=gam, probability=True)
        classifier.fit(self.X_train, self.Y_train)
        pred = classifier.predict(self.X_test)
        acc = np.count_nonzero(pred == self.Y_test) / len(self.Y_test)
        return acc

    def get_bounds(self):
        return {'param1': (0, 1),
                'param2': (0, 1)}
