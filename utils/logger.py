import os
import pickle
from collections import defaultdict


class Logger:
    def __init__(self, what_instance, experiment_name):
        self.id = -1
        self.what_instance = what_instance
        self.experiment_name = experiment_name
        self.__log_data = defaultdict(list)

    def log(self, attribute, data):
        self.__log_data[attribute].append(data)

    def get_all_log(self, attribute):
        return self.__log_data[attribute]

    def get_latest_log(self, attribute):
        if len(self.__log_data[attribute]) == 0:
            return None

        return self.__log_data[attribute][-1]

    def display_all(self, attribute, customized_message=None):
        if len(self.__log_data[attribute]) == 0:
            return

        if customized_message is not None:
            print("- %s : " % customized_message, end="")
        print(self.__log_data[attribute])

    def display_latest(self, attribute, customized_message=None):
        if len(self.__log_data[attribute]) == 0:
            return

        if customized_message is not None:
            print("- %s : " % customized_message, end="")
        else:
            print("- %s : " % attribute, end="")
        print(self.__log_data[attribute][-1])

    def save(self):
        if not os.path.exists("results/" + self.experiment_name):
            os.makedirs("results/" + self.experiment_name)
        with open("results/%s/%s_%d_data.bin" % (self.experiment_name, self.what_instance, self.id), "wb") as f:
            pickle.dump(self.__log_data, f)

    def load(self):
        with open("results/%s/%s_%d_data.bin" % (self.experiment_name, self.what_instance, self.id), "rb") as f:
            self.__log_data = pickle.load(f)


class DeviceLogger(Logger):
    def __init__(self, client_id, experiment_name):
        super().__init__("device", experiment_name)
        self.id = client_id
