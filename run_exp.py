import argparse
import numpy as np
from random import choice
from models.nn import *
from models.devices import ProposedDevice
from utils.logger import DeviceLogger
from utils.data_preprocess import get_dataset, get_loaders

IMPLEMENTED_STRATEGY = ["proposed"]
UNRELATED_ARGS = ["data_path", "verbose", "xpu_device", "dry_run"]
BOOLEAN = ["True", "False"]


class FederatedJob:
    def __init__(self, job_args):
        self.job_args = job_args

        # Init data & data loaders(path is fixed)
        train_data, _ = get_dataset(job_args.data_path, job_args.dataset)
        train_loaders, test_loaders = get_loaders(train_data, job_args.num_devices,
                                                  distribution=job_args.distribution,
                                                  distribution_parameter=eval(job_args.distribution_params),
                                                  batch_size=job_args.train_batch_size)

        # Init model & training auxiliary
        model = eval(job_args.model)
        if job_args.model_params:
            model_params = eval(job_args.model_params)
        else:
            model_params = tuple()

        # Init XPU device
        xpu_device = torch.device(job_args.xpu_device)

        # Init loggers
        self.device_loggers = [DeviceLogger(i, job_args.experiment_name) for i in range(job_args.num_devices)]
        # Init devices
        if job_args.strategy == "proposed":
            self.devices = [ProposedDevice(i, model, model_params, job_args.lr, job_args.lr_decay,
                                           job_args.inference_threshold, job_args.mu, job_args.num_communicate_clients,
                                           xpu_device, job_args.local_epoch, train_loaders[i], test_loaders[i],
                                           self.device_loggers[i], job_args.verbose)
                            for i in range(job_args.num_devices)]

    def conduct_training(self, is_dry_run=False):
        for current_round in range(self.job_args.num_rounds):
            print("# Conducting training round %d" % current_round)
            if self.job_args.sync == "True":
                for device in self.devices:
                    device.communicate(list(set(self.devices) - {device}))
                for device in self.devices:
                    device.train()
            else:
                device_selected = list()
                for _ in range(len(self.devices)):
                    device = choice(self.devices)
                    while device in device_selected:
                        device = choice(self.devices)
                    device_selected.append(device)
                    device.communicate(list(set(self.devices) - {device}))
                for device in device_selected:
                    device.train()
            round_mean_accuracy = np.mean([0 if device_logger.get_latest_log("local_acc") is None
                                           else device_logger.get_latest_log("local_acc")
                                           for device_logger in self.device_loggers])
            round_mean_parameter_transferred = np.mean([0 if device_logger.get_latest_log("comm") is None
                                                        else device_logger.get_latest_log("comm")
                                                        for device_logger in self.device_loggers])
            print(f"Mean accuracy this round is: \033[31m{round_mean_accuracy: .2f}%\033[0m. "
                  f"Mean parameter transferred is: {round_mean_parameter_transferred}.")

        if not is_dry_run:
            for i in range(self.job_args.num_devices):
                self.device_loggers[i].save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Default federated settings
    parser.add_argument("--strategy", default="proposed", type=str, choices=IMPLEMENTED_STRATEGY)
    parser.add_argument("--sync", default="True", type=str, choices=BOOLEAN)
    parser.add_argument("--num_rounds", default=50, type=int)
    parser.add_argument("--num_devices", default=30, type=int)
    parser.add_argument("--local_epoch", default=5, type=int)
    parser.add_argument("--train_batch_size", default=10, type=int)
    parser.add_argument("--model", default="SimpleCNN", type=str)
    parser.add_argument("--model_params", type=str)
    parser.add_argument("--lr", default=5e-2, type=float)
    parser.add_argument("--lr_decay", default=.95, type=float)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--distribution", default="dirichlet", type=str)
    parser.add_argument("--distribution_params", default="(5,)", type=str)

    # For part of the methods
    parser.add_argument("--num_shared_layers", type=int)
    parser.add_argument("--num_communicate_clients", type=int)
    parser.add_argument("--inference_threshold", type=float)

    # Hardware related
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--xpu_device", default="cuda:0", type=str)
    parser.add_argument("--dry_run", default="False", type=str, choices=BOOLEAN)

    # Experiment related
    parser.add_argument("--force_rename", type=str)
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--verbose", default=2, type=int)

    args = parser.parse_args()

    experiment_name = str()
    for option, value in vars(args).items():
        if value is None: continue
        if option in UNRELATED_ARGS: continue
        experiment_name += str(value) + "_"

    print("============== Federated Learning Setting ==============")
    print(args)
    print("============== Federated Learning Setting ==============")

    for run_id in range(1, args.repeat + 1):
        if args.force_rename is None:
            args.experiment_name = experiment_name + str(run_id)
        else:
            args.experiment_name = args.force_rename + str(run_id)
        print("Conducting experiment %s" % args.experiment_name)
        job = FederatedJob(args)
        if args.dry_run == "True":
            job.conduct_training(True)
        else:
            job.conduct_training()
        torch.cuda.empty_cache()
