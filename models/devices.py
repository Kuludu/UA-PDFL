import copy
import torch
from collections import defaultdict, OrderedDict
from random import sample, choice
from utils.evaluate import acc_count
from utils.metrics import js_divergence


class BaseDevice:
    def __init__(self, device_id, model, model_params, lr, lr_decay, xpu_device, local_epoch, train_loader, test_loader,
                 device_logger, verbose):
        self.device_id = device_id
        self.model = model(*model_params).to(xpu_device)
        self.local_epoch = local_epoch
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=.5)
        self.lr_decay = lr_decay
        self.xpu_device = xpu_device
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_train_instances = len(train_loader.dataset)
        self.device_logger = device_logger
        self.remote_device_infos = list()
        self.verbose = verbose
        if verbose > 1:
            print(f"Client #{device_id}, train: {len(train_loader.dataset)} / test: {len(test_loader.dataset)}.")

    def communicate(self, remote_devices):
        raise NotImplementedError

    def train(self):
        # Compute averaged weights
        averaged_weights = self.compute_averaged_weights(self.remote_device_infos)
        # Apply weights
        self.model.set_weights(averaged_weights)
        # Train
        self.local_train()

        if self.verbose > 1:
            print("Parameter transferred: %d" % sum(self.device_logger.get_all_log("comm")))

    def local_train(self):
        num_correct_instances, num_tested_instances = acc_count(self.model, self.test_loader, self.xpu_device)

        if self.verbose:
            print("########## Training client #%d ##########" % self.device_id)
        self.model.train()
        for _ in range(self.local_epoch):
            for data, target in self.train_loader:
                data, target = data.to(self.xpu_device), target.to(self.xpu_device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                target = target.flatten()
                loss = self.loss(outputs, target)
                loss.backward()
                self.optimizer.step()
        self.optimizer.param_groups[0]["lr"] *= self.lr_decay

        self.device_logger.log("correct", num_correct_instances)
        self.device_logger.log("tested", num_tested_instances)
        self.device_logger.log("local_acc", 100.0 * num_correct_instances / num_tested_instances)

        if self.verbose:
            self.device_logger.display_latest("local_acc")

    def compute_averaged_weights(self, remote_device_infos):
        averaged_weights = dict()
        local_weights = self.model.get_weights()
        num_remote_instances = [remote_device_info["num_instances"] for remote_device_info in remote_device_infos]
        self.device_logger.log("comm", sum(self.model.layer_size) * len(remote_device_infos))
        num_total_instances = self.num_train_instances + sum(num_remote_instances)
        for key in local_weights.keys():
            if "num_batches_tracked" in key:
                continue

            averaged_weights[key] = torch.zeros_like(local_weights[key]).to(self.xpu_device)
            for remote_device_info in remote_device_infos:
                weights = remote_device_info["weights"]
                num_remote_instances = remote_device_info["num_instances"]
                averaged_weights[key] += weights[key] * num_remote_instances / num_total_instances
            averaged_weights[key] += local_weights[key] * self.num_train_instances / num_total_instances

        return averaged_weights

    def get_device_info(self):
        return {
            "weights": self.model.get_weights(),
            "num_instances": self.num_train_instances
        }


class ProposedDevice(BaseDevice):
    def __init__(self, device_id, model, model_params, lr, lr_decay, inference_threshold, mu, num_communicate_clients,
                 xpu_device, local_epoch, train_loader, test_loader, device_logger, verbose):
        super().__init__(device_id, model, model_params, lr, lr_decay, xpu_device, local_epoch, train_loader,
                         test_loader, device_logger, verbose)
        self.standard_input = torch.ones((1, model_params[1], model_params[2], model_params[2])).to(self.xpu_device)
        self.num_model_layers = len(self.model.layer_size)
        self.num_feature_extractor_layers = self.model.num_feature_extractor_layers
        self.inference_threshold = inference_threshold
        self.mu = mu
        self.device_representations = self.compute_representations()
        self.num_communicate_clients = num_communicate_clients
        self.model.eval()
        with torch.no_grad():
            self.averaged_common_feature = torch.zeros_like(self.model.head(self.standard_input), requires_grad=False)

    def communicate(self, remote_devices):
        remote_device_this_round = sample(remote_devices, k=min(self.num_communicate_clients, len(remote_devices)))
        self.remote_device_infos = [device.get_device_info() for device in remote_device_this_round]

    def train(self):
        averaged_weights = self.compute_averaged_weights(self.remote_device_infos)
        self.model.set_weights(averaged_weights)

        self.model.train()
        self.local_train()

        # Update device representations each communication round
        self.device_representations = self.compute_representations()

        if self.verbose > 1:
            print("Parameter transferred: %d" % sum(self.device_logger.get_all_log("comm")))

    def local_train(self):
        num_correct_instances, num_tested_instances = acc_count(self.model, self.test_loader, self.xpu_device)

        if self.verbose:
            print("########## Training client #%d ##########" % self.device_id)
        self.model.train()
        for _ in range(self.local_epoch):
            for data, target in self.train_loader:
                data, target = data.to(self.xpu_device), target.to(self.xpu_device)
                target = target.flatten()
                self.optimizer.zero_grad()
                middle_representation = self.model.head(data)
                regularize_loss = torch.mean((middle_representation - self.averaged_common_feature) ** 2)
                outputs = self.model.base(middle_representation)
                loss = self.loss(outputs, target) + regularize_loss * self.mu
                loss.backward()
                self.optimizer.step()
        self.optimizer.param_groups[0]["lr"] *= self.lr_decay

        self.device_logger.log("correct", num_correct_instances)
        self.device_logger.log("tested", num_tested_instances)
        self.device_logger.log("local_acc", 100.0 * num_correct_instances / num_tested_instances)

        if self.verbose:
            self.device_logger.display_latest("local_acc")

    def compute_representations(self):
        self.model.eval()
        with torch.no_grad():
            representations = self.model(self.standard_input)

        return representations

    def compute_averaged_weights(self, remote_device_infos):
        averaged_weights = dict()
        local_weights = copy.deepcopy(self.model.get_weights())
        num_local_instance = self.num_train_instances
        remote_weights = [cached_remote_device_info["weights"]
                          for cached_remote_device_info in remote_device_infos]
        num_remote_instances = [cached_remote_device_info["num_instances"]
                                for cached_remote_device_info in remote_device_infos]
        remote_representations = [cached_remote_device_info["representations"]
                                  for cached_remote_device_info in remote_device_infos]
        remote_divergences = [js_divergence(self.device_representations, remote_representation)
                              for remote_representation in remote_representations]
        num_shared_layers = [self.num_feature_extractor_layers
                             if divergence > self.inference_threshold else
                             self.num_model_layers
                             for divergence in remote_divergences]
        if self.verbose > 1:
            print(remote_divergences)
            print(num_shared_layers)
        num_total_instance_layer_wise = defaultdict(int)
        num_total_remote_instances = sum(num_remote_instances)

        # Compute common feature logit
        self.model.eval()
        self.averaged_common_feature[:] = 0
        for weight, num_instance in zip(remote_weights, num_remote_instances):
            self.model.set_weights(weight)
            with torch.no_grad():
                self.averaged_common_feature += (self.model.head(self.standard_input) *
                                                 num_instance / num_total_remote_instances)

        if (sum([remote_divergence < self.inference_threshold for remote_divergence in remote_divergences])
                == self.num_communicate_clients):
            averaged_weights = choice(remote_weights)
        else:
            # Preprocess layer-wise instance accumulation
            layer_keys = list(local_weights.keys())
            for key in layer_keys:
                num_total_instance_layer_wise[key] += num_local_instance
            for remote_id, num_shared_layer in enumerate(num_shared_layers):
                for layer_id in range(num_shared_layer):
                    num_total_instance_layer_wise[layer_keys[layer_id]] += num_remote_instances[remote_id]

            # Init averaged parameters
            for key in layer_keys:
                averaged_weights[key] = torch.zeros_like(local_weights[key], device=self.xpu_device)

            # Compute averaged parameters remote-wise
            for remote_id, remote_params in enumerate(remote_weights):
                shared_remote_params = layer_keys[0:num_shared_layers[remote_id]]
                self.device_logger.log("comm", sum(self.model.layer_size[0:num_shared_layers[remote_id]]))
                for key in shared_remote_params:
                    if "num_batches_tracked" in key:
                        continue
                    averaged_weights[key] += (remote_params[key] * num_remote_instances[remote_id]
                                              / num_total_instance_layer_wise[key])

            # Compute averaged parameters local-wise
            for key in layer_keys:
                if "num_batches_tracked" in key:
                    continue
                averaged_weights[key] += local_weights[key] * num_local_instance / num_total_instance_layer_wise[key]

        return averaged_weights

    def get_device_info(self):
        return {
            "id": self.device_id,
            "weights": self.model.get_weights(),
            "num_instances": self.num_train_instances,
            "representations": self.device_representations
        }
