"""This file contains classes for managing metrics such as loss."""
from typing import Literal


class Metric():
    """Contains information about one metric in one epoch.

    Simply speaking, Metric serves as a compressed representation of a list
    containing achieved metric at each batch of an epoch. New value about the
    achieved metric is added constantly throughout the training and validation.
    Metric methods can be used to get statistics about these values. Metric
    should not be created separately, but only in the add_metric method of
    MetricsCollection.
    """

    def __init__(self) -> None:
        """Create an empty metric.

        The empty metric does not contain any values, thus len and sum are 0.
        Since nothing was added yet, last is None.
        """
        self.sum = 0
        self.len = 0
        self.last = None

    def add(self, value: float) -> None:
        """Add new value to a metric.

        Parameters:
        value (float):
            value to be added
        """
        self.sum += value
        self.len += 1
        self.last = value

    def get_average(self) -> float:
        """Get average of all added values.

        Returns:
        average of all added values
        """
        if self.len == 0:
            return 0
        return self.sum / self.len

    def get_sum(self) -> float:
        """Get sum of all added values.

        Returns:
        sum of all added values
        """
        return self.sum

    def get_last(self) -> float:
        """Get last added value.

        Returns:
        last added value
        """
        return self.last


class MetricsCollection():
    """Serves as a dictionary containing all metrics used during an epoch.

    Both training and validation should have their own MetricsCollection object
    that is created separately for each epoch.
    """

    def __init__(self, prefix: Literal["train", "val"]) -> None:
        """Create an empty MetricsCollection.

        Each MetricsCollection contains a prefix (either "train" or "val") that
        is used when creating a wandb dictionary.

        Parameters:
        prefix (Literal["train", "val"]):
            specifies whether metrics regards training or validation
        """
        self.content = dict()
        self.prefix = prefix

    def add_metric(self, metric_name: str, metric: Metric = None) -> None:
        """Add a new metric to collection.

        Parameters:
        metric_name (str):
            name of the metric to be added
        metric (Metric, optional):
            metric to be added, if not specified a new one will be created
        """
        if metric:
            self.content[metric_name] = metric
        else:
            self.content[metric_name] = Metric()

    def __getitem__(self, metric_name: str) -> Metric:
        """Get metric directly from the dictionary.

        Parameters:
        metric_name (str):
            name of the metric to be obtained from the dictionary

        Returns:
        metric according to the provided name
        """
        return self.content[metric_name]

    def get_dict(self, mode: Literal["print", "wandb"]) -> dict:
        """Obtain a dict either for print (using tqdm) or for wandb.

        Created dict containg average values by default. Values in dict will be
        floats in wandb and strings in print.

        Parameters:
        mode (Literal["print", "wandb"]):
            specifies whether dict is for print or for wandb
        """
        if mode == "wandb":
            return {
                self.prefix + "_avg_" + metric:
                self.content[metric].get_average()
                for metric in self.content
                }
        elif mode == "print":
            return {
                self.prefix + "_avg_" + metric:
                f"{self.content[metric].get_average():<6.5f}"
                for metric in self.content
                }
        else:
            raise ValueError
