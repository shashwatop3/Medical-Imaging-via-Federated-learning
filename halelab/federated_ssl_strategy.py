import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import EvaluateRes, FitIns, Metrics, NDArrays, Parameters, Scalar
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
import copy
import pandas as pd
import logging

from .ssl_models import ContrastiveSSLModel, RotationSSLModel, JigsawSSLModel


class FedSSLStrategy(FedAvg):
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        ssl_weight: float = 0.5,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
        )
        self.ssl_weight = ssl_weight
        print(f"FedSSL Strategy initialized with SSL weight: {ssl_weight}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Dict]],
        failures: List[Tuple[ClientProxy, Exception]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        ssl_params_by_task = {"contrastive": [], "rotation": [], "jigsaw": []}
        backbone_params = []
        
        for client_proxy, fit_res in results:
            parameters = fit_res.parameters
            task_type = fit_res.metrics.get("ssl_task", "contrastive")
            
            weights = parameters_to_ndarrays(parameters)
            
            backbone_weights = weights[:-2]
            task_weights = weights[-2:]
            
            backbone_params.append((backbone_weights, fit_res.num_examples))
            ssl_params_by_task[task_type].append((task_weights, fit_res.num_examples))
        
        if backbone_params:
            aggregated_backbone = self._weighted_average(backbone_params)
        else:
            return None, {}
        
        aggregated_ssl_params = {}
        for task_type, task_params in ssl_params_by_task.items():
            if task_params:
                aggregated_ssl_params[task_type] = self._weighted_average(task_params)
        
        most_common_task = max(ssl_params_by_task.keys(), 
                              key=lambda k: len(ssl_params_by_task[k]))
        
        if most_common_task in aggregated_ssl_params:
            combined_params = aggregated_backbone + aggregated_ssl_params[most_common_task]
            aggregated_parameters = ndarrays_to_parameters(combined_params)
        else:
            return None, {}
        
        metrics_aggregated = self._aggregate_ssl_metrics(results)
        
        return aggregated_parameters, metrics_aggregated
    
    def _weighted_average(self, params_and_samples: List[Tuple[NDArrays, int]]) -> NDArrays:
        if not params_and_samples:
            return []
        
        total_samples = sum(samples for _, samples in params_and_samples)
        
        weighted_params = None
        
        for params, samples in params_and_samples:
            weight = samples / total_samples
            
            if weighted_params is None:
                weighted_params = [param * weight for param in params]
            else:
                for i in range(len(params)):
                    weighted_params[i] += params[i] * weight
        
        return weighted_params
    
    def _aggregate_ssl_metrics(self, results: List[Tuple[ClientProxy, Dict]]) -> Dict[str, Scalar]:
        metrics = {}
        
        ssl_losses = []
        ssl_accuracies = []
        task_counts = {"contrastive": 0, "rotation": 0}
        
        for _, fit_res in results:
            if "ssl_train_loss" in fit_res.metrics:
                ssl_losses.append(fit_res.metrics["ssl_train_loss"])
            
            if "ssl_accuracy" in fit_res.metrics:
                ssl_accuracies.append(fit_res.metrics["ssl_accuracy"])
            
            task_type = fit_res.metrics.get("ssl_task", "contrastive")
            task_counts[task_type] += 1
        
        if ssl_losses:
            metrics["avg_ssl_loss"] = np.mean(ssl_losses)
        
        if ssl_accuracies:
            metrics["avg_ssl_accuracy"] = np.mean(ssl_accuracies)
        
        metrics.update({f"task_{k}_count": v for k, v in task_counts.items()})
        
        return metrics


class FedPerStrategy(FedAvg):
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        personalization_layers: int = 2,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
        )
        self.personalization_layers = personalization_layers
        self.client_personal_params = {}
        print(f"FedPer Strategy initialized with {personalization_layers} personalization layers")
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, Dict]]:
        config_list = super().configure_fit(server_round, parameters, client_manager)
        
        personalized_configs = []
        for client_proxy, config in config_list:
            client_id = getattr(client_proxy, 'cid', str(client_proxy))
            
            if client_id in self.client_personal_params:
                config["personalized_params"] = self.client_personal_params[client_id]
            
            config["personalization_layers"] = self.personalization_layers
            personalized_configs.append((client_proxy, config))
        
        return personalized_configs
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Dict]],
        failures: List[Tuple[ClientProxy, Exception]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        shared_params = []
        
        for client_proxy, fit_res in results:
            client_id = getattr(client_proxy, 'cid', str(client_proxy))
            parameters = fit_res.parameters
            
            weights = parameters_to_ndarrays(parameters)
            
            num_shared = len(weights) - self.personalization_layers
            shared_weights = weights[:num_shared]
            personal_weights = weights[num_shared:]
            
            self.client_personal_params[client_id] = personal_weights
            
            shared_params.append((shared_weights, fit_res.num_examples))
        
        if shared_params:
            aggregated_shared = self._weighted_average(shared_params)
            aggregated_parameters = ndarrays_to_parameters(aggregated_shared)
        else:
            return None, {}
        
        metrics_aggregated = self._aggregate_metrics(results)
        
        return aggregated_parameters, metrics_aggregated
    
    def _weighted_average(self, params_and_samples: List[Tuple[NDArrays, int]]) -> NDArrays:
        if not params_and_samples:
            return []
        
        total_samples = sum(samples for _, samples in params_and_samples)
        weighted_params = None
        
        for params, samples in params_and_samples:
            weight = samples / total_samples
            
            if weighted_params is None:
                weighted_params = [param * weight for param in params]
            else:
                for i in range(len(params)):
                    weighted_params[i] += params[i] * weight
        
        return weighted_params
    
    def _aggregate_metrics(self, results: List[Tuple[ClientProxy, Dict]]) -> Dict[str, Scalar]:
        metrics = {}
        
        all_metrics = [fit_res.metrics for _, fit_res in results]
        
        for key in ["ssl_train_loss", "ssl_accuracy"]:
            values = [m.get(key) for m in all_metrics if key in m]
            if values:
                metrics[f"avg_{key}"] = np.mean(values)
        
        return metrics


class MultiTaskSSLStrategy(FedAvg):
    
    def __init__(
        self,
        strategy_type: str = "FedAvg",
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        client_task_mapping: Optional[Dict[int, str]] = None,
        evaluate_metrics_aggregation_fn: Optional[callable] = None,
        **strategy_kwargs
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            on_fit_config_fn=self._get_fit_config,
            on_evaluate_config_fn=self._get_evaluate_config,
        )
        
        self.strategy_type = strategy_type
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.client_task_mapping = {1: 'rotation', 2: 'contrastive'}
        self.server_metrics_log = []
        self.client_task_mapping = client_task_mapping or {1: 'rotation', 2: 'contrastive'}

    def _get_fit_config(self, server_round: int) -> Dict[str, any]:
        return {
            "epochs": 3,
            "server_round": server_round,
        }
    
    def _get_evaluate_config(self, server_round: int) -> Dict[str, any]:
        return {
            "server_round": server_round,
        }

    def weighted_loss_avg(self, results: List[Tuple[int, float]]) -> float:
        num_total_examples = sum(num_examples for num_examples, _ in results)
        weighted_losses = [num_examples * loss for num_examples, loss in results]
        return sum(weighted_losses) / num_total_examples
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Dict]],
        failures: List[Tuple[ClientProxy, Exception]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if self.strategy_type == "FedSSL":
            return self._aggregate_fedssl(server_round, results, failures)
        elif self.strategy_type == "FedPer":
            return self._aggregate_fedper(server_round, results, failures)
        else:
            return self._aggregate_fedavg(server_round, results, failures)
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        loss_aggregated = self.weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        if self.evaluate_metrics_aggregation_fn:
            metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.evaluate_metrics_aggregation_fn(metrics)
        else:
            aggregated_metrics = {}

        for client_proxy, evaluate_res in results:
            client_id = evaluate_res.metrics.get("client_id", "unknown")
            accuracy = evaluate_res.metrics.get("accuracy")
            if accuracy is not None:
                logging.info(f"Client {client_id} accuracy: {accuracy}")
        
        if aggregated_metrics.get("accuracy"):
            logging.info(f"Round {server_round} aggregated downstream accuracy: {aggregated_metrics['accuracy']}")

        return loss_aggregated, aggregated_metrics

    def _aggregate_fedavg(self, server_round: int, results: List[Tuple[ClientProxy, Dict]],
                         failures: List[Tuple[ClientProxy, Exception]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        return super().aggregate_fit(server_round, results, failures)
    
    def _aggregate_fedssl(self, server_round: int, results: List[Tuple[ClientProxy, Dict]],
                         failures: List[Tuple[ClientProxy, Exception]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        task_results = {"contrastive": [], "rotation": []}
        
        for client_proxy, fit_res in results:
            task_type = fit_res.metrics.get("ssl_task", "contrastive")
            task_results[task_type].append((client_proxy, fit_res))
        
        task_aggregates = {}
        total_samples = 0
        
        for task_type, task_res in task_results.items():
            if task_res:
                agg_params, agg_metrics = super().aggregate_fit(server_round, task_res, [])
                if agg_params:
                    task_samples = sum(fit_res.num_examples for _, fit_res in task_res)
                    task_aggregates[task_type] = (agg_params, task_samples)
                    total_samples += task_samples
        
        if task_aggregates:
            dominant_task = max(task_aggregates.keys(), 
                              key=lambda k: task_aggregates[k][1])
            final_params = task_aggregates[dominant_task][0]
            
            metrics = self._aggregate_ssl_metrics(results)
            
            return final_params, metrics
        
        return None, {}
    
    def _aggregate_fedper(self, server_round: int, results: List[Tuple[ClientProxy, Dict]],
                         failures: List[Tuple[ClientProxy, Exception]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        shared_params = []
        
        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get("client_id", 1)
            parameters = fit_res.parameters
            
            weights = parameters_to_ndarrays(parameters)
            
            num_personal = getattr(self, 'personalization_layers', 2)
            shared_weights = weights[:-num_personal] if len(weights) > num_personal else weights
            personal_weights = weights[-num_personal:] if len(weights) > num_personal else []
            
            if not hasattr(self, 'client_personal_params'):
                self.client_personal_params = {}
            self.client_personal_params[client_id] = personal_weights
            
            shared_params.append((shared_weights, fit_res.num_examples))
        
        if shared_params:
            aggregated_shared = self._weighted_average(shared_params)
            aggregated_parameters = ndarrays_to_parameters(aggregated_shared)
            
            metrics = self._aggregate_metrics(results)
            
            return aggregated_parameters, metrics
        
        return None, {}
    
    def _weighted_average(self, params_and_samples: List[Tuple[NDArrays, int]]) -> NDArrays:
        if not params_and_samples:
            return []
        
        total_samples = sum(samples for _, samples in params_and_samples)
        weighted_params = None
        
        for params, samples in params_and_samples:
            weight = samples / total_samples
            
            if weighted_params is None:
                weighted_params = [param * weight for param in params]
            else:
                for i in range(len(params)):
                    weighted_params[i] += params[i] * weight
        
        return weighted_params
    
    def _aggregate_ssl_metrics(self, results: List[Tuple[ClientProxy, Dict]]) -> Dict[str, Scalar]:
        metrics = {}
        
        ssl_losses = []
        ssl_accuracies = []
        task_counts = {"contrastive": 0, "rotation": 0}
        
        for _, fit_res in results:
            if "ssl_train_loss" in fit_res.metrics:
                ssl_losses.append(fit_res.metrics["ssl_train_loss"])
            
            if "ssl_accuracy" in fit_res.metrics:
                ssl_accuracies.append(fit_res.metrics["ssl_accuracy"])
            
            task_type = fit_res.metrics.get("ssl_task", "contrastive")
            task_counts[task_type] += 1
        
        if ssl_losses:
            metrics["avg_ssl_loss"] = float(np.mean(ssl_losses))
        
        if ssl_accuracies:
            metrics["avg_ssl_accuracy"] = float(np.mean(ssl_accuracies))
        
        for task, count in task_counts.items():
            metrics[f"task_{task}_count"] = count
        
        return metrics
    
    def _aggregate_evaluation_metrics(self, results: List[Tuple[ClientProxy, EvaluateRes]]) -> Dict[str, Scalar]:
        if not results:
            return {}
        
        all_accuracies = []
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        all_ssl_accuracies = []
        all_ssl_losses = []
        total_examples = 0
        
        for client_proxy, evaluate_res in results:
            metrics = evaluate_res.metrics
            num_examples = evaluate_res.num_examples
            
            if "accuracy" in metrics:
                all_accuracies.append((metrics["accuracy"], num_examples))
            if "precision" in metrics:
                all_precisions.append((metrics["precision"], num_examples))
            if "recall" in metrics:
                all_recalls.append((metrics["recall"], num_examples))
            if "f1_score" in metrics:
                all_f1_scores.append((metrics["f1_score"], num_examples))
            if "ssl_accuracy" in metrics:
                all_ssl_accuracies.append((metrics["ssl_accuracy"], num_examples))
            if "ssl_loss" in metrics:
                all_ssl_losses.append((metrics["ssl_loss"], num_examples))
            
            total_examples += num_examples
        
        aggregated = {}
        
        if all_accuracies:
            aggregated["accuracy"] = sum(acc * examples for acc, examples in all_accuracies) / total_examples
        if all_precisions:
            aggregated["precision"] = sum(prec * examples for prec, examples in all_precisions) / total_examples
        if all_recalls:
            aggregated["recall"] = sum(rec * examples for rec, examples in all_recalls) / total_examples
        if all_f1_scores:
            aggregated["f1_score"] = sum(f1 * examples for f1, examples in all_f1_scores) / total_examples
        if all_ssl_accuracies:
            aggregated["ssl_accuracy"] = sum(ssl_acc * examples for ssl_acc, examples in all_ssl_accuracies) / total_examples
        if all_ssl_losses:
            aggregated["ssl_loss"] = sum(ssl_loss * examples for ssl_loss, examples in all_ssl_losses) / total_examples
        
        return aggregated

    def _aggregate_metrics(self, results: List[Tuple[ClientProxy, Dict]]) -> Dict[str, Scalar]:
        metrics = {}
        
        ssl_train_losses = []
        ssl_accuracies = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for _, fit_res in results:
            metrics_client = fit_res.metrics
            ssl_train_losses.append(metrics_client.get("ssl_train_loss"))
            ssl_accuracies.append(metrics_client.get("ssl_accuracy"))
            accuracies.append(metrics_client.get("accuracy"))
            precisions.append(metrics_client.get("precision"))
            recalls.append(metrics_client.get("recall"))
            f1_scores.append(metrics_client.get("f1_score"))

        metrics["avg_ssl_loss"] = float(np.mean([x for x in ssl_train_losses if x is not None]))
        metrics["avg_ssl_accuracy"] = float(np.mean([x for x in ssl_accuracies if x is not None]))
        metrics["avg_accuracy"] = float(np.mean([x for x in accuracies if x is not None]))
        metrics["avg_precision"] = float(np.mean([x for x in precisions if x is not None]))
        metrics["avg_recall"] = float(np.mean([x for x in recalls if x is not None]))
        metrics["avg_f1_score"] = float(np.mean([x for x in f1_scores if x is not None]))

        best_client_id = -1
        best_accuracy = -1.0
        for _, fit_res in results:
            client_id = fit_res.metrics.get("client_id")
            accuracy = fit_res.metrics.get("accuracy")
            if accuracy is not None and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_client_id = client_id
        metrics["best_client_id"] = best_client_id

        self._log_server_metrics(
            server_round=results[0][1].metrics.get("round", -1),
            aggregation_method=self.strategy_type,
            metrics=metrics
        )

        return metrics

    def _log_server_metrics(self, server_round: int, aggregation_method: str, metrics: Dict[str, Scalar], notes: str = ""):
        metrics_data = {
            "Round": server_round,
            "AggregationMethod": aggregation_method,
            "Acc_Avg": metrics.get("avg_accuracy"),
            "Precision_Avg": metrics.get("avg_precision"),
            "Recall_Avg": metrics.get("avg_recall"),
            "F1_Score_Avg": metrics.get("avg_f1_score"),
            "SSL_Loss_Avg": metrics.get("avg_ssl_loss"),
            "SSL_Accuracy_Avg": metrics.get("avg_ssl_accuracy"),
            "Best_Client": metrics.get("best_client_id"),
            "Comments": notes
        }
        self.server_metrics_log.append(metrics_data)

        print()
        print("🌟" * 40)
        print(f"🖥️  SERVER AGGREGATION RESULTS - ROUND {server_round}")
        print("🌟" * 40)
        print(f"📊 Aggregation Method: {aggregation_method}")
        print(f"🏆 Best Performing Client: {metrics.get('best_client_id', 'N/A')}")
        print()
        print("📈 AGGREGATED CLASSIFICATION METRICS:")
        print(f"   ✅ Average Accuracy:  {metrics.get('avg_accuracy', 0):.4f} ({metrics.get('avg_accuracy', 0)*100:.2f}%)")
        print(f"   🎯 Average Precision: {metrics.get('avg_precision', 0):.4f} ({metrics.get('avg_precision', 0)*100:.2f}%)")
        print(f"   🔍 Average Recall:    {metrics.get('avg_recall', 0):.4f} ({metrics.get('avg_recall', 0)*100:.2f}%)")
        print(f"   ⚖️  Average F1-Score:  {metrics.get('avg_f1_score', 0):.4f} ({metrics.get('avg_f1_score', 0)*100:.2f}%)")
        print()
        print("🔧 SSL METRICS:")
        print(f"   📉 Average SSL Loss:     {metrics.get('avg_ssl_loss', 0):.4f}")
        print(f"   📈 Average SSL Accuracy: {metrics.get('avg_ssl_accuracy', 0):.4f} ({metrics.get('avg_ssl_accuracy', 0)*100:.2f}%)")
        print("🌟" * 40)
        print()

        log_df = pd.DataFrame(self.server_metrics_log)
        log_df.to_csv("server_metrics.csv", index=False)


def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    return [np.array(param) for param in parameters.tensors]


def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    return Parameters(tensors=[arr.tolist() for arr in ndarrays], tensor_type="")


def create_ssl_strategy(
    strategy_type: str = "MultiTaskSSL",
    min_available_clients: int = 2,
    evaluate_metrics_aggregation_fn: Optional[callable] = None,
    **kwargs
):
    print(f"[DEBUG] create_ssl_strategy called")
    
    if strategy_type == "MultiTaskSSL":
        return MultiTaskSSLStrategy(
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            **kwargs
        )
    elif strategy_type == "FedSSL":
        return FedSSLStrategy(
            min_available_clients=min_available_clients,
            **kwargs
        )
    elif strategy_type == "FedPer":
        return FedPerStrategy(
            min_available_clients=min_available_clients,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")