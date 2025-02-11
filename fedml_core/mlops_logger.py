import json
import logging

from fedml_api.distributed.fedavg_cross_silo.SysStats import SysStats


class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, "_instance"):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance


class MLOpsLogger(Singleton):
    def __init__(self):
        self.messenger = None
        self.args = None
        self.run_id = None
        self.edge_id = None
        self.sys_performances = SysStats()

    def set_messenger(self, msg_messenger, args=None):
        self.messenger = msg_messenger
        if args is not None:
            self.args = args
            self.run_id = args.run_id
            client_ids = json.loads(args.client_ids)
            self.edge_id = client_ids[0]

    def report_client_training_status(self, edge_id, status):
        topic_name = "fl_client/mlops/status"
        msg = {"edge_id": edge_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("report_client_training_status. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_client_id_status(self, run_id, edge_id, status):
        topic_name = "fl_client/mlops/" + str(edge_id) + "/status"
        msg = {"run_id": run_id, "edge_id": edge_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("report_client_id_status. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_status(self, run_id, status):
        topic_name = "fl_server/mlops/status"
        msg = {"run_id": run_id, "status": status}
        logging.info("report_server_training_status. msg = %s" % msg)
        message_json = json.dumps(msg)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_id_status(self, run_id, status):
        topic_name = "fl_server/mlops/id/status"
        msg = {"run_id": run_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("report_server_id_status. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_client_training_metric(self, metric_json):
        topic_name = "fl_client/mlops/training_metrics"
        logging.info("report_client_training_metric. message_json = %s" % metric_json)
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_metric(self, metric_json):
        topic_name = "fl_server/mlops/training_progress_and_eval"
        logging.info("report_server_training_metric. message_json = %s" % metric_json)
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_round_info(self, round_info):
        topic_name = "fl_client/mlops/training_roundx"
        logging.info("report_server_training_round_info. message_json = %s" % round_info)
        message_json = json.dumps(round_info)
        self.messenger.send_message_json(topic_name, message_json)

    def report_client_model_info(self, model_info_json):
        topic_name = "fl_server/mlops/client_model"
        logging.info("report_client_model_info. message_json = %s" % model_info_json)
        message_json = json.dumps(model_info_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_aggregated_model_info(self, model_info_json):
        topic_name = "fl_server/mlops/global_aggregated_model"
        logging.info("report_aggregated_model_info. message_json = %s" % model_info_json)
        message_json = json.dumps(model_info_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_system_metric(self, metric_json=None):
        topic_name = "fl_client/mlops/system_performance"
        if metric_json is None:
            self.sys_performances.produce_info()
            metric_json = {
                "run_id": self.run_id,
                "edge_id": self.edge_id,
                "cpu_utilization": round(self.sys_performances.get_cpu_utilization(), 4),
                "SystemMemoryUtilization": round(self.sys_performances.get_system_memory_utilization(), 4),
                "process_memory_in_use": round(self.sys_performances.get_process_memory_in_use(), 4),
                "process_memory_in_use_size": round(self.sys_performances.get_process_memory_in_use_size(), 4),
                "process_memory_available": round(self.sys_performances.get_process_memory_available(), 4),
                "process_cpu_threads_in_use": round(self.sys_performances.get_process_cpu_threads_in_use(), 4),
                "disk_utilization": round(self.sys_performances.get_disk_utilization(), 4),
                "network_traffic": round(self.sys_performances.get_network_traffic(), 4),
                "gpu_utilization": round(self.sys_performances.get_gpu_utilization(), 4),
                "gpu_temp": round(self.sys_performances.get_gpu_temp(), 4),
                "gpu_time_spent_accessing_memory": round(self.sys_performances.get_gpu_time_spent_accessing_memory(), 4),
                "gpu_memory_allocated": round(self.sys_performances.get_gpu_memory_allocated(), 4),
                "gpu_power_usage": round(self.sys_performances.get_gpu_power_usage(), 4),
            }
        logging.info("report_metric. message_json = %s" % metric_json)
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)


if __name__ == "__main__":
    pass
