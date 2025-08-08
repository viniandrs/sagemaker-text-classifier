from transformers import TrainerCallback

# Custom callback to log metrics
class LogMetricsCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:  # Only log once per node
            self.logger.info(f"Step: {state.global_step}, Metrics: {logs}")

# Custom callback for CloudWatch logging
class CloudWatchMetricsCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Format metrics for CloudWatch
            metrics = {
                'train_loss': logs.get('loss', None),
                'eval_loss': logs.get('eval_loss', None),
                'eval_accuracy': logs.get('eval_accuracy', None),
                'epoch': logs.get('epoch', None),
                'step': state.global_step
            }
            self.logger.info(f"[Metrics] {metrics}")