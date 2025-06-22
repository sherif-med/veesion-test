import os
from pytorch_lightning.callbacks import Callback

class PostPredictionCallback(Callback):
    def __init__(self, output_dir, write_interval, preds_items_callback):
        self.output_dir = output_dir
        self.write_interval = write_interval
        self.preds_items_callback = preds_items_callback

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.write_interval == "batch":
            self.write_predictions(outputs, batch)

    def write_predictions(self, outputs, batch):
        for key, callback in self.preds_items_callback.items():
            output_dir = os.path.join(self.output_dir, key)
            os.makedirs(output_dir, exist_ok=True)
            callback(output_dir, outputs[key], batch)
