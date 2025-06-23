import argparse
import os
from torch.utils.data import DataLoader
from src.tasks.video_classification import VideoClassificationTask
from src.datasets.videos_dataset import VideosDataset
from src.callbacks.post_pred_callback import PostPredictionCallback
import pytorch_lightning as pl

def main():
    parser = argparse.ArgumentParser(description='Run prediction on a trained model')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing input data (mp4 files)')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save prediction outputs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for prediction')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run prediction on (cpu or cuda)')

    args = parser.parse_args()

    # Load the trained model
    task = VideoClassificationTask.load_from_checkpoint(args.checkpoint_path)

    # Create a dataset and data loader for prediction
    predict_dataset = VideosDataset(
        root_dir=args.input_folder,
    )
    predict_loader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False)

    # Define a PostPredictionCallback to write predictions to output folder
    def write_predictions(output_dir, predictions, batch):
        output_text = "\n".join([f"{f} {p.item()}" for f, p in zip(batch["file_path"], predictions)])
        with open(os.path.join(output_dir, 'predictions.txt'), 'a') as f:
            f.write(output_text)
            f.write('\n')

    post_pred_callback = PostPredictionCallback(
        output_dir=args.output_folder,
        write_interval="batch",
        preds_items_callback={"class": write_predictions}
    )

    # Run prediction
    trainer = pl.Trainer(
        enable_progress_bar=True,
        logger=False,
        accelerator=args.device,
        callbacks=[post_pred_callback]
    )
    trainer.predict(task, dataloaders=predict_loader, return_predictions=False)

if __name__ == '__main__':
    main()