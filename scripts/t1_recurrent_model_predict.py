import argparse
import os
from torch.utils.data import DataLoader
from src.tasks.pose_classification_task import PoseClassificationTask
from src.datasets.skeleton_dataset import SkeletonDataset
from src.callbacks.post_pred_callback import PostPredictionCallback
import pytorch_lightning as pl

def main():
    parser = argparse.ArgumentParser(description='Run prediction on a trained model')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing input data (JSON files)')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save prediction outputs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for prediction')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run prediction on (cpu or cuda)')
    parser.add_argument('--seq_len', type=int, default=150, help='Sequence length for prediction')

    args = parser.parse_args()

    # Load the trained model
    task = PoseClassificationTask.load_from_checkpoint(args.checkpoint_path)

    # Create a dataset and data loader for prediction
    predict_dataset = SkeletonDataset(
        keypoints_dir=args.input_folder,
        file_list=None,  # Use all files in the input folder
        labels=None,  # Don't need labels for prediction
        max_len=args.seq_len  # Use the same sequence length as training
    )
    predict_loader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False)

    # Define a PostPredictionCallback to write predictions to output folder
    def write_predictions(output_dir, predictions, batch):
        output_text = "\n".join([f"{f} {p.item()}" for f, p in zip(batch["filename"], predictions)])
        with open(os.path.join(output_dir, 'predictions.txt'), 'a') as f:
            f.write(output_text)
            f.write('\n')

    post_pred_callback = PostPredictionCallback(
        output_dir=args.output_folder,
        write_interval="batch",
        preds_items_callback={"class": write_predictions}
    )

    # Run prediction
    task.to(args.device)
    task.eval()
    trainer = pl.Trainer(
        enable_progress_bar=True,
        logger=False,
        accelerator=args.device,
        callbacks=[post_pred_callback]
    )
    trainer.predict(task, dataloaders=predict_loader, return_predictions=False)

if __name__ == '__main__':
    main()