from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src import MEDIA_PIPE_SKELETONS_DIR
from src.tasks.pose_classification_task import PoseClassificationTask
from src.datasets.skeleton_dataset import SkeletonDataset

# --- Configuration ---
SEQUENCE_LENGTH = 150  # Number of frames in each sequence
NUM_KEYPOINTS = 33    # Number of keypoints per frame (e.g., COCO format)
NUM_FEATURES_PER_FRAME = NUM_KEYPOINTS * 2 # x, y for each keypoint
NUM_CLASSES = 4       # Example: 'throw', 'pull', 'squat', 'tennis'
BATCH_SIZE = 4
HIDDEN_SIZE = 256
NUM_LAYERS = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 300

if __name__ == "__main__":
    # --- 1. Data ---
    train_dataset = SkeletonDataset(
            keypoints_dir=MEDIA_PIPE_SKELETONS_DIR,
            file_list=["0003.json", "1300.json", "1806.json", '2200.json', '2202.json'],
            labels=[0, 1, 2, 3, 3],
            max_len=SEQUENCE_LENGTH
        )
    val_dataset = SkeletonDataset(
            keypoints_dir=MEDIA_PIPE_SKELETONS_DIR,
            file_list=["0009.json", "1301.json", "1807.json", '2204.json'],
            labels=[0, 1, 2, 3],
            max_len=SEQUENCE_LENGTH
        )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


    # --- 3. Training ---

    # Initialize the model
    task = PoseClassificationTask(
        input_size=NUM_FEATURES_PER_FRAME,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-pose-classifier-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20, # Stop if val_loss doesn't improve for 20 epochs
        verbose=True,
        mode='min'
    )

    # Initialize a Trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='auto', # Use GPU if available, otherwise CPU
        devices=1,          # Use 1 device (GPU or CPU)
        callbacks=[
            checkpoint_callback,
            # early_stopping_callback,
            ],
        logger=True,        # Enable TensorBoard logging
        log_every_n_steps=1, # Log every N steps in training
    )

    # Train the model
    print("\n--- Starting Training ---")
    trainer.fit(task, train_loader, val_loader)
