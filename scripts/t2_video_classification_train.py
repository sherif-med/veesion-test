from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.tasks.video_classification import VideoClassificationTask
from src.datasets.videos_dataset import VideosDataset
from src import VIDEOS_DIR

# --- Configuration ---
BATCH_SIZE = 1
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_CLASSES = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
PRETRAINED_FRAME_ENCODER_PATH = "checkpoints/enocder/best-frame-encoder-epoch=18-ssl_loss=0.88.ckpt"

if __name__ == "__main__":
    # --- 1. Data ---
    train_dataset = VideosDataset(
        root_dir=VIDEOS_DIR,
        file_list=["0003.mp4", "1300.mp4", "1806.mp4", '2200.mp4', '2202.mp4'],
        labels=[0, 1, 2, 3, 3],
        )
    val_dataset = VideosDataset(
            root_dir=VIDEOS_DIR,
            file_list=["0009.mp4", "1301.mp4", "1807.mp4", '2204.mp4'],
            labels=[0, 1, 2, 3],
        )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- 3. Training ---

    # Initialize the model
    task = VideoClassificationTask(
        frame_encoder_ckpt=PRETRAINED_FRAME_ENCODER_PATH,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/video_classification/',
        filename='best-video-classifier-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5, # Stop if val_loss doesn't improve for 5 epochs
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
            early_stopping_callback,
            ],
        logger=True,        # Enable TensorBoard logging
        log_every_n_steps=1, # Log every N steps in training
    )

    # Train the model
    print("\n--- Starting Training ---")
    trainer.fit(task, train_loader, val_loader)
