from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.tasks.ssl_frame import SSLFrameTask
from src.datasets.frame_pair_dataset import FramePairDataset
from src import VIDEOS_DIR

# --- Configuration ---
BATCH_SIZE = 4
PROJECTION_DIM = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20

if __name__ == "__main__":
    # --- 1. Data ---
    train_dataset = FramePairDataset(root_dir=VIDEOS_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # --- 3. Training ---

    # Initialize the model
    task = SSLFrameTask(
        batch_size=BATCH_SIZE,
        projection_dim=PROJECTION_DIM,
        learning_rate=LEARNING_RATE
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='ssl_loss',
        dirpath='checkpoints/enocder/',
        filename='best-frame-encoder-{epoch:02d}-{ssl_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    early_stopping_callback = EarlyStopping(
        monitor='ssl_loss',
        patience=5, # Stop if ssl_loss doesn't improve for 20 epochs
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
    trainer.fit(task, train_loader)
