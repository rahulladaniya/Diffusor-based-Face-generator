import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from facedatamodule import obj_facedatamodule
from diffuisonmodel import obj_diffusionmodel

def train_model(data_dir, img_size=64, batch_size=8, max_epochs=100, checkpoint_dir='checkpoints'):

    os.makedirs(checkpoint_dir, exist_ok=True)
    print("enter")
    data_module = obj_facedatamodule(
        data_dir,
        img_size=img_size,
        batch_size=batch_size
    )

    model = obj_diffusionmodel(img_size=img_size)

    # Define multiple checkpoint callbacks
    checkpoints = [
        # Save best models based on training loss
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-checkpoint-{epoch:02d}-{train_loss:.2f}',
            monitor='train_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        # Save periodic checkpoints every 5 epochs
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='periodic-checkpoint-{epoch:02d}',
            every_n_epochs=5,
            save_on_train_epoch_end=True,
        ),
        # Save last checkpoint
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='last-checkpoint',
            save_last=True,
        )
    ]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=max_epochs,
        callbacks=checkpoints,
        gradient_clip_val=1.0,
        precision="16-mixed",
        accumulate_grad_batches=4,
        enable_progress_bar=True,
        logger=True
    )

    # Train the model
    trainer.fit(model, data_module)
    
    # Save the final model state
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'img_size': img_size,
        'timesteps': model.timesteps
    }, final_model_path)
    
    return model


if __name__ == "__main__":
    # Set parameters
    DATA_DIR = "img_align_celeba"  # Update this path to your dataset location
    IMG_SIZE = 64
    BATCH_SIZE = 8
    MAX_EPOCHS = 1

    # Verify data directory exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist!")

    # Print number of images found
    image_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images in {DATA_DIR}")

    # Train model
    model = train_model(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS
    )

    