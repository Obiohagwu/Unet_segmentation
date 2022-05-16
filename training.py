
import torch 
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim 
from modelUnet import UNET

from utils import(
    get_loaders,
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_img,
)

# lets initialoze the hyperparameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_WIDTH = 240 # Was initially 1918
IMAGE_HEIGHT = 160 # Was initially 1280
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
PIN_MEMORY = True
LOAD_MODEL = False






# Implement train function

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader) # for progress bars

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # FORWARD PASS
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # BACKWARDS PASS
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop

        loop.set_postfix(loss=loss.item())


def main():
    # We are going to transform our train and val input using albumentaios

    train_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


    # If it were the case that we were doing some multiclass segmentation,
    # we would have out_channel=3 for example
    # We would change loss_fn to cross entropy
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # becuase we didn't apply sigmoid
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)
        check_accuracy(val_loader, model, device=DEVICE)
        save_predictions_as_img(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

        


def test():
    print("Test!")

if __name__ == "__main__":
    main()