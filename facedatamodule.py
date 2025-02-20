from torchvision import transforms
from torch.utils.data import DataLoader
from facedataset import obj_facedataset
import pytorch_lightning as pl


class FaceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, img_size=64, batch_size=8):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        self.dataset = obj_facedataset(self.data_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
obj_facedatamodule=FaceDataModule()
