import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os

class SimpleModel(L.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(10, 1)
        self.loss_fn = nn.MSELoss()
        self.train_predictions = []
        self.val_predictions = []

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        predictions = y_hat.cpu().detach().numpy().tolist()
        self.train_predictions.extend(predictions)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        predictions = y_hat.cpu().detach().numpy().tolist()
        self.val_predictions.extend(predictions)
        return loss
    
    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            print(f'Validation Before all gather || Epoch: {self.current_epoch} | number of predictions: {len(self.val_predictions)} | Global rank: ({self.global_rank}) | Local Rank: ({self.trainer.local_rank})')

         # Synchronize processes to ensure all predictions are gathered
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Gather val_predictions from all GPUs
        gathered_predictions = self.all_gather(torch.tensor(self.val_predictions, device=self.device))

        # Only proceed on the global rank 0
        if self.global_rank == 0:
            # Flatten the gathered tensor and convert to list
            gathered_predictions = gathered_predictions.view(-1).cpu().numpy().tolist()
            # Print the combined length of val_predictions from all GPUs
            print(f'Validation All gathered - global_rank 0 || Global rank ({self.global_rank}): Epoch: {self.current_epoch} | number of predictions: {len(gathered_predictions)} | Global rank: ({self.global_rank}) | Local Rank: ({self.trainer.local_rank})')

        self.val_predictions.clear()
        return
    
    def on_train_epoch_end(self):
        if self.global_rank == 0:
            print(f'Train Before all gather || Epoch: {self.current_epoch} | number of predictions: {len(self.train_predictions)} | Global rank: ({self.global_rank}) | Local Rank: ({self.trainer.local_rank})')

         # Synchronize processes to ensure all predictions are gathered
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Gather val_predictions from all GPUs
        gathered_predictions = self.all_gather(torch.tensor(self.train_predictions, device=self.device))

        # Only proceed on the global rank 0
        if self.global_rank == 0:
            # Flatten the gathered tensor and convert to list
            gathered_predictions = gathered_predictions.view(-1).cpu().numpy().tolist()
            # Print the combined length of val_predictions from all GPUs
            print(f'Train All gathered - global_rank 0 || Global rank ({self.global_rank}): Epoch: {self.current_epoch} | number of predictions: {len(gathered_predictions)} | Global rank: ({self.global_rank}) | Local Rank: ({self.trainer.local_rank})')

        self.train_predictions.clear()
        return
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def main():
    # Create a simple dataset
    dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 1))
    val_dataset = TensorDataset(torch.randn(200, 10), torch.randn(200, 1))
    train_loader = DataLoader(dataset, batch_size=20, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=20)
    print(len(train_loader))
    print(len(val_loader))

    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    # Initialize model
    model = SimpleModel()

    # Training system setup
    nodes = int(os.getenv("SLURM_NNODES", 1))
    devices = int(os.getenv("SLURM_GPUS_ON_NODE", 1))
    accelerator = "gpu"
    strategy = 'ddp'
    L.seed_everything(42, workers=True)
    print(f"Number of nodes {nodes}, and GPUs {devices}")

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=5,
        num_nodes=nodes,
        devices=devices,  # Use all available GPUs
        accelerator=accelerator, 
        strategy=strategy,
        deterministic=False, 
        enable_progress_bar=False)

    # Start training
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
