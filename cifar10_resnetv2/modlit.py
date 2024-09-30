import torch
import torch.nn as nn
import torchvision
import lightning as L
from torchmetrics import Accuracy
from cifar10_resnetv2 import model


class CIFARResNetV2(L.LightningModule):
    def __init__(self, m):
        super().__init__()
        self.model = model.pick(m)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        def lr_fn(step):
            if step < 400:
                return 0.1
            elif step < 32000:
                return 1
            elif step < 48000:
                return 0.1
            else:
                return 0.01
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn),
            'name': 'learning_rate',
            'interval': 'step'
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        x_tr, y_tr = batch
        hypothesis = self.model(x_tr)
        correct_pred = torch.argmax(hypothesis, dim=1)
        loss = self.loss_fn(hypothesis, y_tr)
        acc = self.accuracy(correct_pred, y_tr)
        error_rate = (1 - acc)*100
        self.log("training loss", loss.item())
        self.log("training error rate", error_rate)

        return loss

    def validation_step(self, batch, batch_idx):
        x_val, y_val = batch
        hypothesis = self.model(x_val)
        loss = self.loss_fn(hypothesis, y_val)
        correct_pred = torch.argmax(hypothesis, dim=1)
        acc = self.accuracy(correct_pred, y_val)
        error_rate = (1 - acc) * 100
        self.log("validation loss", loss.item())
        self.log("validation error rate", error_rate)

    def test_step(self, batch, batch_idx):
        x_test, y_test = batch
        hypothesis = self.model(x_test)
        correct_pred = torch.argmax(hypothesis, dim=1)
        acc = self.accuracy(correct_pred, y_test)
        error_rate = (1 - acc) * 100
        self.log("test error rate", error_rate)