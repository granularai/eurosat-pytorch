import os, glob, json, torch

import numpy as np

from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from dataset import EuroSATDataset
from utils import update_confusion_matrix, print_loss_metrics

root_dir = "/home/akash/Downloads/EuroSAT"
num_workers = 4
batch_size = 512

num_epochs = 100

lr = 0.01
momemtum = 0.9
weight_decay = 1e-4

device = torch.device("cuda")

train_ds = EuroSATDataset(root_dir=root_dir, mode="train")
val_ds = EuroSATDataset(root_dir=root_dir, mode="val")

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

classes = train_ds.classes
num_classes = len(classes)

model = resnet50(num_classes=num_classes).to(device)

criterion = CrossEntropyLoss().to(device)
optimizer = SGD(model.parameters(),lr=lr,momentum=momemtum,weight_decay=weight_decay)
scheduler = StepLR(optimizer,step_size=30,gamma=0.1)

max_val_f1 = -1
for epoch in range(1,101):
    print(f"epoch {epoch}")

    weights_path = "/home/akash/Downloads/EuroSAT/weights/resnet50_ckpt_{:03d}.pth".format(epoch)

    losses = []
    conf_matrix = np.zeros([num_classes,num_classes], dtype=np.uint32) 
    for idx, sample in enumerate(train_dl):
        inputs = sample["image"].float().to(device)
        labels = sample["label"].to(device)
        
        outputs = model(inputs)
        
        loss = criterion(outputs,labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        losses += [loss.item()]
        update_confusion_matrix(conf_matrix, outputs=outputs, labels=labels)

    print("\t Train : ")
    train_f1 = print_loss_metrics(losses=losses, conf_matrix=conf_matrix, classes=classes)

    losses = []
    conf_matrix = np.zeros([num_classes,num_classes], dtype=np.uint32)
    for idx, sample in enumerate(val_dl):
        inputs = sample["image"].float().to(device)
        labels = sample["label"].to(device)

        outputs = model(inputs)
        
        loss = criterion(outputs,labels)

        losses += [loss.item()]
        update_confusion_matrix(conf_matrix, outputs=outputs, labels=labels)
    
    print("\t Val : ")
    val_f1 = print_loss_metrics(losses=losses, conf_matrix=conf_matrix, classes=classes)
    
    if val_f1 > max_val_f1:
        max_val_f1 = val_f1
        torch.save(model.state_dict(), weights_path)


