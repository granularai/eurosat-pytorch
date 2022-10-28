import argparse, os, glob, json, torch, requests

import numpy as np

from tqdm.auto import tqdm
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from dataset import EuroSATDataset
from utils import update_confusion_matrix, print_loss_metrics, upload_artifacts

parser = argparse.ArgumentParser(description="Arguments for eurosat training")

parser.add_argument("--user", type=str, dest="email")
parser.add_argument("--pass", type=str, dest="passwd")
parser.add_argument("--org", type=str, dest="org_id")

args = parser.parse_args()

root_url = "https://api-staging.granular.ai"

login_url = f"{root_url}/callisto/auth/v1/login"
experiment_url = f"{root_url}/dione/api/v1/experiments"
exp_artifacts_url = f"{root_url}/dione/api/v1/artifacts"

payload = json.dumps({"email": args.email, "password": args.passwd })
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

response = requests.post(url=login_url, headers=headers, data=payload)

user_id = response.json()["user"]["id"]
auth_cookie = response.cookies["callisto_auth"]

version = f"EuroSAT-{datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')}"

root_dir = "/home/akash/Downloads/EuroSAT"

root_exp_dir = f"{root_dir}/experiments/{version}"

root_exp_logs_dir = f"{root_exp_dir}/logs"
root_exp_states_dir = f"{root_exp_dir}/states"

os.makedirs(root_exp_dir, exist_ok=True)

os.makedirs(root_exp_logs_dir, exist_ok=True)
os.makedirs(root_exp_states_dir, exist_ok=True)

code_dir = "/home/akash/Documents/eurosat-pytorch"

task_id = "622946b7e031612147d3fbd5"
dataset_id = "6336e917dbb9589c9ad501ca"

dst_bucket_path = f"gs://geoengine-dataset-{task_id}"

upload_artifacts(mode="code",src_path=code_dir, dst_path=dst_bucket_path, version=version)

model_name = "ResNet50"

num_workers = 4
batch_size = 512
num_epochs = 100

lr = 0.01
momemtum = 0.9
weight_decay = 1e-4

writer = SummaryWriter(log_dir=root_exp_logs_dir)

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

payload = json.dumps({
    "projectId": task_id,
    "exportId": dataset_id,
    "name": "EuroSAT Experiment",
    "description": "ResNet50 training on EuroSAT dataset",
    "tags": ["classification", "imported", "standard", "resnet"],
    "gitUrl": "https://github.com/granularai/eurosat-pytorch.git",
    "framework": "PyTorch",
    "experimentUrl": f"{dst_bucket_path}/experiments/{version}",
    "params": {
        "device": "cuda:0",
        "num-epochs": num_epochs,
        "num-workers": num_workers,
        "batch-size": batch_size,
        "optimizer": {
            "name": "SGD",
            "learning-rate": 0.01,
            "momentum": 0.4,
            "weight-decay": 1e-4
        },
        "scheduler": {
            "name": "StepLR",
            "step-size": 30,
            "gamma": 0.1
        },
        "model": {
            "name": model_name,
            "num-channels": 3,
            "num-classes": num_classes
        },
        "loss": "CrossEntropyLoss",
        "metrics": ["precision", "recall", "f1-score"]
    }
})
params = { "orgId": args.org_id }
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'Cookie': f'callisto_auth={auth_cookie}'}

response = requests.post(
    url=experiment_url, 
    data=payload, 
    headers=headers, params=params)

print(response)

experiment_id = response.json()["experiment"]["id"]

print(f"Experiment Created : {experiment_id}")

try:
    max_val_f1 = -1
    for epoch in range(1,num_epochs+1):
        print(f"Epoch : {epoch}")

        artifact_meta = {}

        state_file_path = f"{root_exp_states_dir}/{model_name}-{'ckpt-{:03d}.pth'.format(epoch)}"

        print("\t Train : ")

        losses = []
        conf_matrix = np.zeros([num_classes,num_classes], dtype=np.uint32) 
        for idx, sample in enumerate(tqdm(train_dl)):
            inputs = sample["image"].float().to(device)
            labels = sample["label"].to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs,labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            losses += [loss.item()]
            update_confusion_matrix(conf_matrix, outputs=outputs, labels=labels)

        avg_loss, avg_metric_map = print_loss_metrics(losses=losses, conf_matrix=conf_matrix, classes=classes)

        artifact_meta["Train/Loss"] = avg_loss
        for met_key in avg_metric_map:
            artifact_meta[f"Train/{met_key.title()}"] = avg_metric_map[met_key]

        for artf_key in artifact_meta:
            if artf_key.startswith("Train"):
                writer.add_scalar(artf_key, artifact_meta[artf_key], epoch)
        
        print("\t Val : ")

        losses = []
        conf_matrix = np.zeros([num_classes,num_classes], dtype=np.uint32)
        for idx, sample in enumerate(tqdm(val_dl)):
            inputs = sample["image"].float().to(device)
            labels = sample["label"].to(device)

            outputs = model(inputs)
            
            loss = criterion(outputs,labels)

            losses += [loss.item()]
            update_confusion_matrix(conf_matrix, outputs=outputs, labels=labels)
        

        avg_loss, avg_metric_map = print_loss_metrics(losses=losses, conf_matrix=conf_matrix, classes=classes)
        
        artifact_meta["Val/Loss"] = avg_loss
        for met_key in avg_metric_map:
            artifact_meta[f"Val/{met_key.title()}"] = avg_metric_map[met_key]

        for artf_key in artifact_meta:
            if artf_key.startswith("Val"):
                writer.add_scalar(artf_key, artifact_meta[artf_key], epoch)

        log_file_path = glob.glob(f"{root_exp_logs_dir}/*")[0]
        upload_artifacts(mode="logs",src_path=log_file_path, dst_path=dst_bucket_path, version=version)

        payload = json.dumps({
            "experimentId": experiment_id,
            "step": epoch,
            "metadata": artifact_meta
        })

        response = requests.post(url=exp_artifacts_url, data=payload, headers=headers, params=params)

        if artifact_meta["Val/F1 Score"] > max_val_f1:
            max_val_f1 = artifact_meta["Val/F1 Score"]

            chkpnt_map = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": artifact_meta
            }

            torch.save(chkpnt_map, state_file_path)
            upload_artifacts(mode="state",src_path=state_file_path, dst_path=dst_bucket_path, version=version)

            response = requests.put(
                url=f"{experiment_url}/{experiment_id}",
                data=json.dumps({ "bestModel": f"{dst_bucket_path}/experiments/{version}/states/{state_file_path.split('/')[-1]}"}),
                headers=headers, params=params
            )            

    response = requests.put(
        url=f"{experiment_url}/{experiment_id}",
        data=json.dumps({ "status": "success"}),
        headers=headers, params=params
    )
except Exception:
    response = requests.put(
        url=f"{experiment_url}/{experiment_id}",
        data=json.dumps({ "status": "failed"}),
        headers=headers, params=params
    )



