import json
import pandas as pd
import rasterio as rio
from torch.utils.data import Dataset

class EuroSATDataset(Dataset):
    def __init__(self, mode, root_dir):
        vec_file = f"{root_dir}/vectors/{mode}.csv"
        meta_file = f"{root_dir}/vectors/metadata.json"
        with open(meta_file) as out:
            task_meta = json.load(out)

        classes = [lbl_meta["options"] for lbl_meta in task_meta["label:metadata"]][0]
        cls_idx_map = {cls: idx for idx, cls in enumerate(classes)}

        vec_df = pd.read_csv(vec_file)
        vec_df["image"] = vec_df["image:01"].apply(lambda x: f'{root_dir}/rasters/{x.split("/")[-1]}')
        vec_df["label"] = vec_df["land-use-land-cover-class"].apply(lambda x: cls_idx_map[x])
        vec_df.drop(['image-id','image:01','date:01','type','geometry','land-use-land-cover-class'],axis=1,inplace=True)

        self.vec_df = vec_df
        self.classes = classes

    def __len__(self):
        return len(self.vec_df)

    def __getitem__(self, idx):
        df_entry = self.vec_df.loc[idx]
        smpl_map = {
            "image": rio.open(df_entry["image"]).read(),
            "label": df_entry["label"]
        }
        
        return smpl_map