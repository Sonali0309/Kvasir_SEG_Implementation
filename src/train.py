import os, random, time, zipfile, urllib.request, csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models import UNet, count_parameters

SEED = 42
SIZE = 256
BATCH = 8
EPOCHS = 20
LR = 1e-3
DATA_ROOT = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def download_and_extract():
    DATA_ROOT.mkdir(exist_ok=True)
    archive = DATA_ROOT / "kvasir-seg.zip"
    if not archive.exists():
        url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
        print("Downloading official Kvasir-SEG archive...")
        urllib.request.urlretrieve(url, archive)
    extract_dir = DATA_ROOT / "raw"
    marker = extract_dir / ".extracted"
    if not marker.exists():
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zf: zf.extractall(extract_dir)
        marker.write_text("ok")
    image_files = list(extract_dir.rglob("images/*.jpg")) + list(extract_dir.rglob("images/*.png"))
    mask_files = list(extract_dir.rglob("masks/*.jpg")) + list(extract_dir.rglob("masks/*.png"))
    if not image_files or not mask_files:
        # fallback for archive folder naming variations
        all_imgs = [p for p in extract_dir.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")]
        mask_files = [p for p in all_imgs if "mask" in str(p.parent).lower()]
        image_files = [p for p in all_imgs if "mask" not in str(p.parent).lower()]
    masks_by_stem = {p.stem:p for p in mask_files}
    pairs = [(p, masks_by_stem[p.stem]) for p in image_files if p.stem in masks_by_stem]
    if len(pairs) < 900:
        raise RuntimeError(f"Expected approximately 1000 paired samples; found {len(pairs)}. Inspect {extract_dir}.")
    pairs = sorted(pairs, key=lambda x:x[0].stem)
    return pairs

def split_pairs(pairs):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(pairs))
    n = len(idx); n_train=int(.8*n); n_val=int(.1*n)
    train=[pairs[i] for i in idx[:n_train]]
    val=[pairs[i] for i in idx[n_train:n_train+n_val]]
    test=[pairs[i] for i in idx[n_train+n_val:]]
    return train,val,test

class PolypDataset(Dataset):
    def __init__(self, pairs, train=False):
        self.pairs=pairs; self.train=train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        image=Image.open(ip).convert("RGB").resize((SIZE,SIZE), Image.Resampling.BILINEAR)
        mask=Image.open(mp).convert("L").resize((SIZE,SIZE), Image.Resampling.NEAREST)
        x=np.asarray(image,dtype=np.float32)/255.0
        y=(np.asarray(mask,dtype=np.uint8)>127).astype(np.float32)
        if self.train and random.random()<0.5:
            x=np.fliplr(x).copy(); y=np.fliplr(y).copy()
        x=torch.from_numpy(x).permute(2,0,1)
        y=torch.from_numpy(y).unsqueeze(0)
        return x,y,ip.name

def dice_from_logits(logits, targets, eps=1e-7):
    probs=torch.sigmoid(logits); pred=(probs>=0.5).float()
    inter=(pred*targets).sum()
    return (2*inter+eps)/(pred.sum()+targets.sum()+eps)

def evaluate(model, loader, device):
    model.eval()
    tp=fp=fn=tn=0
    with torch.no_grad():
        for x,y,_ in loader:
            x=x.to(device); y=y.to(device)
            pred=(torch.sigmoid(model(x))>=0.5)
            y=y.bool()
            tp += (pred & y).sum().item()
            fp += (pred & ~y).sum().item()
            fn += (~pred & y).sum().item()
            tn += (~pred & ~y).sum().item()
    eps=1e-9
    fg_iou=tp/(tp+fp+fn+eps)
    bg_iou=tn/(tn+fp+fn+eps)
    miou=(fg_iou+bg_iou)/2
    dice=2*tp/(2*tp+fp+fn+eps)
    precision=tp/(tp+fp+eps)
    recall=tp/(tp+fn+eps)
    return {"foreground_iou":fg_iou,"background_iou":bg_iou,"miou":miou,
            "dice":dice,"precision":precision,"recall":recall}

def train_one(dilation, train_loader, val_loader, device, name):
    model=UNet(bottleneck_dilation=dilation).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    criterion=nn.BCEWithLogitsLoss()
    best=-1; patience=5; stale=0
    history=[]
    for epoch in range(1,EPOCHS+1):
        model.train(); losses=[]
        for x,y,_ in train_loader:
            x=x.to(device); y=y.to(device)
            optimizer.zero_grad()
            logits=model(x); loss=criterion(logits,y)
            loss.backward(); optimizer.step()
            losses.append(loss.item())
        val_metrics=evaluate(model,val_loader,device)
        score=val_metrics["dice"]
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),**val_metrics})
        print(f"{name} | epoch {epoch:02d}/{EPOCHS} | loss={np.mean(losses):.4f} | val_dice={score:.4f}")
        if score>best:
            best=score; stale=0
            torch.save(model.state_dict(),OUT/f"{name}_best.pt")
        else:
            stale+=1
            if stale>=patience: print("Early stopping."); break
    with open(OUT/f"{name}_history.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    model.load_state_dict(torch.load(OUT/f"{name}_best.pt",map_location=device))
    return model

def measure_latency(model, loader, device, warmup=10, repeats=50):
    x,_,_=next(iter(loader)); x=x[:1].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup): _=model(x)
        if device.type=="cuda": torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(repeats): _=model(x)
        if device.type=="cuda": torch.cuda.synchronize()
    return (time.perf_counter()-start)*1000/repeats

def main():
    seed_everything()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device, torch.cuda.get_device_name(0) if device.type=="cuda" else "")
    pairs=download_and_extract()
    train_pairs,val_pairs,test_pairs=split_pairs(pairs)
    print(f"Pairs: {len(pairs)} | train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    train_loader=DataLoader(PolypDataset(train_pairs,True),batch_size=BATCH,shuffle=True,num_workers=2,pin_memory=device.type=="cuda")
    val_loader=DataLoader(PolypDataset(val_pairs),batch_size=BATCH,shuffle=False,num_workers=2)
    test_loader=DataLoader(PolypDataset(test_pairs),batch_size=BATCH,shuffle=False,num_workers=2)
    rows=[]
    for name,dilation in [("baseline_unet",1),("dilated_unet",2)]:
        model=train_one(dilation,train_loader,val_loader,device,name)
        metrics=evaluate(model,test_loader,device)
        latency=measure_latency(model,test_loader,device)
        params=count_parameters(model)
        row={"model":name,"dilation":dilation,"parameters":params,"latency_ms_image":latency,**metrics}
        rows.append(row)
        print("TEST:",row)
    with open(OUT/"results_summary.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("\\nSaved outputs/results_summary.csv")
    print("Use the CSV values in the report; do not substitute example values.")

if __name__=="__main__": main()
