import argparse
import os
import shutil
import sys
import zipfile
import warnings

warnings.filterwarnings("ignore")

import timm
import timm.data
import torch
import numpy as np
import pandas as pd
from PIL import Image
import clip
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

DATASET_CONFIGS = {
    "waterbirds_0.95": {
        "name": "Waterbirds-0.95",
        "data_dir": "./data/waterbirds/waterbird0.95/",
    },
    "waterbirds_1.0": {
        "name": "Waterbirds-1.0",
        "data_dir": "./data/waterbirds/waterbird1.0/",
    },
}


class LogWriter:
    def __init__(self, log_f):
        self.log_f = log_f

    def write(self, s):
        self.log_f.write(s)
        self.log_f.flush()

    def flush(self):
        self.log_f.flush()


class TeeWriter:
    def __init__(self, console_f, log_f):
        self.console_f = console_f
        self.log_f = log_f

    def write(self, s):
        self.console_f.write(s)
        self.console_f.flush()
        self.log_f.write(s)
        self.log_f.flush()

    def flush(self):
        self.console_f.flush()
        self.log_f.flush()


def create_original_transform(model_name="resnet50"):
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
    data_config = timm.data.resolve_model_data_config(backbone)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return transform


def _quantile_normalize(cam_i, p_low=1, p_high=99):
    vmin, vmax = np.percentile(cam_i, [p_low, p_high])
    if vmax - vmin < 1e-8:
        return np.zeros_like(cam_i, dtype=np.float32)
    cam_i = np.clip(cam_i, vmin, vmax)
    return (cam_i - vmin) / (vmax - vmin)


class CLIPGradCAM:
    def __init__(self, prompts, model_name="RN50"):
        self.prompts = prompts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.target_layer = self.model.visual.layer4[2]
        self._features = None
        self._gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._features = output

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0]

        self.fh = self.target_layer.register_forward_hook(forward_hook)
        self.bh = self.target_layer.register_full_backward_hook(backward_hook)

    def _generate_cam(self):
        alpha = self._gradients.mean(dim=(2, 3), keepdim=True)
        weighted = (alpha * self._features).sum(dim=1)
        cam = F.relu(weighted)
        B, _, _ = cam.shape
        for i in range(B):
            c_min, c_max = cam[i].min(), cam[i].max()
            if (c_max - c_min) > 1e-8:
                cam[i] = (cam[i] - c_min) / (c_max - c_min)
        return cam.detach().cpu().numpy()

    def calculate_gradcam(self, image):
        if isinstance(image, Image.Image):
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)

        text_tokens = clip.tokenize(self.prompts).to(self.device)
        accum_cam = None

        for tokens in text_tokens:
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(tokens.unsqueeze(0))

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            sim = 100.0 * image_features @ text_features.T
            score = sim.squeeze(1)

            gradients = torch.autograd.grad(
                score, self._features, create_graph=False, retain_graph=True
            )[0]

            alpha = gradients.mean(dim=(2, 3), keepdim=True)
            weighted = (alpha * self._features).sum(dim=1)
            cam = F.relu(weighted)

            cam_i = _quantile_normalize(cam[0].detach().cpu().numpy(), p_low=1, p_high=99)

            if accum_cam is None:
                accum_cam = cam_i
            else:
                accum_cam += cam_i

        return accum_cam / len(self.prompts)

    def remove_hooks(self):
        self.fh.remove()
        self.bh.remove()


class WaterbirdDataset(Dataset):
    def __init__(
        self,
        df,
        img_dir,
        transform=None,
        calculate_gradcam=True,
        prompts=None,
        tqdm_file=None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.calculate_gradcam = calculate_gradcam
        self.prompts = prompts if prompts is not None else ["a bird"]

        if self.calculate_gradcam:
            self.clip_gradcam = CLIPGradCAM(prompts=self.prompts)
            self.gradcam_maps = {}

            total_images = len(self.df)
            with tqdm(
                total=total_images,
                desc="Calculating GradCAM",
                unit="img",
                file=tqdm_file,
            ) as pbar:
                for idx in range(total_images):
                    img_path = os.path.join(
                        self.img_dir, self.df.iloc[idx]["img_filename"]
                    )
                    image = Image.open(img_path).convert("RGB")
                    cam = self.clip_gradcam.calculate_gradcam(image)
                    self.gradcam_maps[idx] = cam
                    pbar.update(1)

            self.clip_gradcam.remove_hooks()
            del self.clip_gradcam

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["img_filename"])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(row["y"], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        else:
            default_transform = create_original_transform()
            image = default_transform(image)

        if self.calculate_gradcam:
            cam_np = self.gradcam_maps[idx]
            gradcam = torch.tensor(cam_np, dtype=torch.float32)
            return image, label, gradcam
        else:
            return image, label


class VAE_TWOCLASSIFIERS(nn.Module):
    def __init__(self, latent_dim=512):
        super(VAE_TWOCLASSIFIERS, self).__init__()

        hidden_dims = [2048, 1024, 512, 256, 128]

        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.target_layer = self.encoder[7][-1]

        self.fc_mu = nn.Linear(hidden_dims[0], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[0], latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[1], 4, 2, 1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dims[1], hidden_dims[2], 4, 2, 1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dims[2], hidden_dims[3], 4, 2, 1),
            nn.BatchNorm2d(hidden_dims[3]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dims[3], hidden_dims[4], 4, 2, 1),
            nn.BatchNorm2d(hidden_dims[4]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dims[4], 3, 4, 2, 1),
            nn.Sigmoid(),
        )

        half_dim = latent_dim // 2

        self.classifier1 = nn.Sequential(
            nn.Linear(half_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2),
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(half_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2),
        )

        self.features = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_features)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        self.features = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def calculate_gradcam(self, pred, label):
        was_training = self.training
        self.eval()

        B = pred.size(0)
        scores = pred[torch.arange(B), label.long()]

        gradients = torch.autograd.grad(
            scores.sum(), self.features, create_graph=True, retain_graph=True
        )[0]

        features = self.features
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1)
        cam = F.relu(cam)

        cams = []
        for i in range(B):
            cam_i = cam[i]
            c_min = cam_i.min()
            c_max = cam_i.max()
            diff = c_max - c_min
            if diff < 1e-8:
                cam_i = torch.zeros_like(cam_i)
            else:
                cam_i = (cam_i - c_min) / diff
            cams.append(cam_i.unsqueeze(0))

        batch_cams = torch.cat(cams, dim=0)

        if was_training:
            self.train()

        return batch_cams

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * torch.clamp(log_var, min=-20, max=20))
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 2048, 7, 7)
        x = self.decoder(x)
        return x

    def forward(self, x, use_mean=False):
        mu, log_var = self.encode(x)
        z = mu if use_mean else self.reparameterize(mu, log_var)
        recon = self.decode(z)

        half_dim = mu.size(1) // 2
        mu1 = mu[:, :half_dim]
        mu2 = mu[:, half_dim:]

        pred1 = self.classifier1(mu1)
        pred2 = self.classifier2(mu2)

        return recon, mu, log_var, pred1, pred2


def gradcam_similarity_loss(model_cam, clip_cam):
    return F.mse_loss(model_cam, clip_cam)


def gradcam_dissimilarity_loss(model_cam, clip_cam):
    return F.mse_loss(model_cam, 1.0 - clip_cam)


def loss_function(
    model,
    recon_x,
    x,
    mu,
    log_var,
    pred1,
    pred2,
    labels,
    clip_cam,
    beta,
    cls_weight,
    bce_weight,
    gcam_weight,
    l2_weight,
):
    bce = F.mse_loss(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    cls1 = F.cross_entropy(pred1, labels.long())
    cls2 = F.cross_entropy(pred2, labels.long())

    model_cams1 = model.calculate_gradcam(pred1, labels)
    model_cams2 = model.calculate_gradcam(pred2, labels)

    gcam1 = gradcam_similarity_loss(model_cams1, clip_cam)
    gcam2 = gradcam_dissimilarity_loss(model_cams2, clip_cam)

    l2_reg = 0
    weight_count = 0
    for name, param in model.classifier1.named_parameters():
        if "weight" in name:
            l2_reg += torch.sum(param.pow(2))
            weight_count += param.numel()
    if weight_count > 0:
        l2_reg = l2_reg / weight_count

    total_loss = (
        bce_weight * bce
        + beta * kld
        + cls_weight * (cls1 + cls2)
        + gcam_weight * (gcam1 + gcam2)
        + l2_weight * l2_reg
    )

    return total_loss


def count_groups(df):
    counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for _, row in df.iterrows():
        y = int(row["y"])
        place = int(row["place"])
        counts[(y, place)] += 1
    return counts


def format_group_counts_line(prefix, counts):
    parts = []
    for y in [0, 1]:
        for place in [0, 1]:
            parts.append(f"Group ({y}, {place}): {counts[(y, place)]}")
    return f"{prefix}: " + "; ".join(parts)


def evaluate_groups(model, dataloader, device):
    model.eval()
    group_correct = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    group_total = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    total_correct = 0
    total = 0

    full_df = dataloader.dataset.df.reset_index(drop=True)
    idx_offset = 0

    with torch.no_grad():
        for batch in dataloader:
            data, labels = batch[:2]
            bsz = data.size(0)
            data = data.to(device)
            labels = labels.to(device)

            _, _, _, pred1, _ = model(data, use_mean=True)
            pred_labels = pred1.argmax(dim=1)

            total_correct += (pred_labels == labels.long()).sum().item()
            total += labels.size(0)

            for i in range(bsz):
                row_idx = idx_offset + i
                y = int(full_df.loc[row_idx, "y"])
                place = int(full_df.loc[row_idx, "place"])
                key = (y, place)
                group_total[key] += 1
                group_correct[key] += int(
                    (pred_labels[i].item() == int(labels[i].item()))
                )
            idx_offset += bsz

    group_acc = {}
    for key in group_total:
        if group_total[key] > 0:
            group_acc[key] = 100.0 * group_correct[key] / group_total[key]
        else:
            group_acc[key] = 0.0

    worst = min(group_acc.values())
    avg = 100.0 * total_correct / total if total > 0 else 0.0
    return group_acc, worst, avg


def format_group_acc_line(prefix, group_acc):
    parts = []
    for y in [0, 1]:
        for place in [0, 1]:
            parts.append(f"Group ({y}, {place}): {group_acc[(y, place)]:.2f}%")
    return f"{prefix}: " + "; ".join(parts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys())
    )
    return parser.parse_args()


def _infer_wb_version(dataset_key: str) -> str:
    parts = dataset_key.split("_", 1)
    if len(parts) == 2 and parts[1]:
        return parts[1]
    return "unknown"


def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    prompt = ["a bird"]
    beta = 1.0
    cls_weight = 1.0
    bce_weight = 1.0
    gcam_weight = 40.0
    l2_weight = 1000.0
    weight_decay = 1e-4
    seed = 42
    lr = 1e-5
    batch_size = 32
    epochs = 30

    wb_version = _infer_wb_version(args.dataset)
    os.environ["WB_VERSION"] = wb_version

    os.makedirs("./outputs", exist_ok=True)
    log_path = f"./outputs/waterbirds_{wb_version}.log"
    model_save_path = f"./outputs/best_waterbirds_{wb_version}.pth"

    original_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8") as log_f:
        sys.stdout = LogWriter(log_f)
        sys.stderr = LogWriter(log_f)
        tee = TeeWriter(original_stdout, log_f)

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

        data_dir = cfg["data_dir"]
        csv_file = os.path.join(data_dir, "metadata.csv")
        df = pd.read_csv(csv_file)

        df_train = df[df["split"] == 0].copy()
        df_valid = df[df["split"] == 1].copy()
        df_test = df[df["split"] == 2].copy()

        train_counts = count_groups(df_train)
        valid_counts = count_groups(df_valid)
        test_counts = count_groups(df_test)

        print(format_group_counts_line("Train", train_counts), file=tee, flush=True)
        print(format_group_counts_line("Val", valid_counts), file=tee, flush=True)
        print(format_group_counts_line("Test", test_counts), file=tee, flush=True)

        transform = create_original_transform("resnet50")

        print("Creating datasets with GradCAM", file=tee, flush=True)
        train_dataset = WaterbirdDataset(
            df_train,
            data_dir,
            transform=transform,
            calculate_gradcam=True,
            prompts=prompt,
            tqdm_file=tee,
        )
        valid_dataset = WaterbirdDataset(
            df_valid,
            data_dir,
            transform=transform,
            calculate_gradcam=False,
            prompts=prompt,
            tqdm_file=tee,
        )
        test_dataset = WaterbirdDataset(
            df_test,
            data_dir,
            transform=transform,
            calculate_gradcam=False,
            prompts=prompt,
            tqdm_file=tee,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = VAE_TWOCLASSIFIERS().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_worst = 0.0

        for epoch in range(epochs):
            model.train()

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", file=tee, leave=True)
            for batch in loop:
                data, labels, clip_cams = batch
                data = data.to(device)
                labels = labels.to(device)
                clip_cams = clip_cams.to(device)

                optimizer.zero_grad()
                recon_batch, mu, log_var, pred1, pred2 = model(data, use_mean=False)

                loss = loss_function(
                    model,
                    recon_batch,
                    data,
                    mu,
                    log_var,
                    pred1,
                    pred2,
                    labels,
                    clip_cams,
                    beta=beta,
                    cls_weight=cls_weight,
                    bce_weight=bce_weight,
                    gcam_weight=gcam_weight,
                    l2_weight=l2_weight,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            _, worst_acc, avg_acc = evaluate_groups(model, valid_loader, device)

            saved = False
            if worst_acc > best_worst:
                best_worst = worst_acc
                torch.save(model.state_dict(), model_save_path)
                saved = True

            line = (
                f"[Epoch {epoch+1}] Worst group accuracy: {worst_acc:.2f}% "
                f"Average accuracy: {avg_acc:.2f}%"
            )
            if saved:
                line += " (Model saved)"
            print(line, file=tee, flush=True)

        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        test_group_acc, test_worst, test_avg = evaluate_groups(model, test_loader, device)
        print(
            f"[Test] Worst group accuracy: {test_worst:.2f}% Average accuracy: {test_avg:.2f}%",
            file=tee,
            flush=True,
        )
        print(
            format_group_acc_line("[Test] Group", test_group_acc).replace(
                "[Test] Group:", "[Test] "
            ),
            file=tee,
            flush=True,
        )


if __name__ == "__main__":
    main()