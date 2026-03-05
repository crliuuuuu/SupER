import argparse
import os
import pickle
import random
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import clip
import numpy as np
import timm
import timm.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from spawrious.torch import get_spawrious_dataset
from spawrious.torch import set_model_name
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

DATASET_CONFIGS = {
    "o2o_easy": {
        "name": "O2O-Easy",
        "dataset_type": "o2o_easy",
        "beta": 10.0,
        "gcam_weight": 10.0,
    },
    "o2o_medium": {
        "name": "O2O-Medium",
        "dataset_type": "o2o_medium",
        "beta": 1.0,
        "gcam_weight": 80.0,
    },
    "o2o_hard": {
        "name": "O2O-Hard",
        "dataset_type": "o2o_hard",
        "beta": 1.0,
        "gcam_weight": 80.0,
    },
    "m2m_easy": {
        "name": "M2M-Easy",
        "dataset_type": "m2m_easy",
        "beta": 10.0,
        "gcam_weight": 50.0,
    },
    "m2m_medium": {
        "name": "M2M-Medium",
        "dataset_type": "m2m_medium",
        "beta": 1.0,
        "gcam_weight": 50.0,
    },
    "m2m_hard": {
        "name": "M2M-Hard",
        "dataset_type": "m2m_hard",
        "beta": 1.0,
        "gcam_weight": 50.0,
    },
}


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


class NullWriter:
    def write(self, s):
        return

    def flush(self):
        return


def _quantile_normalize(cam_i, p_low=1, p_high=99):
    vmin, vmax = np.percentile(cam_i, [p_low, p_high])
    if vmax - vmin < 1e-8:
        return np.zeros_like(cam_i, dtype=np.float32)
    cam_i = np.clip(cam_i, vmin, vmax)
    return (cam_i - vmin) / (vmax - vmin)


class CLIPGradCAM:
    def __init__(self, model_name="RN50"):
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

    def calculate_gradcam(self, image, prompts):
        self.model.zero_grad()
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(prompts).to(self.device)
        accum_cam = None

        for tokens in text_tokens:
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(tokens.unsqueeze(0))

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            sim = 100.0 * image_features @ text_features.T
            score = sim.squeeze(1)

            gradients = torch.autograd.grad(
                score,
                self._features,
                create_graph=False,
                retain_graph=True,
            )[0]

            alpha = gradients.mean(dim=(2, 3), keepdim=True)
            weighted = (alpha * self._features).sum(dim=1)
            cam = F.relu(weighted)
            cam_i = _quantile_normalize(
                cam[0].detach().cpu().numpy(), p_low=1, p_high=100
            )

            if accum_cam is None:
                accum_cam = cam_i
            else:
                accum_cam += cam_i

        return (accum_cam / len(prompts)).astype(np.float32)

    def remove_hooks(self):
        self.fh.remove()
        self.bh.remove()


def get_dataset_combinations(dataset_type):
    combinations_config = {
        "o2o_easy": (
            ["desert", "jungle", "dirt", "snow"],
            ["dirt", "snow", "desert", "jungle"],
            "beach",
        ),
        "o2o_medium": (
            ["mountain", "beach", "dirt", "jungle"],
            ["jungle", "dirt", "beach", "snow"],
            "desert",
        ),
        "o2o_hard": (
            ["jungle", "mountain", "snow", "desert"],
            ["mountain", "snow", "desert", "jungle"],
            "beach",
        ),
        "m2m_hard": (
            ["dirt", "jungle", "snow", "beach"],
            ["snow", "beach", "dirt", "jungle"],
            None,
        ),
        "m2m_easy": (
            ["desert", "mountain", "dirt", "jungle"],
            ["dirt", "jungle", "mountain", "desert"],
            None,
        ),
        "m2m_medium": (
            ["beach", "snow", "mountain", "desert"],
            ["desert", "mountain", "beach", "snow"],
            None,
        ),
    }

    if dataset_type not in combinations_config:
        raise ValueError(dataset_type)

    group, test, filler = combinations_config[dataset_type]
    total = 3168

    combinations = {}
    if "m2m" in dataset_type:
        counts = [total, total]
        combinations["train_combinations"] = {
            ("bulldog",): [(group[0], counts[0]), (group[1], counts[1])],
            ("dachshund",): [(group[1], counts[0]), (group[0], counts[1])],
            ("labrador",): [(group[2], counts[0]), (group[3], counts[1])],
            ("corgi",): [(group[3], counts[0]), (group[2], counts[1])],
        }
        combinations["test_combinations"] = {
            ("bulldog",): [test[0], test[1]],
            ("dachshund",): [test[1], test[0]],
            ("labrador",): [test[2], test[3]],
            ("corgi",): [test[3], test[2]],
        }
    else:
        counts = [int(0.97 * total), int(0.87 * total)]
        combinations["train_combinations"] = {
            ("bulldog",): [(group[0], counts[0]), (group[0], counts[1])],
            ("dachshund",): [(group[1], counts[0]), (group[1], counts[1])],
            ("labrador",): [(group[2], counts[0]), (group[2], counts[1])],
            ("corgi",): [(group[3], counts[0]), (group[3], counts[1])],
            ("bulldog", "dachshund", "labrador", "corgi"): [
                (filler, total - counts[0]),
                (filler, total - counts[1]),
            ],
        }
        combinations["test_combinations"] = {
            ("bulldog",): [test[0], test[0]],
            ("dachshund",): [test[1], test[1]],
            ("labrador",): [test[2], test[2]],
            ("corgi",): [test[3], test[3]],
        }

    return combinations


def get_image_paths(root_dir, dataset_type):
    base_dir = os.path.join(root_dir, "spawrious224")
    if not os.path.exists(base_dir):
        raise ValueError(base_dir)

    class_list = ["bulldog", "corgi", "dachshund", "labrador"]
    locations_list = ["desert", "jungle", "dirt", "mountain", "snow", "beach"]

    is_o2o = dataset_type.startswith("o2o")
    _ = get_dataset_combinations(dataset_type)

    image_paths_map = {}
    for split_id in ["0", "1"] if is_o2o else ["0"]:
        for class_idx, class_name in enumerate(class_list):
            for location_idx, location_name in enumerate(locations_list):
                path = os.path.join(base_dir, split_id, location_name, class_name)
                if not os.path.exists(path):
                    continue
                image_files = [
                    f
                    for f in os.listdir(path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if not image_files:
                    continue
                key = (class_idx, location_idx)
                if key not in image_paths_map:
                    image_paths_map[key] = []
                image_paths = set([os.path.join(path, f) for f in image_files])
                image_paths_map[key].extend(list(image_paths))

    for key in image_paths_map:
        image_paths_map[key].sort()

    return image_paths_map


def create_original_transform(model_name):
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
    data_config = timm.data.resolve_model_data_config(backbone)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return transform


class SpawriousImageDataset(Dataset):
    def __init__(
        self,
        spawrious_dataset,
        data_dir,
        dataset_type,
        transform=None,
        calculate_gradcam=True,
        prompts=None,
        tqdm_file=None,
    ):
        self.dataset = spawrious_dataset
        self.transform = transform
        self.calculate_gradcam = calculate_gradcam
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.prompts = prompts if prompts is not None else ["a dog"]
        self.image_paths_map = get_image_paths(data_dir, dataset_type)
        if not self.image_paths_map:
            raise ValueError("No images found")

        self.df = []
        self.env_counts = {}

        for i in range(len(spawrious_dataset)):
            tensor_img, label, location = spawrious_dataset[i]
            label_int = int(label)
            location_int = int(location)
            key = (label_int, location_int)
            if key not in self.image_paths_map:
                continue

            env_paths = self.image_paths_map[key]
            if key not in self.env_counts:
                self.env_counts[key] = 0

            img_idx = self.env_counts[key]
            if img_idx >= len(env_paths):
                img_idx = img_idx % len(env_paths)

            img_path = env_paths[img_idx]
            self.env_counts[key] += 1

            self.df.append(
                {
                    "index": i,
                    "y": label_int,
                    "place": location_int,
                    "img_path": img_path,
                }
            )

        if self.calculate_gradcam:
            self.clip_gradcam = CLIPGradCAM()
            self.gradcam_maps = {}
            total_images = len(self.df)
            with tqdm(
                total=total_images,
                desc="Calculating GradCAM",
                unit="img",
                file=tqdm_file,
            ) as pbar:
                for idx in range(total_images):
                    img_path = self.df[idx]["img_path"]
                    try:
                        img = Image.open(img_path).convert("RGB")
                        cam = self.clip_gradcam.calculate_gradcam(img, self.prompts)
                        self.gradcam_maps[idx] = cam
                    except Exception:
                        self.gradcam_maps[idx] = np.zeros((7, 7), dtype=np.float32)
                    pbar.update(1)
            self.clip_gradcam.remove_hooks()
            del self.clip_gradcam

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df[idx]
        img_path = item["img_path"]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image_tensor = self.transform(image)
            else:
                default_transform = create_original_transform("resnet50")
                image_tensor = default_transform(image)
        except Exception:
            image_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

        label = torch.tensor(item["y"], dtype=torch.float32)

        if self.calculate_gradcam:
            gradcam = torch.tensor(self.gradcam_maps[idx], dtype=torch.float32)
            return image_tensor, label, gradcam
        return image_tensor, label


class WrappedSubset(Dataset):
    def __init__(self, subset, original_dataset):
        self.subset = subset
        self.original = original_dataset
        self.df = [original_dataset.df[i] for i in subset.indices]
        self.gradcam_maps = {
            i: original_dataset.gradcam_maps[subset.indices[i]]
            for i in range(len(subset))
        }
        self.calculate_gradcam = original_dataset.calculate_gradcam

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.original[self.subset.indices[idx]]


def split_test_to_validation(
    test_dataset, val_split=0.1, num_classes=4, num_locations=6, seed=42
):
    random.seed(seed)

    env_indices = {}
    for idx, item in enumerate(test_dataset.df):
        y = int(item["y"])
        place = int(item["place"])
        env_id = y * num_locations + place
        if env_id not in env_indices:
            env_indices[env_id] = []
        env_indices[env_id].append(idx)

    valid_indices = []
    test_indices = []

    for env_id, indices in env_indices.items():
        random.shuffle(indices)
        val_size = int(len(indices) * val_split)
        valid_indices.extend(indices[:val_size])
        test_indices.extend(indices[val_size:])

    return valid_indices, test_indices


def prepare_spawrious_dataset(
    dataset_type, data_dir, prompts, tqdm_file, val_split=0.1, save_path=None
):
    MODEL_NAME = "resnet50"
    set_model_name(MODEL_NAME)
    spawrious = get_spawrious_dataset(dataset_name=dataset_type, root_dir=data_dir)

    train_set = spawrious.get_train_dataset()
    test_set = spawrious.get_test_dataset()

    transform = create_original_transform(MODEL_NAME)

    train_dataset = SpawriousImageDataset(
        train_set,
        data_dir=data_dir,
        dataset_type=dataset_type,
        transform=transform,
        calculate_gradcam=True,
        prompts=prompts,
        tqdm_file=tqdm_file,
    )

    test_dataset = SpawriousImageDataset(
        test_set,
        data_dir=data_dir,
        dataset_type=dataset_type,
        transform=transform,
        calculate_gradcam=True,
        prompts=prompts,
        tqdm_file=tqdm_file,
    )

    valid_indices, test_indices = split_test_to_validation(
        test_dataset, val_split=val_split
    )

    valid_subset = Subset(test_dataset, valid_indices)
    test_subset = Subset(test_dataset, test_indices)

    valid_dataset = WrappedSubset(valid_subset, test_dataset)
    test_dataset_new = WrappedSubset(test_subset, test_dataset)

    def save_dataset_info(dataset):
        df_copy = dataset.df.copy() if hasattr(dataset.df, "copy") else dataset.df[:]
        dataset_copy = {"df": df_copy, "gradcam_maps": dataset.gradcam_maps}
        return dataset_copy

    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "train_dataset": save_dataset_info(train_dataset),
                "valid_dataset": save_dataset_info(valid_dataset),
                "test_dataset": save_dataset_info(test_dataset_new),
            },
            f,
        )

    return train_dataset, valid_dataset, test_dataset_new


class SpawriousImageDatasetCached(Dataset):
    def __init__(self, df=None, gradcam_maps=None, transform=None, calculate_gradcam=True):
        self.df = df if df is not None else []
        self.gradcam_maps = gradcam_maps if gradcam_maps is not None else {}
        self.transform = transform
        self.calculate_gradcam = calculate_gradcam

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df[idx]
        img_path = item["img_path"]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image_tensor = self.transform(image)
            else:
                default_transform = create_original_transform("resnet50")
                image_tensor = default_transform(image)
        except Exception:
            image_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

        label = torch.tensor(item["y"], dtype=torch.float32)

        if self.calculate_gradcam and idx in self.gradcam_maps:
            gradcam = torch.tensor(self.gradcam_maps[idx], dtype=torch.float32)
            return image_tensor, label, gradcam

        if self.calculate_gradcam:
            empty_gradcam = torch.zeros(7, 7, dtype=torch.float32)
            return image_tensor, label, empty_gradcam

        return image_tensor, label


class VAE_TWOCLASSIFIERS(nn.Module):
    def __init__(self, latent_dim=512, hidden_dims=None, num_classes=4):
        super(VAE_TWOCLASSIFIERS, self).__init__()

        if hidden_dims is None:
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
            nn.Linear(128, num_classes),
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
            nn.Linear(128, num_classes),
        )

        self.features = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_features)
        self.target_layer.register_full_backward_hook(self._save_gradients)

        self.hidden_dims = hidden_dims

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
        x = x.view(x.size(0), self.hidden_dims[0], 7, 7)
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


def gradcam_similarity_loss(model_cam, clip_cam, use_threshold=False, threshold=1e-6):
    if not use_threshold:
        return F.mse_loss(model_cam, clip_cam)
    mask = clip_cam < threshold
    masked_model_cam = model_cam[mask]
    if masked_model_cam.numel() == 0:
        return torch.tensor(0.0, device=clip_cam.device)
    return F.mse_loss(masked_model_cam, torch.zeros_like(masked_model_cam))


def gradcam_dissimilarity_loss(model_cam, clip_cam, use_threshold=False, threshold=1e-6):
    if not use_threshold:
        return F.mse_loss(model_cam, 1.0 - clip_cam)
    mask = clip_cam > threshold
    masked_model_cam = model_cam[mask]
    if masked_model_cam.numel() == 0:
        return torch.tensor(0.0, device=clip_cam.device)
    return F.mse_loss(masked_model_cam, torch.zeros_like(masked_model_cam))


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
    beta=1.0,
    cls_weight=0.1,
    bce_weight=1.0,
    gcam_weight=10.0,
    l1_weight=0.0,
    l2_weight=0.0,
    use_threshold=False,
    gcam_threshold=1e-6,
):
    BCE = F.mse_loss(recon_x, x, reduction="mean")
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    CLS1 = F.cross_entropy(pred1, labels.long())
    CLS2 = F.cross_entropy(pred2, labels.long())

    model_cams1 = model.calculate_gradcam(pred1, labels)
    model_cams2 = model.calculate_gradcam(pred2, labels)

    GCAM1 = gradcam_similarity_loss(
        model_cams1, clip_cam, use_threshold=use_threshold, threshold=gcam_threshold
    )
    GCAM2 = gradcam_dissimilarity_loss(
        model_cams2, clip_cam, use_threshold=use_threshold, threshold=gcam_threshold
    )

    L1_REG = 0
    weight_count = 0
    for name, param in model.classifier1.named_parameters():
        if "weight" in name:
            L1_REG += torch.sum(torch.abs(param))
            weight_count += param.numel()
    if weight_count > 0:
        L1_REG = L1_REG / weight_count

    L2_REG = 0
    weight_count = 0
    for name, param in model.classifier1.named_parameters():
        if "weight" in name:
            L2_REG += torch.sum(param.pow(2))
            weight_count += param.numel()
    if weight_count > 0:
        L2_REG = L2_REG / weight_count

    total_loss = (
        bce_weight * BCE
        + beta * KLD
        + cls_weight * (CLS1 + CLS2)
        + gcam_weight * (GCAM1 + GCAM2)
        + l1_weight * L1_REG
        + l2_weight * L2_REG
    )

    return total_loss


def evaluate_environments(model, dataloader, device, num_classes=4, num_locations=6):
    model.eval()
    env_correct = {}
    env_total = {}

    for cls in range(num_classes):
        for loc in range(num_locations):
            env_id = cls * num_locations + loc
            env_correct[env_id] = 0
            env_total[env_id] = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            data, labels, clip_cams = batch
            data, labels = data.to(device), labels.to(device)

            _, _, _, pred1, _ = model(data, use_mean=True)
            pred_labels = pred1.argmax(dim=1)

            for i in range(labels.size(0)):
                label = int(labels[i].item())
                idx = i + batch_idx * dataloader.batch_size
                if idx < len(dataloader.dataset.df):
                    place = int(dataloader.dataset.df[idx]["place"])
                    env_id = label * num_locations + place
                    if env_id in env_correct:
                        env_correct[env_id] += (pred_labels[i] == labels[i]).item()
                        env_total[env_id] += 1

    env_accuracies = {}
    for env_id in env_correct.keys():
        if env_total[env_id] > 0:
            env_accuracies[env_id] = 100.0 * env_correct[env_id] / env_total[env_id]
        else:
            env_accuracies[env_id] = 0.0

    return env_accuracies


def worst_and_average_from_env_accuracies(env_accuracies):
    populated_envs = [env for env, acc in env_accuracies.items() if acc > 0]
    accs = [acc for env, acc in env_accuracies.items() if env in populated_envs]
    worst = min(accs) if accs else 0.0
    avg = (sum(accs) / len(accs)) if accs else 0.0
    return worst, avg


def count_groups_from_df_list(df_list, num_classes=4, num_locations=6):
    counts = {}
    for y in range(num_classes):
        for place in range(num_locations):
            counts[(y, place)] = 0
    for item in df_list:
        y = int(item["y"])
        place = int(item["place"])
        counts[(y, place)] += 1
    return counts


def format_group_counts_line(prefix, counts, num_classes=4, num_locations=6):
    parts = []
    for y in range(num_classes):
        for place in range(num_locations):
            parts.append(f"Group ({y}, {place}): {counts[(y, place)]}")
    return f"{prefix}: " + "; ".join(parts)


def format_group_acc_line(prefix, env_acc, num_classes=4, num_locations=6):
    parts = []
    for y in range(num_classes):
        for place in range(num_locations):
            env_id = y * num_locations + place
            parts.append(f"Group ({y}, {place}): {env_acc[env_id]:.2f}%")
    return f"{prefix} " + "; ".join(parts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys())
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    dataset_type = cfg["dataset_type"]

    data_dir = "./data/spawrious"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    prompt = ["a dog"]
    num_classes = 4
    num_locations = 6
    cls_weight = 1.0
    bce_weight = 1.0
    l2_weight = 100.0
    weight_decay = 1e-4
    lr = 1e-6
    batch_size = 32
    epochs = 30
    val_split = 0.1
    seed = 42
    beta = cfg["beta"]
    gcam_weight = cfg["gcam_weight"]

    log_path = os.path.join(output_dir, f"spawrious_{dataset_type}.log")
    model_save_path = os.path.join(output_dir, f"best_{dataset_type}.pth")
    cache_path = os.path.join(output_dir, f"{dataset_type}_processed_data.pkl")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_f:
        tee = TeeWriter(original_stdout, log_f)
        null = NullWriter()
        sys.stdout = null
        sys.stderr = null

        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(seed)

        print("Creating datasets with GradCAM", file=tee, flush=True)
        prepare_spawrious_dataset(
            dataset_type=dataset_type,
            data_dir=data_dir,
            prompts=prompt,
            tqdm_file=tee,
            val_split=val_split,
            save_path=cache_path,
        )

        with open(cache_path, "rb") as f:
            data_dict = pickle.load(f)

        transform = create_original_transform("resnet50")

        train_dataset = SpawriousImageDatasetCached(
            df=data_dict["train_dataset"]["df"],
            gradcam_maps=data_dict["train_dataset"]["gradcam_maps"],
            transform=transform,
            calculate_gradcam=True,
        )

        valid_dataset = SpawriousImageDatasetCached(
            df=data_dict["valid_dataset"]["df"],
            gradcam_maps=data_dict["valid_dataset"]["gradcam_maps"],
            transform=transform,
            calculate_gradcam=True,
        )

        test_dataset = SpawriousImageDatasetCached(
            df=data_dict["test_dataset"]["df"],
            gradcam_maps=data_dict["test_dataset"]["gradcam_maps"],
            transform=transform,
            calculate_gradcam=True,
        )

        train_counts = count_groups_from_df_list(
            train_dataset.df, num_classes=num_classes, num_locations=num_locations
        )
        valid_counts = count_groups_from_df_list(
            valid_dataset.df, num_classes=num_classes, num_locations=num_locations
        )
        test_counts = count_groups_from_df_list(
            test_dataset.df, num_classes=num_classes, num_locations=num_locations
        )

        print(
            format_group_counts_line(
                "Train",
                train_counts,
                num_classes=num_classes,
                num_locations=num_locations,
            ),
            file=tee,
            flush=True,
        )
        print(
            format_group_counts_line(
                "Val",
                valid_counts,
                num_classes=num_classes,
                num_locations=num_locations,
            ),
            file=tee,
            flush=True,
        )
        print(
            format_group_counts_line(
                "Test",
                test_counts,
                num_classes=num_classes,
                num_locations=num_locations,
            ),
            file=tee,
            flush=True,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = VAE_TWOCLASSIFIERS(latent_dim=512, num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_worst_env_acc = 0.0

        for epoch in range(epochs):
            model.train()

            loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}]", leave=True, file=tee)
            for batch in loop:
                try:
                    if len(batch) == 3:
                        data, labels, clip_cams = batch
                    else:
                        data, labels = batch
                        clip_cams = torch.rand(data.size(0), 7, 7)

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
                        l1_weight=0.0,
                        l2_weight=l2_weight,
                        use_threshold=False,
                        gcam_threshold=1e-6,
                    )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                except Exception:
                    continue

            valid_env_accuracies = evaluate_environments(
                model,
                valid_loader,
                device,
                num_classes=num_classes,
                num_locations=num_locations,
            )
            worst_env_acc, average_env_acc = worst_and_average_from_env_accuracies(
                valid_env_accuracies
            )

            saved = False
            if worst_env_acc > best_worst_env_acc:
                best_worst_env_acc = worst_env_acc
                torch.save(model.state_dict(), model_save_path)
                saved = True

            line = (
                f"[Epoch {epoch+1}] Worst group accuracy: {worst_env_acc:.2f}% "
                f"Average accuracy: {average_env_acc:.2f}%"
            )
            if saved:
                line += " (Model saved)"
            print(line, file=tee, flush=True)

        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        test_env_accuracies = evaluate_environments(
            model,
            test_loader,
            device,
            num_classes=num_classes,
            num_locations=num_locations,
        )
        test_worst, test_avg = worst_and_average_from_env_accuracies(test_env_accuracies)

        print(
            f"[Test] Worst group accuracy: {test_worst:.2f}% Average accuracy: {test_avg:.2f}%",
            file=tee,
            flush=True,
        )
        print(
            format_group_acc_line(
                "[Test]",
                test_env_accuracies,
                num_classes=num_classes,
                num_locations=num_locations,
            ),
            file=tee,
            flush=True,
        )

        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception:
                pass

        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == "__main__":
    main()