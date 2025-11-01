import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import faiss
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import pickle
from tqdm import tqdm
from collections import defaultdict
from statsmodels.distributions.empirical_distribution import ECDF



def extract_descriptor(image_path, model, preprocess, device):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        descriptor = model(image).flatten()
     #   descriptor = torch.nn.functional.normalize(descriptor, p=2, dim=0)
    return descriptor.cpu().numpy()


def analyze_faiss_distances():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=False)
    model.fc = nn.Identity()
    model.load_state_dict(torch.load('models/SMALL_HOUSE_FINAL_MODEL.pth', map_location=device))
    model = model.to(device)
    model.eval()


    with open("SMALL_HOUSE_FINAL_MEAN.pkl", "rb") as f:
        norm_values = pickle.load(f)
    mean = norm_values["mean"]
    std = norm_values["std"]

    # Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Load validation data embeddings
    base_path = "small_house_FINAL_split_data/val/"
    image_paths = []
    labels = []

    class_to_idx = {}
    idx = 0

    for class_name in sorted(os.listdir(base_path)):
        class_folder = os.path.join(base_path, class_name)
        if os.path.isdir(class_folder):
            class_to_idx[class_name] = idx
            for img_file in sorted(os.listdir(class_folder)):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    image_paths.append(os.path.join(class_folder, img_file))
                    labels.append(class_name)
            idx += 1


    descriptors = []
    for path in tqdm(image_paths, desc="Extracting embeddings"):
        desc = extract_descriptor(path, model, preprocess, device)
        descriptors.append(desc)
    descriptors = np.vstack(descriptors)


    d = descriptors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(descriptors)


    positive_dists = []
    negative_dists = []

    for i in tqdm(range(len(descriptors)), desc="Comparing distances"):
        query = descriptors[i:i+1]
        label = labels[i]

        D, I = index.search(query, k=len(descriptors))
        for dist, j in zip(D[0][1:], I[0][1:]):  # Skip self-match
            if labels[j] == label:
                positive_dists.append(dist)
            else:
                negative_dists.append(dist)

    # Plot histogram (positives)
    plt.figure(figsize=(6, 4))
    plt.hist(positive_dists, bins=200, alpha=0.8, label='Positive pairs')
    plt.xlabel('Euclidean Distance', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distance Distribution of Positive Pairs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
   # plt.show()

    pos_arr = np.array(positive_dists, dtype=np.float32)
    ecdf_obj = ECDF(pos_arr)
    pos_sorted = np.sort(pos_arr)
    ecdf_vals = ecdf_obj(pos_sorted)

    # Extend both curves on both sides
    x_min = float(pos_sorted[0])
    x_max = float(pos_sorted[-1])
    x_pad_left  = x_min - 0.05 * (x_max - x_min)   # small extension left
    x_pad_right = x_max + 0.05 * (x_max - x_min)  # small extension right

    # Extended arrays
    pos_sorted_ext = np.concatenate(([x_pad_left], pos_sorted, [x_pad_right]))
    pos_sorted_ext2 = np.concatenate(([0], pos_sorted, [x_pad_right]))
    ecdf_vals_ext  = np.concatenate(([0.0], ecdf_vals, [1.0]))
    conf_vals_ext  = np.concatenate(([100.0], (1.0 - ecdf_vals) * 100.0, [0.0]))

    # ---- Plot CDF ----
    plt.figure(figsize=(6, 4))
    plt.step(pos_sorted_ext, ecdf_vals_ext, where='post')
    plt.xlabel('Euclidean Distance', fontsize=14)
    plt.ylabel('F(d) = P(D \u2264 d)', fontsize=14)
    plt.title('CDF of Positive Pair Distances')
    plt.grid(True)
    plt.xlim(x_pad_left, x_pad_right)
    plt.ylim(-0.05, 1.05)  # keep CDF in [0,1]
    plt.tight_layout()

    # ---- Plot Confidence ----
    plt.figure(figsize=(6, 4))
    plt.step(pos_sorted_ext2, conf_vals_ext, where='post')
    plt.xlabel('Euclidean Distance', fontsize=14)
    plt.ylabel('Confidence (%)', fontsize=14)
    plt.title('Confidence from Positive Pair Distances')
    plt.grid(True)
    plt.xlim(x_pad_left, x_pad_right)
    plt.ylim(-5, 105)  # keep confidence in [0,100]
    plt.tight_layout()
            
    


    # Example: compute distance -> confidence using ECDF (Conf = 1 - F(d))
    probe_ds = [0.0317056]
    for probe in probe_ds:
        Fd = ecdf_obj(probe)
        conf = 1.0 - Fd                   # smaller d => higher confidence
        print(f"d = {probe:.4f}  ->  F(d) = {Fd:.3f},  Conf = {100*conf:.1f}%")


    xs_conf = pos_sorted_ext2.astype(np.float32)   # distances (monotonic non-decreasing)
    ys_conf = conf_vals_ext.astype(np.float32)     # confidence in percent [0..100]

    np.savez_compressed(
        "conf_curve_small_house_final_v1.npz",
        xs=xs_conf,
        ys=ys_conf,
        meta=np.string_("Conf(d) = (1 - ECDF(d)) * 100; val split; model=SMALL_HOUSE_FINAL_MODEL.pth")
    )

    # Option 2 (also fine): JSON (slightly larger, language-agnostic)
    # import json
    # with open("conf_curve_small_house_final_v1.json", "w") as f:
    #     json.dump({"xs": xs_conf.tolist(), "ys": ys_conf.tolist()}, f)

    print(f"Saved confidence LUT with {xs_conf.size} points.")
    plt.show()


output = analyze_faiss_distances()
