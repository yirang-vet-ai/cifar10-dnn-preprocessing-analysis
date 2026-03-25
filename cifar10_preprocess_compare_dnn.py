
# Copyright (c) 2026 Yirang Jung. All rights reserved.
# Licensed under the Apache License, Version 2.0
# 이 코드는 저작권자의 허가 없이 무단 전제 및 재배포를 금지합니다.
# 저작권자: 네이버 검색 "정이랑 수의사"

# ============================================================
# CIFAR-10 전처리 후보 자동 생성 + DNN 성능 비교 전체 코드
# 상대경로 기반 GitHub 업로드용 버전
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
import copy
import shutil
import random
import platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from pathlib import Path
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# 1. 환경 설정
# ------------------------------------------------------------
if platform.system() == "Windows":
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    matplotlib.rcParams["font.family"] = "AppleGothic"
else:
    matplotlib.rcParams["font.family"] = "NanumGothic"

matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

torch.backends.cudnn.benchmark = True

# ------------------------------------------------------------
# 2. 상대경로 기반 폴더 설정
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

SOURCE_ROOT = BASE_DIR / "data" / "cifar10_source_all"
OUTPUT_ROOT = BASE_DIR / "candidate_data"
RESULT_ROOT = BASE_DIR / "results"

EXPECTED_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

if not SOURCE_ROOT.exists():
    raise FileNotFoundError(
        f"입력 폴더가 없습니다: {SOURCE_ROOT}\n"
        f"다음 위치에 클래스 폴더를 넣어주세요:\n"
        f"{BASE_DIR / 'data' / 'cifar10_source_all'}"
    )

class_dirs = sorted([p for p in SOURCE_ROOT.iterdir() if p.is_dir()])
source_classes = [p.name for p in class_dirs]

print("프로젝트 기준 경로:", BASE_DIR)
print("입력 폴더:", SOURCE_ROOT)
print("전처리 출력 폴더:", OUTPUT_ROOT)
print("결과 저장 폴더:", RESULT_ROOT)
print("입력 클래스:", source_classes)

if source_classes != EXPECTED_CLASSES:
    print("[주의] 클래스 순서가 CIFAR-10 기본 순서와 다를 수 있습니다.")
    print("현재:", source_classes)
    print("기대:", EXPECTED_CLASSES)

# ------------------------------------------------------------
# 3. 전처리 함수들
# ------------------------------------------------------------
def preprocess_original(img_bgr):
    out = cv2.resize(img_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    return out

def preprocess_clahe_light(img_bgr):
    img = cv2.resize(img_bgr, (32, 32), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    l2 = clahe.apply(l)

    merged = cv2.merge((l2, a, b))
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return out

def preprocess_bilateral_clahe_light(img_bgr):
    img = cv2.resize(img_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    l2 = clahe.apply(l)

    merged = cv2.merge((l2, a, b))
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return out

def preprocess_grayscale_3ch(img_bgr):
    img = cv2.resize(img_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return out

def preprocess_edge_3ch(img_bgr):
    img = cv2.resize(img_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, threshold1=50, threshold2=150)
    out = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    return out

def preprocess_blur_edge_3ch(img_bgr):
    img = cv2.resize(img_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(blur, threshold1=40, threshold2=120)
    out = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    return out

PREPROCESSORS = {
    "original": preprocess_original,
    "clahe_light": preprocess_clahe_light,
    "bilateral_clahe_light": preprocess_bilateral_clahe_light,
    "grayscale_3ch": preprocess_grayscale_3ch,
    "edge_3ch": preprocess_edge_3ch,
    "blur_edge_3ch": preprocess_blur_edge_3ch,
}

# ------------------------------------------------------------
# 4. train/val/test 분할 파일 목록 생성
# ------------------------------------------------------------
def get_split_file_lists(class_dir):
    files = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    files = sorted(files)
    random.Random(SEED).shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files

# ------------------------------------------------------------
# 5. 후보 데이터셋 생성
# ------------------------------------------------------------
def build_candidate_datasets():
    for candidate_name, preprocess_fn in PREPROCESSORS.items():
        candidate_root = OUTPUT_ROOT / candidate_name

        if candidate_root.exists():
            shutil.rmtree(candidate_root)

        print(f"\n[후보 생성 시작] {candidate_name}")

        for split in ["train", "val", "test"]:
            (candidate_root / split).mkdir(parents=True, exist_ok=True)

        for class_dir in class_dirs:
            class_name = class_dir.name
            train_files, val_files, test_files = get_split_file_lists(class_dir)

            split_map = {
                "train": train_files,
                "val": val_files,
                "test": test_files,
            }

            for split_name, file_list in split_map.items():
                save_dir = candidate_root / split_name / class_name
                save_dir.mkdir(parents=True, exist_ok=True)

                for file_path in file_list:
                    img = cv2.imread(str(file_path))
                    if img is None:
                        print(f"[경고] 이미지 읽기 실패: {file_path}")
                        continue

                    processed = preprocess_fn(img)
                    save_path = save_dir / file_path.name
                    cv2.imwrite(str(save_path), processed)

            print(
                f"{class_name:>12} | "
                f"train {len(train_files):4d} | "
                f"val {len(val_files):4d} | "
                f"test {len(test_files):4d}"
            )

        print(f"[후보 생성 완료] {candidate_name} -> {candidate_root}")

# ------------------------------------------------------------
# 6. DNN 모델
# ------------------------------------------------------------
class ImprovedDNN_CIFAR10(nn.Module):
    def __init__(self, input_size=32 * 32 * 3, num_classes=10):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

# ------------------------------------------------------------
# 7. 학습 / 평가 함수
# ------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds

# ------------------------------------------------------------
# 8. 단일 후보 학습 함수
# ------------------------------------------------------------
def run_experiment(candidate_name):
    print(f"\n{'=' * 90}")
    print(f"[실험 시작] {candidate_name}")
    print(f"{'=' * 90}")

    candidate_root = OUTPUT_ROOT / candidate_name

    transform_common = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    trainset = torchvision.datasets.ImageFolder(
        root=str(candidate_root / "train"),
        transform=transform_common
    )
    valset = torchvision.datasets.ImageFolder(
        root=str(candidate_root / "val"),
        transform=transform_common
    )
    testset = torchvision.datasets.ImageFolder(
        root=str(candidate_root / "test"),
        transform=transform_common
    )

    use_cuda = torch.cuda.is_available()

    trainloader = DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda
    )
    valloader = DataLoader(
        valset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda
    )
    testloader = DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda
    )

    print("클래스:", trainset.classes)
    print(f"train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    model = ImprovedDNN_CIFAR10(
        input_size=32 * 32 * 3,
        num_classes=len(trainset.classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    N_EPOCHS = 30
    PATIENCE = 6

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_loss = float("inf")
    early_stop_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_time = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, valloader, criterion, device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        mark = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            mark = "  * best val"
        else:
            early_stop_counter += 1

        print(
            f"[{candidate_name}] "
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%{mark}"
        )

        if early_stop_counter >= PATIENCE:
            print(f"Early stopping: {PATIENCE}회 연속 val 개선 없음")
            break

    elapsed = time.time() - start_time

    model.load_state_dict(best_model_wts)

    test_loss, test_acc, _, _ = evaluate(model, testloader, criterion, device)

    save_model_path = RESULT_ROOT / f"best_{candidate_name}.pth"
    torch.save(model.state_dict(), save_model_path)

    plt.figure(figsize=(12, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], marker="s", label="Val Loss")
    plt.title(f"{candidate_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], marker="o", label="Train Acc")
    plt.plot(epochs_range, history["val_acc"], marker="s", label="Val Acc")
    plt.axhline(best_val_acc, linestyle="--", label=f"Best Val={best_val_acc:.2f}%")
    plt.title(f"{candidate_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULT_ROOT / f"{candidate_name}_curve.png", dpi=150)
    plt.close()

    result = {
        "candidate": candidate_name,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_test_acc": test_acc,
        "final_test_loss": test_loss,
        "epochs_ran": len(history["train_loss"]),
        "elapsed_sec": elapsed,
        "model_path": str(save_model_path.relative_to(BASE_DIR)),
    }

    print(f"[실험 완료] {candidate_name}")
    print(result)
    return result

# ------------------------------------------------------------
# 9. 전체 실행
# ------------------------------------------------------------
def main():
    print("\n[1] 전처리 후보 데이터셋 생성 시작")
    build_candidate_datasets()

    print("\n[2] 후보별 DNN 성능 비교 시작")
    all_results = []

    for candidate_name in PREPROCESSORS.keys():
        result = run_experiment(candidate_name)
        all_results.append(result)

    all_results = sorted(all_results, key=lambda x: x["best_val_acc"], reverse=True)

    print("\n" + "=" * 100)
    print("최종 후보 비교 결과 (validation accuracy 기준)")
    print("=" * 100)
    for i, r in enumerate(all_results, start=1):
        print(
            f"{i:02d}. {r['candidate']:<24} | "
            f"best_val_acc={r['best_val_acc']:.2f}% | "
            f"test_acc={r['final_test_acc']:.2f}% | "
            f"epochs={r['epochs_ran']:02d} | "
            f"time={r['elapsed_sec']:.1f}s"
        )

    best = all_results[0]
    print("\n[최종 선택된 최적 전처리]")
    print(best)

    result_txt = RESULT_ROOT / "candidate_comparison.txt"
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write("최종 후보 비교 결과 (validation accuracy 기준)\n")
        f.write("=" * 80 + "\n")
        for i, r in enumerate(all_results, start=1):
            f.write(
                f"{i:02d}. {r['candidate']:<24} | "
                f"best_val_acc={r['best_val_acc']:.2f}% | "
                f"test_acc={r['final_test_acc']:.2f}% | "
                f"epochs={r['epochs_ran']:02d} | "
                f"time={r['elapsed_sec']:.1f}s\n"
            )
        f.write("\n최종 선택된 최적 전처리\n")
        f.write(str(best) + "\n")

    print(f"\n결과 저장 완료: {result_txt}")
    print(f"곡선 이미지 저장 위치: {RESULT_ROOT}")

if __name__ == "__main__":
    main()