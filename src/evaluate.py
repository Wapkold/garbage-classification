"""Оценка обученной модели на отложенной выборке.

Считает confusion matrix, per-class precision/recall/F1 и вытаскивает примеры,
на которых модель ошиблась сильнее всего.

Запуск:
    python -m src.evaluate --checkpoint checkpoints/resnet18_transfer/best.pt

Результаты кладутся в reports/<имя прогона>/, где имя прогона берётся из
папки чекпоинта — так отчёты по разным архитектурам не смешиваются:
    confusion_matrix.png       — матрица ошибок
    classification_report.txt  — отчёт в том же виде, что в консоли
    classification_report.json — он же машиночитаемо
    worst_errors.png           — сетка худших ошибок
    worst_errors.csv           — они же таблицей, с путями к файлам
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DEFAULT_VARIANT, IMG_SIZE, get_dataloaders
from src.model import load_checkpoint

REPORTS_DIR = PROJECT_ROOT / "reports"

# См. комментарий в train.py: в файл прогресс-бар писать не надо.
TQDM_DISABLE = not sys.stderr.isatty()
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)

# --- Палитра ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
AXIS = "#c3c2b7"

# Матрица ошибок кодирует величину, а не идентичность, поэтому ramp одноцветный
# и светлеет к нулю: пустая клетка должна сливаться с фоном, а не быть «ещё одним
# цветом». Радужные карты (jet, viridis) для этого не годятся — читатель не может
# упорядочить их оттенки на глаз.
SEQUENTIAL_BLUE = [
    SURFACE,
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)


# --- Инференс --------------------------------------------------------------


@torch.no_grad()
def predict(
    model: torch.nn.Module, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Прогон по выборке. Возвращает (истина, предсказание, вероятности)."""
    model.eval()
    trues, preds, probs = [], [], []

    for images, labels in tqdm(loader, desc="инференс", unit="батч", disable=TQDM_DISABLE):
        images = images.to(device, non_blocking=True)
        batch_probs = torch.softmax(model(images), dim=1).cpu()
        probs.append(batch_probs)
        preds.append(batch_probs.argmax(dim=1))
        trues.append(labels)

    return (
        torch.cat(trues).numpy(),
        torch.cat(preds).numpy(),
        torch.cat(probs).numpy(),
    )


def get_file_paths(loader) -> list[str]:
    """Пути к файлам в порядке выдачи лоадера.

    Работает только потому, что val/test-лоадеры собраны с shuffle=False:
    i-й прогнанный пример — это i-й индекс в Subset.
    """
    subset = loader.dataset  # Subset поверх ImageFolder
    source = subset.dataset
    return [source.samples[i][0] for i in subset.indices]


# --- Матрица ошибок --------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], out_path: Path
) -> np.ndarray:
    """Цвет — доля от строки, подпись — абсолютное число. Возвращает матрицу счётчиков.

    Нормировка по строке обязательна при нашем дисбалансе: в тесте у clothes
    ~284 примера, у trash ~68. Раскрась мы сырые счётчики — шкалу занял бы
    самый крупный класс, а ошибки редких классов слились бы с нулями.
    Строка = истинный класс, поэтому диагональ нормированной матрицы это recall.
    """
    counts = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    row_sums = counts.sum(axis=1, keepdims=True)
    shares = counts / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(8.2, 7.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(shares, cmap=CMAP, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes)), labels=classes)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.set_xlabel("предсказано", fontsize=10, color=INK_SECONDARY, labelpad=10)
    ax.set_ylabel("истинный класс", fontsize=10, color=INK_SECONDARY, labelpad=10)
    ax.set_title("Матрица ошибок", loc="left", fontsize=12, color=INK_PRIMARY, pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Волосяная сетка между клетками вместо рамок вокруг них.
    ax.set_xticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(classes)):
        for j in range(len(classes)):
            if counts[i, j] == 0:
                continue  # нули не подписываем: пустая клетка и так читается
            # На тёмной заливке тёмный текст не виден — переключаем на светлый.
            color = SURFACE if shares[i, j] > 0.55 else INK_SECONDARY
            ax.text(
                j, i, f"{counts[i, j]}", ha="center", va="center", fontsize=8.5, color=color
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    cbar.set_label("доля класса (строки)", fontsize=9, color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    cbar.outline.set_edgecolor(AXIS)
    cbar.outline.set_linewidth(0.8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return counts


# --- Худшие ошибки ---------------------------------------------------------


def find_worst_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    paths: list[str],
    classes: list[str],
    top_k: int = 10,
) -> list[dict]:
    """Ошибки, отсортированные по «насколько сильно» модель промахнулась.

    Мера — отрыв p(предсказанный) - p(истинный). Она строже, чем просто
    уверенность в ответе: наверх поднимаются случаи, где модель не только
    уверена в неправильном классе, но и почти не рассматривала правильный.
    Ошибка с p(pred)=0.55 при p(true)=0.40 — это близкий промах, а не грубый.
    """
    wrong = np.flatnonzero(y_true != y_pred)
    if wrong.size == 0:
        return []

    p_pred = probs[wrong, y_pred[wrong]]
    p_true = probs[wrong, y_true[wrong]]
    order = wrong[np.argsort(-(p_pred - p_true))][:top_k]

    return [
        {
            "path": paths[i],
            "true": classes[y_true[i]],
            "pred": classes[y_pred[i]],
            "p_pred": float(probs[i, y_pred[i]]),
            "p_true": float(probs[i, y_true[i]]),
            "margin": float(probs[i, y_pred[i]] - probs[i, y_true[i]]),
        }
        for i in order
    ]


def plot_worst_errors(errors: list[dict], out_path: Path, cols: int = 5) -> None:
    """Сетка исходных изображений с подписью «истина -> предсказание»."""
    if not errors:
        return

    rows = (len(errors) + cols - 1) // cols
    # constrained, а не tight_layout: у imshow жёсткое соотношение сторон, из-за
    # чего бокс осей выше картинки, и tight_layout сажает подпись следующего
    # ряда на изображение предыдущего.
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.6 * cols, 3.3 * rows), dpi=150, layout="constrained"
    )
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()

    for ax, err in zip(axes, errors):
        ax.set_facecolor(SURFACE)
        # Показываем исходный файл, а не тензор из лоадера: нормализованную
        # картинку пришлось бы денормализовать, и смотрел бы человек всё равно
        # не на то, что лежит на диске.
        with Image.open(err["path"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(
            f"{err['true']} -> {err['pred']}\n"
            f"p={err['p_pred']:.2f}  (истинный {err['p_true']:.2f})",
            fontsize=8.5,
            color=INK_SECONDARY,
            pad=6,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(AXIS)
            spine.set_linewidth(0.8)

    for ax in axes[len(errors) :]:
        ax.axis("off")

    fig.suptitle("Самые грубые ошибки", x=0.01, ha="left", fontsize=12, color=INK_PRIMARY)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_worst_errors_csv(errors: list[dict], out_path: Path) -> None:
    """Табличный двойник картинки: по нему ошибки можно листать и фильтровать."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "true", "pred", "p_pred", "p_true", "margin"]
        )
        writer.writeheader()
        writer.writerows(errors)


# --- Аргументы -------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Оценка модели на отложенной выборке")
    p.add_argument("--checkpoint", type=str, default="checkpoints/resnet18_transfer/best.pt")
    p.add_argument("--split", type=str, default="test", choices=["test", "val"])
    p.add_argument("--top-k", type=int, default=10, help="сколько худших ошибок показать")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--variant", type=str, default=DEFAULT_VARIANT)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--seed", type=int, default=42, help="должен совпадать с обучением")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


# --- Точка входа -----------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Нет чекпоинта: {checkpoint}. Сначала обучите: python -m src.train")

    # Отчёты кладём в папку того же прогона, что и чекпоинт: train.py пишет
    # веса в checkpoints/<run>/best.pt, значит имя прогона — это имя папки.
    reports_dir = REPORTS_DIR / checkpoint.parent.name

    model, classes, payload = load_checkpoint(checkpoint, device=device)

    # seed тот же, что при обучении, иначе сплит разъедется и в «тесте»
    # окажутся картинки, на которых модель училась.
    _train_loader, val_loader, test_loader, loader_classes = get_dataloaders(
        data_dir=args.data_dir,
        variant=args.variant,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    if loader_classes != classes:
        raise ValueError(
            "Классы в чекпоинте и в папке с данными разошлись:\n"
            f"  чекпоинт: {classes}\n  данные:   {loader_classes}"
        )

    loader = test_loader if args.split == "test" else val_loader

    print(f"чекпоинт:    {checkpoint}")
    print(f"архитектура: {payload.get('arch', 'simple_cnn')}, эпоха {payload.get('epoch')}")
    print(f"выборка:     {args.split}, {len(loader.dataset)} изображений")
    print(f"устройство:  {device}\n")

    y_true, y_pred, probs = predict(model, loader, device)
    paths = get_file_paths(loader)

    accuracy = float((y_true == y_pred).mean())
    balanced = float(balanced_accuracy_score(y_true, y_pred))
    report_text = classification_report(
        y_true, y_pred, target_names=classes, digits=3, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )

    print(report_text)
    print(f"accuracy:          {accuracy:.4f}")
    # Среднее recall по классам: показывает, не куплена ли accuracy
    # за счёт того, что редкие классы просто игнорируются.
    print(f"balanced accuracy: {balanced:.4f}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    counts = plot_confusion_matrix(y_true, y_pred, classes, reports_dir / "confusion_matrix.png")

    errors = find_worst_errors(y_true, y_pred, probs, paths, classes, top_k=args.top_k)
    plot_worst_errors(errors, reports_dir / "worst_errors.png")
    save_worst_errors_csv(errors, reports_dir / "worst_errors.csv")

    with open(reports_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"чекпоинт: {checkpoint}\nвыборка: {args.split}\n\n")
        f.write(report_text)
        f.write(f"\naccuracy:          {accuracy:.4f}\n")
        f.write(f"balanced accuracy: {balanced:.4f}\n")

    with open(reports_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint),
                "split": args.split,
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "per_class": report_dict,
                "confusion_matrix": counts.tolist(),
                "classes": classes,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nНайдено ошибок: {int((y_true != y_pred).sum())} из {len(y_true)}")
    if errors:
        print(f"Топ-{len(errors)} самых грубых:")
        for i, err in enumerate(errors, start=1):
            name = Path(err["path"]).name
            print(
                f"  {i:>2}. {err['true']:<11} -> {err['pred']:<11} "
                f"p={err['p_pred']:.3f} (истинный {err['p_true']:.3f})  {name}"
            )

    print(f"\nОтчёты: {reports_dir}")


if __name__ == "__main__":
    main()
