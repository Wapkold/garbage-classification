"""Обучение модели.

Цикл по эпохам с логированием loss/accuracy на train и val, сохранением
лучшего чекпоинта по val accuracy и построением learning curves в конце.

Запуск:
    python -m src.train --epochs 25 --batch-size 32

Результаты раскладываются по имени прогона (--run-name, по умолчанию имя
архитектуры), чтобы прогоны не затирали друг друга:
    checkpoints/<run>/best.pt         — лучшие веса по val accuracy
    checkpoints/<run>/last.pt         — состояние после последней эпохи
    reports/<run>/history.json        — метрики по эпохам (двойник графика)
    reports/<run>/learning_curves.png — графики
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без GUI: скрипт должен работать и на headless-машине

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator, PercentFormatter
from tqdm import tqdm

# Чтобы работал и `python -m src.train`, и `python src/train.py` из IDE.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DEFAULT_VARIANT, IMG_SIZE, compute_class_weights, get_dataloaders
from src.model import ARCHITECTURES, TransferResNet18, build_from_arch, count_parameters, save_checkpoint

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Когда вывод перенаправлен в файл, tqdm пишет туда каждый кадр прогресс-бара:
# две эпохи дали 71 КБ мусора, двадцать пять дадут мегабайты. В терминале бар
# нужен, в файле — нет. Итоговые строки эпох идут через tqdm.write и остаются.
TQDM_DISABLE = not sys.stderr.isatty()

# Обучение упирается не в арифметику, а в распаковку JPEG на CPU. Замер на
# i7-8700K (6 ядер / 12 потоков) + RTX 3070, 224 px: 4 воркера — 90 с на эпоху,
# 10 воркеров с persistent_workers — 17 с. Дефолт в 4 оставлял пятикратный
# запас неиспользованным, поэтому подбираем от числа потоков.
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)

# --- Палитра графиков ------------------------------------------------------
# Слоты 1-2 референсной палитры: идентичность серии, не величина.
SERIES_TRAIN = "#2a78d6"  # синий
SERIES_VAL = "#eb6834"  # оранжевый
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


# --- Утилиты ---------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Одна эпоха ------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, float]:
    """Прогон по train-выборке. Возвращает (средний loss, accuracy)."""
    model.train()
    running_loss, correct, seen = 0.0, 0, 0

    bar = tqdm(
        loader,
        desc=f"эпоха {epoch:>2}/{epochs} train",
        leave=False,
        unit="батч",
        disable=TQDM_DISABLE,
    )
    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch = labels.size(0)
        running_loss += loss.item() * batch  # loss усредняется по батчу -> домножаем
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += batch
        bar.set_postfix(loss=f"{running_loss / seen:.4f}", acc=f"{correct / seen:.3f}")

    return running_loss / seen, correct / seen


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, float]:
    """Прогон по val-выборке без обновления весов."""
    model.eval()
    running_loss, correct, seen = 0.0, 0, 0

    bar = tqdm(
        loader,
        desc=f"эпоха {epoch:>2}/{epochs}   val",
        leave=False,
        unit="батч",
        disable=TQDM_DISABLE,
    )
    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch = labels.size(0)
        running_loss += loss.item() * batch
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += batch
        bar.set_postfix(loss=f"{running_loss / seen:.4f}", acc=f"{correct / seen:.3f}")

    return running_loss / seen, correct / seen


# --- Графики ---------------------------------------------------------------


def _style_axes(ax, title: str) -> None:
    """Общая отделка: убираем рамку, оставляем волосяную сетку."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK_PRIMARY, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")  # сплошная, не пунктир
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.set_xlabel("эпоха", fontsize=9, color=INK_SECONDARY)
    # Эпохи целые: без этого при коротком прогоне ось размечается как 1.0, 1.2, ...
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_learning_curves(history: dict, out_path: Path) -> None:
    """Две панели: loss и accuracy. Именно две, а не одна с двумя осями Y.

    Совмещать loss и accuracy на общей оси нельзя: у них разные шкалы и
    разный смысл, а произвольное совмещение двух шкал рисует корреляцию,
    которой в данных нет.
    """
    epochs = history["epoch"]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    # --- Loss ---
    _style_axes(ax_loss, "Функция потерь")
    ax_loss.plot(epochs, history["train_loss"], color=SERIES_TRAIN, linewidth=2, label="train")
    ax_loss.plot(epochs, history["val_loss"], color=SERIES_VAL, linewidth=2, label="val")
    # Кривые падают -> правый верхний угол свободен, легенда туда не налезет на данные.
    ax_loss.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_SECONDARY)

    # --- Accuracy ---
    _style_axes(ax_acc, "Доля верных ответов")
    ax_acc.plot(epochs, history["train_acc"], color=SERIES_TRAIN, linewidth=2, label="train")
    ax_acc.plot(epochs, history["val_acc"], color=SERIES_VAL, linewidth=2, label="val")
    ax_acc.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_acc.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK_SECONDARY)

    # Подписываем ровно одну точку — ту, ради которой всё считалось.
    best_i = int(np.argmax(history["val_acc"]))
    best_epoch, best_acc = epochs[best_i], history["val_acc"][best_i]
    ax_acc.plot(
        best_epoch,
        best_acc,
        marker="o",
        markersize=7,
        color=SERIES_VAL,
        markeredgecolor=SURFACE,  # кольцо цвета фона отделяет маркер от линии
        markeredgewidth=2,
        zorder=5,
    )
    # Подпись уходит вверх, в зазор между val и train: снизу к маркеру вплотную
    # подходит сама val-кривая, а справа внизу стоит легенда.
    # У правого края разворачиваем выключку, чтобы текст не вылез за пределы осей.
    near_right_edge = best_epoch > epochs[0] + 0.75 * (epochs[-1] - epochs[0])
    ax_acc.annotate(
        f"лучшая val {best_acc:.1%}\nэпоха {best_epoch}",
        xy=(best_epoch, best_acc),
        xytext=(-6 if near_right_edge else 0, 12),
        textcoords="offset points",
        ha="right" if near_right_edge else "center",
        va="bottom",
        fontsize=9,
        linespacing=1.4,
        color=INK_SECONDARY,  # текст носит текстовый цвет, а не цвет серии
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- Аргументы -------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение CNN на Garbage Classification")
    p.add_argument(
        "--arch",
        type=str,
        default="simple_cnn",
        choices=list(ARCHITECTURES),
        help="simple_cnn — своя сеть с нуля; resnet18_transfer — ImageNet-веса",
    )
    p.add_argument(
        "--unfreeze",
        action="store_true",
        help="для resnet18_transfer: разморозить backbone (полный fine-tuning). "
        "Тогда снижайте --lr примерно до 1e-4, иначе предобученные признаки "
        "разрушатся на первых же шагах",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--variant", type=str, default=DEFAULT_VARIANT)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"процессов загрузки данных (по умолчанию {DEFAULT_WORKERS}); "
        "при ошибках spawn на Windows ставьте 0",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="имя прогона; задаёт подпапки checkpoints/<run>/ и reports/<run>/. "
        "По умолчанию — имя архитектуры, чтобы прогоны не затирали друг друга",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--class-weights",
        action="store_true",
        help="взвесить loss обратно частоте классов (разброс в датасете 4.2x)",
    )
    return p.parse_args()


# --- Точка входа -----------------------------------------------------------


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    # Каждый прогон в свою подпапку: иначе обучение второй архитектуры молча
    # затирает веса и графики первой, и сравнить их уже нечем.
    run_name = args.run_name or args.arch
    checkpoint_dir = CHECKPOINT_DIR / run_name
    reports_dir = REPORTS_DIR / run_name
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, _test_loader, classes = get_dataloaders(
        data_dir=args.data_dir,
        variant=args.variant,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    build_kwargs = {"dropout": args.dropout}
    if args.arch == TransferResNet18.ARCH:
        build_kwargs["freeze_backbone"] = not args.unfreeze
    model = build_from_arch(args.arch, num_classes=len(classes), **build_kwargs).to(device)

    weights = None
    if args.class_weights:
        weights = compute_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Оптимизатору отдаём только размороженные параметры: у замороженного
    # backbone градиентов нет, и держать их в группах оптимизатора незачем.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    # По val accuracy, а не по loss: сохраняем модель по этой же метрике,
    # логично и LR резать по ней же. mode="max" — метрика растёт.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    print(f"прогон:      {run_name}")
    print(f"архитектура: {args.arch}")
    print(f"устройство:  {device}")
    print(f"классы ({len(classes)}): {', '.join(classes)}")
    print(f"train/val:   {len(train_loader.dataset)} / {len(val_loader.dataset)} изображений")
    print(
        f"параметров:  {count_parameters(model):,} обучаемых "
        f"из {count_parameters(model, trainable_only=False):,}"
    )
    print(f"взвешивание классов: {'да' if args.class_weights else 'нет'}\n")

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_epoch = 0.0, 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, args.epochs)

        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - started

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        is_best = val_acc > best_acc
        if is_best:
            best_acc, best_epoch = val_acc, epoch
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                classes,
                epoch=epoch,
                optimizer=optimizer,
                metrics={"val_acc": val_acc, "val_loss": val_loss},
            )
        save_checkpoint(checkpoint_dir / "last.pt", model, classes, epoch=epoch, optimizer=optimizer)

        # tqdm.write, а не print: иначе строка ломает активный прогресс-бар.
        tqdm.write(
            f"эпоха {epoch:>2}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:6.2%} | "
            f"val loss {val_loss:.4f} acc {val_acc:6.2%} | "
            f"lr {lr:.1e} | {elapsed:5.1f}s" + ("  <- лучшая" if is_best else "")
        )

    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "args": vars(args), "history": history}, f, indent=2)

    curves_path = reports_dir / "learning_curves.png"
    plot_learning_curves(history, curves_path)

    print(f"\nЛучшая val accuracy: {best_acc:.2%} (эпоха {best_epoch})")
    print(f"Веса:    {checkpoint_dir / 'best.pt'}")
    print(f"Графики: {curves_path}")
    print(f"Метрики: {reports_dir / 'history.json'}")


if __name__ == "__main__":
    # Guard обязателен: на Windows DataLoader с num_workers > 0 запускает
    # воркеры через spawn, и без него они рекурсивно переимпортируют модуль.
    main()
