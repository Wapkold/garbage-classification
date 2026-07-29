"""Загрузка и подготовка датасета Garbage Classification (Kaggle).

Данные лежат в data/garbage_classification/<variant>/<class_name>/*.jpg,
что напрямую ложится на torchvision.datasets.ImageFolder: имя папки = метка класса.

Ключевые решения:
- аугментации применяются только к train, val/test прогоняются детерминированно;
- разбиение 70/15/15 стратифицированное — классы сильно несбалансированы
  (clothes 1892 против trash 453), при случайном сплите редкие классы «плывут»;
- нормализация под ImageNet-статистику, потому что бэкбон берётся предобученным.

Запуск для проверки: python -m src.dataset
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# --- Константы -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "garbage_classification"

# В датасете три варианта одних и тех же изображений: original (разный размер),
# standardized_256 (256x256) и standardized_384 (384x384).
#
# Берём original, хотя это и не самый удобный формат. Причина: оба
# standardized-варианта приведены к квадрату ДОПОЛНЕНИЕМ СЕРЫМ ПОЛЕМ (114,114,114).
# Замер по 60 файлов на класс: поля есть у 77% файлов, в среднем 21% кадра,
# максимум 76%. Доля сильно зависит от класса — 43% файлов у battery против 95%
# у biological, — то есть количество серого коррелирует с меткой, и сеть может
# читать его как признак вместо самого объекта. Плюс кроп 224 из 256 при поле в
# 20% регулярно вырезал бы преимущественно паддинг.
# В original полей нет вообще (проверено на той же выборке), а приведение к
# нужному размеру мы и так делаем трансформами ниже.
DEFAULT_VARIANT = "original"

RESIZE_SIZE = 256  # к этому размеру приводим короткую сторону перед кропом
IMG_SIZE = 224  # вход сети (стандарт для ImageNet-бэкбонов)

# Статистика ImageNet: предобученные веса учились на так нормализованных данных,
# поэтому вход должен иметь то же распределение.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Тот же mean, но в шкале 0..255 — им заполняем углы после поворота.
# После нормализации такая заливка даёт ~0, то есть «нейтральный» для сети пиксель,
# а не чёрное пятно, которого в реальных фото не бывает.
ROTATION_FILL = tuple(int(round(c * 255)) for c in IMAGENET_MEAN)

SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test
SEED = 42


# --- Трансформы ------------------------------------------------------------


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Аугментации для обучения.

    Обоснование каждого шага — в README раздела «Подход» и в docstring модуля.
    """
    return transforms.Compose(
        [
            # Resize с одним числом масштабирует КОРОТКУЮ сторону, сохраняя
            # пропорции; кадрирование до квадрата делает следующий шаг.
            # Вариант Resize((256, 256)) сплющил бы вытянутые фото — а бутылка,
            # сжатая по вертикали, начинает выглядеть как банка.
            transforms.Resize(RESIZE_SIZE),
            # Случайный кроп вместо центрального: объект оказывается в разных
            # местах кадра, сеть перестаёт полагаться на его положение.
            transforms.RandomCrop(img_size),
            # Мусор не имеет «правильной» левой и правой стороны — отражение
            # по горизонтали даёт валидное изображение того же класса.
            transforms.RandomHorizontalFlip(p=0.5),
            # Съёмка «сверху в мусорное ведро» — угол поворота произвольный,
            # но ±15° хватает: сильнее вращать смысла нет, а артефактов больше.
            transforms.RandomRotation(
                degrees=15,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=ROTATION_FILL,
            ),
            # Освещение в кадре меняется сильно (кухня, улица, вспышка).
            # Насыщенность/яркость не должны становиться признаком класса.
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.25,
                hue=0.03,  # оттенок трогаем чуть-чуть: цвет — реальный признак
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Детерминированный препроцессинг для val/test — без единой аугментации.

    Метрика должна быть воспроизводимой: один и тот же файл обязан давать
    один и тот же ответ модели при любом запуске.
    """
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),  # короткая сторона, пропорции целы
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --- Разбиение -------------------------------------------------------------


def _resolve_data_dir(data_dir: str | os.PathLike | None, variant: str) -> Path:
    path = Path(data_dir) if data_dir is not None else DATA_ROOT / variant
    if not path.is_dir():
        raise FileNotFoundError(
            f"Не найдена папка с данными: {path}\n"
            f"Ожидается структура <data_dir>/<class_name>/*.jpg. "
            f"Доступные варианты в {DATA_ROOT}: "
            f"{[p.name for p in DATA_ROOT.iterdir() if p.is_dir()] if DATA_ROOT.is_dir() else '—'}"
        )
    return path


def stratified_split(
    targets: list[int],
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Индексы train/val/test с сохранением пропорции классов в каждой части.

    Стратификация здесь не косметика: у самого редкого класса (trash, 453 файла)
    при случайном сплите доля в тесте гуляет настолько, что per-class recall
    становится несопоставимым между запусками.
    """
    train_ratio, val_ratio, test_ratio = ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Доли должны давать в сумме 1.0, получено {ratios}")

    targets_arr = np.asarray(targets)
    indices = np.arange(len(targets_arr))

    train_idx, rest_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=targets_arr,
        random_state=seed,
        shuffle=True,
    )
    # Остаток (30%) делим пополам -> 15% / 15% от исходного объёма.
    val_share = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=val_share,
        stratify=targets_arr[rest_idx],
        random_state=seed,
        shuffle=True,
    )
    return train_idx, val_idx, test_idx


def build_datasets(
    data_dir: str | os.PathLike | None = None,
    variant: str = DEFAULT_VARIANT,
    img_size: int = IMG_SIZE,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = SEED,
) -> tuple[Subset, Subset, Subset, list[str]]:
    """Три подвыборки одного ImageFolder, но с разными трансформами.

    Приём: создаём два ImageFolder поверх одной папки — с train- и eval-трансформами.
    ImageFolder сортирует файлы детерминированно, поэтому индексы у них совпадают,
    и один и тот же набор индексов можно брать то из одного, то из другого.
    Так train получает аугментации, а val/test — нет; при random_split поверх
    одного датасета трансформ был бы общий на все три части.
    """
    root = _resolve_data_dir(data_dir, variant)

    train_source = ImageFolder(root, transform=get_train_transforms(img_size))
    eval_source = ImageFolder(root, transform=get_eval_transforms(img_size))

    train_idx, val_idx, test_idx = stratified_split(train_source.targets, ratios, seed)

    train_ds = Subset(train_source, train_idx.tolist())
    val_ds = Subset(eval_source, val_idx.tolist())
    test_ds = Subset(eval_source, test_idx.tolist())

    return train_ds, val_ds, test_ds, train_source.classes


# --- DataLoader'ы ----------------------------------------------------------


def get_dataloaders(
    data_dir: str | os.PathLike | None = None,
    variant: str = DEFAULT_VARIANT,
    batch_size: int = 32,
    img_size: int = IMG_SIZE,
    num_workers: int = 4,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Готовые загрузчики train/val/test и список имён классов.

    На Windows num_workers > 0 требует, чтобы вызов был под
    `if __name__ == "__main__":` — иначе процессы-воркеры уйдут в рекурсивный
    импорт. Если ловите такую ошибку — поставьте num_workers=0.
    """
    train_ds, val_ds, test_ds, classes = build_datasets(
        data_dir=data_dir, variant=variant, img_size=img_size, ratios=ratios, seed=seed
    )

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    common = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        # На Windows воркеры порождаются через spawn: каждый заново импортирует
        # torch и модули проекта — это единицы секунд на воркера. Без
        # persistent_workers эта плата вносится КАЖДУЮ эпоху.
        common["persistent_workers"] = True
        # Очередь готовых батчей на воркера. Узкое место здесь — распаковка
        # JPEG, а не арифметика, так что запас побольше держит GPU занятым.
        common["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # ровные батчи: стабильнее BatchNorm
        generator=generator,
        **common,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader, classes


def compute_class_weights(dataset: Subset) -> torch.Tensor:
    """Веса классов, обратные их частоте — для CrossEntropyLoss(weight=...).

    Разброс в датасете 4.2x (clothes 1892 / trash 453), поэтому без взвешивания
    (или иной компенсации дисбаланса) модель охотно жертвует редкими классами.
    """
    source = dataset.dataset  # ImageFolder под Subset
    targets = np.asarray(source.targets)[dataset.indices]
    counts = np.bincount(targets, minlength=len(source.classes)).astype(np.float64)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


# --- Проверка --------------------------------------------------------------

if __name__ == "__main__":
    train_ds, val_ds, test_ds, classes = build_datasets()
    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"Классы ({len(classes)}): {classes}")
    print(f"Всего: {total} | train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    source = train_ds.dataset
    all_targets = np.asarray(source.targets)
    print("\nРаспределение по классам (train / val / test):")
    for i, name in enumerate(classes):
        row = [int((all_targets[ds.indices] == i).sum()) for ds in (train_ds, val_ds, test_ds)]
        print(f"  {name:<12} {row[0]:>5} {row[1]:>5} {row[2]:>5}")

    images, labels = next(iter(DataLoader(train_ds, batch_size=4, shuffle=True)))
    print(f"\nБатч: {tuple(images.shape)}, dtype={images.dtype}, метки={labels.tolist()}")
    print(f"Диапазон значений после нормализации: [{images.min():.2f}, {images.max():.2f}]")
