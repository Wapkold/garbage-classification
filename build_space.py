"""Собирает папку deploy/ — готовый к загрузке Hugging Face Space.

Space не имеет ни датасета, ни истории обучения, поэтому в него кладутся:
облегчённый чекпоинт (без состояния оптимизатора), десяток примеров картинок
и код инференса. Файлы не пишутся заново, а копируются из проекта — иначе
копия app.py со временем разъедется с оригиналом.

Запуск:
    python build_space.py
    python build_space.py --checkpoint checkpoints/resnet18_finetune/best.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DATA_ROOT, DEFAULT_VARIANT
from src.model import export_for_inference

DEPLOY_DIR = PROJECT_ROOT / "deploy"

# Torch с индекса PyTorch: на Spaces бесплатный тариф — CPU, и обычная сборка
# с PyPI притащила бы CUDA-зависимости, раздув образ на несколько гигабайт.
SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
gradio
pillow
numpy
scikit-learn
"""

SPACE_README = """---
title: {title}
emoji: ♻️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: {sdk_version}
app_file: app.py
pinned: false
---

# Классификация мусора по фотографии

Модель относит изображение к одному из 10 типов отходов и подсказывает, что с
предметом делать.

- архитектура: `{arch}` (прогон `{run_name}`)
- точность на валидации: **{val_acc:.1%}**
- обучено на [Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification), 12 259 изображений

Правила раздельного сбора отличаются от города к городу — подсказки в интерфейсе
общие и не заменяют местный регламент.

Исходный код обучения, метрики и разбор ошибок — в основном репозитории проекта.
"""


def pick_checkpoint() -> Path:
    """Лучший по val accuracy среди checkpoints/*/best.pt."""
    candidates = sorted((PROJECT_ROOT / "checkpoints").glob("*/best.pt"))
    if not candidates:
        raise FileNotFoundError("Нет ни одного чекпоинта. Сначала обучите: python -m src.train")

    def val_acc(path: Path) -> float:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return float(payload.get("metrics", {}).get("val_acc", 0.0))

    return max(candidates, key=val_acc)


def copy_examples(dst: Path, per_class: int = 1) -> int:
    """По одному изображению на класс, с осмысленными именами файлов."""
    src_dir = DATA_ROOT / DEFAULT_VARIANT
    if not src_dir.is_dir():
        print(f"  ВНИМАНИЕ: {src_dir} не найдена, примеры пропущены")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for class_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        for i, file in enumerate(sorted(class_dir.glob("*.jpg"))[:per_class]):
            shutil.copy2(file, dst / f"{class_dir.name}{'' if per_class == 1 else f'_{i}'}.jpg")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборка папки для Hugging Face Space")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--title", type=str, default="Куда выбросить?")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else pick_checkpoint()
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run_name = payload.get("run_name") or checkpoint.parent.name
    arch = payload.get("arch", "?")
    val_acc = float(payload.get("metrics", {}).get("val_acc", 0.0))

    print(f"чекпоинт: {checkpoint.relative_to(PROJECT_ROOT)}")
    print(f"прогон:   {run_name} ({arch}), val accuracy {val_acc:.2%}\n")

    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)
    DEPLOY_DIR.mkdir(parents=True)

    # Код инференса — копией, чтобы деплой и проект не разъезжались.
    shutil.copy2(PROJECT_ROOT / "app" / "app.py", DEPLOY_DIR / "app.py")
    src_dst = DEPLOY_DIR / "src"
    src_dst.mkdir()
    for name in ("dataset.py", "model.py"):
        shutil.copy2(PROJECT_ROOT / "src" / name, src_dst / name)
    print("  app.py, src/dataset.py, src/model.py скопированы")

    # Имя прогона дописываем в сам чекпоинт: на Space папки прогона нет,
    # и вывести его в интерфейсе больше неоткуда.
    was, now = export_for_inference(
        checkpoint, DEPLOY_DIR / "model.pt", extra={"run_name": run_name}
    )
    print(f"  model.pt: {was / 1e6:.1f} МБ -> {now / 1e6:.1f} МБ")

    n = copy_examples(DEPLOY_DIR / "examples")
    print(f"  examples/: {n} изображений")

    (DEPLOY_DIR / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
    (DEPLOY_DIR / "README.md").write_text(
        SPACE_README.format(
            title=args.title,
            sdk_version=__import__("gradio").__version__,
            arch=arch,
            run_name=run_name,
            val_acc=val_acc,
        ),
        encoding="utf-8",
    )
    print("  requirements.txt, README.md записаны")

    total = sum(f.stat().st_size for f in DEPLOY_DIR.rglob("*") if f.is_file())
    print(f"\nГотово: {DEPLOY_DIR}  ({total / 1e6:.1f} МБ)")
    print("\nДальше — загрузить содержимое deploy/ в Space (Gradio SDK).")


if __name__ == "__main__":
    main()
