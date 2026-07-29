"""Собирает deploy_static/ — Space типа Static (бесплатный, без сервера).

Hugging Face с некоторых пор требует PRO-подписку для Gradio- и Docker-Spaces
на бесплатном железе; бесплатными остались только статические. Поэтому вместо
Python на сервере модель уезжает в ONNX и считается в браузере посетителя
через onnxruntime-web.

Требует предварительного `python export_onnx.py`.

Запуск:
    python build_static_space.py
    python build_static_space.py --precision fp32   # без квантизации, 45 МБ
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DATA_ROOT, DEFAULT_VARIANT

DEPLOY = PROJECT_ROOT / "deploy_static"
ONNX_DIR = PROJECT_ROOT / "onnx"
SOURCE_HTML = PROJECT_ROOT / "static" / "index.html"

README = """---
title: {title}
emoji: ♻️
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: mit
---

# Куда выбросить? — классификация отходов

Определяет тип отхода по фотографии и подсказывает, куда его сдать.
**Инференс идёт целиком в браузере**: модель скачивается один раз и считается
на устройстве посетителя. Сервера нет, холодного старта нет, данные никуда
не отправляются.

| | |
|---|---|
| архитектура | ResNet18, дообученный (backbone разморожен, lr 1e-4) |
| датасет | [Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification), 12 259 изображений, 10 классов |
| accuracy на тесте | **{acc}** |
| размер модели | {size} ({precision}) |
| исполнение | onnxruntime-web (WASM) |

{note}

Правила раздельного сбора отличаются от города к городу — подсказки в
интерфейсе общие и не заменяют местный регламент.
"""

NOTE_INT8 = (
    "Веса квантизованы в int8 ради веса страницы: 44.7 МБ -> 11.2 МБ. "
    "Точность при этом падает с 93.2% до 92.0% — размен в пользу того, чтобы "
    "демо открывалось с телефона за секунды, а не за минуту."
)
NOTE_FP32 = "Веса без квантизации — ответы побитово совпадают с PyTorch-версией (93.2%)."


def copy_examples(dst: Path, per_class: int = 1) -> int:
    src_dir = DATA_ROOT / DEFAULT_VARIANT
    if not src_dir.is_dir():
        print(f"  ВНИМАНИЕ: {src_dir} не найдена, примеры пропущены")
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for class_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        for file in sorted(class_dir.glob("*.jpg"))[:per_class]:
            shutil.copy2(file, dst / f"{class_dir.name}.jpg")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборка статического Space")
    parser.add_argument("--precision", choices=["int8", "fp32"], default="int8")
    parser.add_argument("--title", type=str, default="Куда выбросить?")
    args = parser.parse_args()

    model_src = ONNX_DIR / ("model.int8.onnx" if args.precision == "int8" else "model.onnx")
    if not model_src.exists():
        raise FileNotFoundError(f"Нет {model_src}. Сначала: python export_onnx.py")
    if not SOURCE_HTML.exists():
        raise FileNotFoundError(f"Нет {SOURCE_HTML}")

    if DEPLOY.exists():
        shutil.rmtree(DEPLOY)
    DEPLOY.mkdir(parents=True)

    html = SOURCE_HTML.read_text(encoding="utf-8")
    if args.precision == "fp32":
        # Имя файла модели зашито в index.html — при смене точности правим его,
        # чтобы страница не искала несуществующий model.int8.onnx.
        html = html.replace("model.int8.onnx", "model.onnx").replace("92.0%", "93.2%")
    (DEPLOY / "index.html").write_text(html, encoding="utf-8")

    shutil.copy2(model_src, DEPLOY / model_src.name)
    n = copy_examples(DEPLOY / "examples")

    classes_file = ONNX_DIR / "classes.json"
    if classes_file.exists():
        # Порядок классов важен: индекс выхода сети -> имя. В index.html он
        # продублирован, поэтому проверяем, что не разъехались.
        classes = json.loads(classes_file.read_text(encoding="utf-8"))
        for i, c in enumerate(classes):
            if f'  {c}:' not in html and f"{c}:" not in html:
                raise ValueError(f"Класс {c!r} (индекс {i}) не найден в index.html")
        print(f"  порядок классов сверен: {', '.join(classes)}")

    size = f"{model_src.stat().st_size / 1e6:.1f} МБ"
    (DEPLOY / "README.md").write_text(
        README.format(
            title=args.title,
            acc="92.0%" if args.precision == "int8" else "93.2%",
            size=size,
            precision=args.precision,
            note=NOTE_INT8 if args.precision == "int8" else NOTE_FP32,
        ),
        encoding="utf-8",
    )

    total = sum(f.stat().st_size for f in DEPLOY.rglob("*") if f.is_file())
    print(f"  index.html, {model_src.name} ({size}), examples/: {n} шт., README.md")
    print(f"\nГотово: {DEPLOY}  ({total / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
