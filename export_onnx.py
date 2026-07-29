"""Экспорт обученной модели в ONNX для инференса в браузере.

Статический Space не умеет исполнять Python, поэтому модель уезжает в ONNX и
считается на устройстве посетителя через onnxruntime-web. Веса квантизуются в
int8: 45 МБ фронтенду отдавать неприлично, страница будет грузиться минуту.

Квантизация — это потеря точности, поэтому скрипт не просто конвертирует, а
СРАВНИВАЕТ три модели на настоящей тестовой выборке. Если int8 просядет
заметно, лучше знать об этом здесь, а не по жалобам пользователей.

Запуск:
    python export_onnx.py
    python export_onnx.py --limit 300     # быстрее, по подвыборке
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import IMG_SIZE, get_dataloaders
from src.model import load_checkpoint

OUT_DIR = PROJECT_ROOT / "onnx"


def pick_checkpoint() -> Path:
    candidates = sorted((PROJECT_ROOT / "checkpoints").glob("*/best.pt"))
    if not candidates:
        raise FileNotFoundError("Нет чекпоинтов. Сначала обучите: python -m src.train")

    def val_acc(path: Path) -> float:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return float(payload.get("metrics", {}).get("val_acc", 0.0))

    return max(candidates, key=val_acc)


def export(model: torch.nn.Module, path: Path, img_size: int) -> None:
    """ONNX с фиксированным батчем 1: в браузере всегда одна картинка."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def collect_batches(loader, limit: int | None):
    """Тензоры и метки тестовой выборки — общие для всех трёх замеров."""
    xs, ys = [], []
    seen = 0
    for images, labels in loader:
        xs.append(images.numpy())
        ys.append(labels.numpy())
        seen += labels.numel()
        if limit and seen >= limit:
            break
    x = np.concatenate(xs)[:limit] if limit else np.concatenate(xs)
    y = np.concatenate(ys)[:limit] if limit else np.concatenate(ys)
    return x, y


def run_onnx(path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    # Батч зафиксирован единицей — гоняем по одному, как это будет в браузере.
    return np.concatenate([sess.run(None, {name: x[i : i + 1]})[0] for i in range(len(x))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт модели в ONNX + квантизация")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="ограничить размер проверки")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else pick_checkpoint()
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    model, classes, payload = load_checkpoint(checkpoint, device="cpu")
    model.eval()
    print(f"чекпоинт: {checkpoint.relative_to(PROJECT_ROOT)}")
    print(f"прогон:   {payload.get('run_name') or checkpoint.parent.name}\n")

    fp32 = OUT_DIR / "model.onnx"
    int8 = OUT_DIR / "model.int8.onnx"

    print("экспорт в ONNX...")
    export(model, fp32, IMG_SIZE)
    print(f"  {fp32.name}: {fp32.stat().st_size / 1e6:.1f} МБ")

    print("квантизация весов в int8...")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8)
    print(f"  {int8.name}: {int8.stat().st_size / 1e6:.1f} МБ "
          f"(в {fp32.stat().st_size / int8.stat().st_size:.1f} раза меньше)\n")

    # --- сверка на настоящих данных ---
    _tr, _va, test_loader, loader_classes = get_dataloaders(batch_size=32, num_workers=0)
    if loader_classes != classes:
        raise ValueError("Классы чекпоинта и данных разошлись")

    x, y = collect_batches(test_loader, args.limit)
    print(f"сверка на {len(y)} изображениях теста\n")

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(x)).numpy()

    results = {"PyTorch": torch_logits, "ONNX fp32": run_onnx(fp32, x), "ONNX int8": run_onnx(int8, x)}

    base_pred = results["PyTorch"].argmax(1)
    print(f"{'модель':<12} {'accuracy':>9} {'совпадений с PyTorch':>22} {'max |Δ логита|':>16}")
    for name, logits in results.items():
        pred = logits.argmax(1)
        acc = (pred == y).mean()
        agree = (pred == base_pred).mean()
        delta = np.abs(logits - torch_logits).max()
        print(f"{name:<12} {acc:>9.4f} {agree:>21.2%} {delta:>16.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "classes.json").write_text(
        __import__("json").dumps(classes, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nклассы записаны: {OUT_DIR / 'classes.json'}")


if __name__ == "__main__":
    main()
