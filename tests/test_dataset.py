"""Сплит и препроцессинг."""
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms, stratified_split

ROOT = Path(__file__).resolve().parents[1]

# Размеры классов из README: 10 классов, дисбаланс в 4.2 раза
CLASS_SIZES = [756, 699, 1411, 1892, 1736, 930, 1336, 1597, 1449, 453]
TARGETS = np.repeat(np.arange(len(CLASS_SIZES)), CLASS_SIZES)


def test_split_is_a_partition():
    train, val, test = stratified_split(TARGETS)
    joined = np.concatenate([train, val, test])
    assert len(joined) == len(TARGETS)
    assert len(np.unique(joined)) == len(TARGETS)
    assert (len(train), len(val), len(test)) == (8581, 1839, 1839)


def test_split_keeps_class_proportions():
    train, val, test = stratified_split(TARGETS)
    overall = np.bincount(TARGETS) / len(TARGETS)
    for part in (train, val, test):
        share = np.bincount(TARGETS[part], minlength=len(CLASS_SIZES)) / len(part)
        assert np.abs(share - overall).max() < 0.005


def test_split_is_reproducible_with_seed():
    a = stratified_split(TARGETS, seed=42)
    b = stratified_split(TARGETS, seed=42)
    c = stratified_split(TARGETS, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[2], c[2])


def test_eval_transforms_are_deterministic_and_keep_aspect():
    # Вытянутый кадр: Resize(256) по короткой стороне + CenterCrop(224)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 500, 3), dtype=np.uint8))
    tf = get_eval_transforms()
    x1, x2 = tf(img), tf(img)
    assert x1.shape == (3, 224, 224)
    assert torch.equal(x1, x2)


def test_browser_demo_uses_same_preprocessing():
    # JS-препроцессинг в static/index.html обязан совпадать с get_eval_transforms()
    js = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert re.search(r"RESIZE\s*=\s*256", js) and re.search(r"CROP\s*=\s*224", js)

    def floats(name):
        m = re.search(name + r"\s*=\s*\[([^\]]+)\]", js)
        return tuple(float(v) for v in m.group(1).split(","))

    assert floats("MEAN") == IMAGENET_MEAN
    assert floats("STD") == IMAGENET_STD


def test_browser_demo_class_order_matches_imagefolder():
    # ImageFolder нумерует классы по алфавиту; порядок в JS должен совпадать
    js = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    block = js[js.index("const CLASS_INFO = {"):js.index("};", js.index("const CLASS_INFO = {"))]
    classes = re.findall(r"^\s+(\w+):\s*\[", block, flags=re.M)
    assert classes == sorted(classes) and len(classes) == len(CLASS_SIZES)
