"""Архитектуры моделей: своя CNN с нуля и transfer learning на ResNet18.

В модуле два варианта, между которыми выбирает train.py флагом --arch:

    simple_cnn        — build_model(): 4 conv-блока, ~0.39M параметров,
                        все обучаемые, весов на старте нет;
    resnet18_transfer — build_transfer_model(): ResNet18 с весами ImageNet,
                        backbone заморожен, обучается только новая голова
                        (5 130 параметров при 10 классах).

Сравнение подходов — в README, раздел «Подход».

--- simple_cnn --------------------------------------------------------------

Главное ограничение, из которого следует всё остальное: обучающая выборка —
около 8.6k изображений (70% от 12 259), предобученных весов нет. Значит сеть
должна быть маленькой по числу параметров, иначе она выучит выборку наизусть
раньше, чем найдёт обобщающие признаки.

Схема (вход 3x224x224):

    блок 1:  Conv3x3(3->32)   -> BN -> ReLU -> MaxPool2  ->  32x112x112
    блок 2:  Conv3x3(32->64)  -> BN -> ReLU -> MaxPool2  ->  64x56x56
    блок 3:  Conv3x3(64->128) -> BN -> ReLU -> MaxPool2  -> 128x28x28
    блок 4:  Conv3x3(128->256)-> BN -> ReLU -> MaxPool2  -> 256x14x14
    голова:  GlobalAvgPool -> Dropout -> Linear(256 -> num_classes)

Итого ~0.39M обучаемых параметров при num_classes=10.

--- resnet18_transfer -------------------------------------------------------

Свёрточная часть ResNet18 берётся как есть (веса ImageNet-1k) и замораживается,
её выход 512 признаков идёт в новый Linear(512 -> num_classes). Это «линейный
пробинг»: сеть не учит признаки, а только решает, как из готовых признаков
собрать наши классы.

Нормализация в dataset.py уже под ImageNet-статистику — ровно то, что нужно
этому варианту: предобученные слои калиброваны под это распределение входа.

Запуск для проверки: python -m src.model
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Один свёрточный блок: Conv -> BN -> ReLU -> MaxPool.

    bias=False у свёртки не опечатка: следом идёт BatchNorm со своим
    сдвигом (beta), и bias свёртки был бы ровно дублирующим параметром.
    Порядок Conv -> BN -> ReLU, а не Conv -> ReLU -> BN: нормализуем то,
    что ещё симметрично относительно нуля, до обрезки отрицательной части.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class SimpleCNN(nn.Module):
    """Четырёхблочная свёрточная сеть для классификации на num_classes классов.

    Args:
        num_classes: число выходов классификатора.
        channels: число каналов в каждом блоке; длина = число блоков.
        dropout: вероятность зануления перед последним полносвязным слоем.
        in_channels: каналов на входе (3 для RGB).
    """

    ARCH = "simple_cnn"  # пишется в чекпоинт, по нему load_checkpoint выбирает класс

    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.3,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels не может быть пустым: нужен хотя бы один блок")

        blocks = []
        prev = in_channels
        for width in channels:
            blocks.append(conv_block(prev, width))
            prev = width
        self.features = nn.Sequential(*blocks)

        # Global Average Pooling: карта признаков любого размера -> вектор из
        # channels[-1] чисел. Заменяет Flatten + большой Linear, см. пояснение
        # в docstring модуля и в README.
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming He init — под ReLU, чтобы дисперсия активаций не затухала.

        Дефолтная инициализация PyTorch рассчитана на симметричные нелинейности;
        ReLU обнуляет половину значений, и без поправки на это сигнал в глубину
        сети гаснет. Для сети, обучаемой с нуля, это заметно.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)  # логиты; softmax делает loss или инференс


# --- Transfer learning: ResNet18 -------------------------------------------


class TransferResNet18(nn.Module):
    """ResNet18 с весами ImageNet и новой головой под num_classes.

    Args:
        num_classes: число выходов классификатора.
        dropout: вероятность зануления перед новым Linear.
        freeze_backbone: заморозить свёрточную часть (линейный пробинг).
            False — полный fine-tuning, тогда нужен LR на порядок-два меньше.
        pretrained: грузить веса ImageNet. False имеет смысл только при
            восстановлении из чекпоинта — веса всё равно будут перезаписаны,
            и скачивать их из сети незачем.
    """

    ARCH = "resnet18_transfer"

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.resnet18(weights=weights)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.net.parameters():
                param.requires_grad = False

        # Голову ставим ПОСЛЕ заморозки: новый слой создаётся с
        # requires_grad=True и под цикл выше не попадает.
        in_features = self.net.fc.in_features  # 512 у ResNet18
        self.net.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        nn.init.normal_(self.net.fc[1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net.fc[1].bias)

    def train(self, mode: bool = True):
        """model.train() не должен размораживать статистики BatchNorm.

        Тонкий момент: requires_grad=False останавливает градиенты, но НЕ
        running_mean/running_var — их BatchNorm пересчитывает в train-режиме
        независимо от градиентов. Замороженный backbone при этом медленно
        уплывает под статистику нашего датасета, и предобученные признаки
        деградируют без единого шага оптимизатора. Поэтому при frozen=True
        BN-слои принудительно держим в eval.
        """
        super().train(mode)
        if mode and self.freeze_backbone:
            for module in self.net.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Отдельный no_grad для backbone не нужен: и вход, и его веса имеют
        # requires_grad=False, поэтому граф начинает строиться только с головы.
        return self.net(x)


# --- Фабрики ---------------------------------------------------------------


def build_model(num_classes: int, dropout: float = 0.3, **kwargs) -> SimpleCNN:
    """CNN с нуля. Её зовут train.py, evaluate.py и app.py."""
    return SimpleCNN(num_classes=num_classes, dropout=dropout, **kwargs)


def build_transfer_model(
    num_classes: int,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> TransferResNet18:
    """Предобученный ResNet18 с замороженным backbone и новой головой."""
    return TransferResNet18(
        num_classes=num_classes,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained,
    )


ARCHITECTURES = {
    SimpleCNN.ARCH: build_model,
    TransferResNet18.ARCH: build_transfer_model,
}


def build_from_arch(arch: str, num_classes: int, **kwargs) -> nn.Module:
    """Диспетчер по имени архитектуры — им пользуются train.py и load_checkpoint."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"Неизвестная архитектура {arch!r}. Доступны: {list(ARCHITECTURES)}")
    return ARCHITECTURES[arch](num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Число параметров. Для transfer-модели два ответа сильно расходятся:
    всего ~11.2M, обучаемых — 5 130."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# --- Чекпоинты -------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    classes: list[str],
    epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: dict | None = None,
) -> None:
    """Сохраняет веса вместе с именами классов.

    Имена классов лежат в чекпоинте намеренно: иначе evaluate.py и app.py
    зависят от того, что папка с данными на месте и порядок классов в ней
    не изменился. Чекпоинт должен быть самодостаточным.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        # Без имени архитектуры чекпоинт неоднозначен: веса ResNet18 легли бы
        # в SimpleCNN и упали на несовпадении ключей state_dict.
        "arch": getattr(model, "ARCH", SimpleCNN.ARCH),
        "classes": classes,
        "num_classes": len(classes),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def export_for_inference(
    src_path: str | Path, dst_path: str | Path, extra: dict | None = None
) -> tuple[int, int]:
    """Копия чекпоинта без состояния оптимизатора. Возвращает (было, стало) в байтах.

    AdamW хранит для каждого параметра два момента, поэтому при полном
    fine-tuning состояние оптимизатора весит вдвое больше самих весов:
    134 МБ файла против 45 МБ полезной нагрузки. Для продолжения обучения это
    нужно, для инференса и выкладки — мёртвый груз.

    extra дописывается в payload — так в выгруженный чекпоинт попадают поля,
    которых не было на момент обучения (например, имя прогона).
    """
    src_path, dst_path = Path(src_path), Path(dst_path)
    payload = torch.load(src_path, map_location="cpu", weights_only=False)
    payload.pop("optimizer_state", None)
    if extra:
        payload.update(extra)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst_path)
    return src_path.stat().st_size, dst_path.stat().st_size


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[nn.Module, list[str], dict]:
    """Восстанавливает модель из чекпоинта. Возвращает (model, classes, payload).

    Архитектуру берём из самого чекпоинта, поэтому вызывающему коду
    (evaluate.py, app.py) не нужно знать, чем обучали.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    classes = payload["classes"]
    arch = payload.get("arch", SimpleCNN.ARCH)  # старые чекпоинты без поля

    build_kwargs = {}
    if arch == TransferResNet18.ARCH:
        # Веса сейчас придут из чекпоинта — качать ImageNet-веса незачем.
        build_kwargs["pretrained"] = False

    model = build_from_arch(arch, num_classes=payload["num_classes"], **build_kwargs)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    return model, classes, payload


# --- Проверка --------------------------------------------------------------

if __name__ == "__main__":
    model = build_model(num_classes=10)
    print(model)
    print(f"\nОбучаемых параметров: {count_parameters(model):,}")

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        # Размеры карт признаков после каждого блока
        h = x
        for i, block in enumerate(model.features, start=1):
            h = block(h)
            print(f"после блока {i}: {tuple(h.shape)}")
        y = model(x)
    print(f"выход: {tuple(y.shape)}  (ожидается (2, 10))")

    # GAP делает сеть независимой от размера входа — проверяем на 160x160
    with torch.no_grad():
        print(f"вход 160x160 -> выход: {tuple(model(torch.randn(1, 3, 160, 160)).shape)}")

    # --- transfer learning ---
    # pretrained=False, чтобы проверка структуры не тянула 45 МБ весов из сети.
    print("\n--- resnet18_transfer ---")
    tm = build_transfer_model(num_classes=10, pretrained=False)
    total = count_parameters(tm, trainable_only=False)
    trainable = count_parameters(tm)
    print(f"параметров всего: {total:,}")
    print(f"из них обучаемых: {trainable:,}  ({trainable / total:.2%})")

    tm.train()
    bn_in_train = [n for n, m in tm.named_modules() if isinstance(m, nn.BatchNorm2d) and m.training]
    print(f"BatchNorm-слоёв в train-режиме после .train(): {len(bn_in_train)}  (ожидается 0)")
    print(f"голова обучается: {tm.net.fc[1].weight.requires_grad}  (ожидается True)")

    with torch.no_grad():
        print(f"выход: {tuple(tm(torch.randn(2, 3, 224, 224)).shape)}  (ожидается (2, 10))")
