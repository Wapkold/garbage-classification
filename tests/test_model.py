"""Архитектуры: форма выхода, число параметров, заморозка backbone."""
import torch
import torch.nn as nn

from src.model import SimpleCNN, TransferResNet18, build_from_arch, count_parameters

NUM_CLASSES = 10


def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=NUM_CLASSES).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, NUM_CLASSES)


def test_simple_cnn_parameter_count_matches_readme():
    assert count_parameters(SimpleCNN(num_classes=NUM_CLASSES)) == 391_466


def test_simple_cnn_accepts_other_input_sizes():
    # Global Average Pooling: размер входа не зашит в голову сети
    model = SimpleCNN(num_classes=NUM_CLASSES).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 160, 200)).shape == (1, NUM_CLASSES)


def test_transfer_frozen_trains_only_head():
    model = TransferResNet18(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
    assert count_parameters(model) == 5_130
    assert count_parameters(model, trainable_only=False) == 11_181_642


def test_transfer_unfrozen_trains_everything():
    model = TransferResNet18(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=False)
    assert count_parameters(model) == 11_181_642


def test_frozen_backbone_keeps_batchnorm_in_eval():
    # requires_grad=False не останавливает running-статистики BN, поэтому
    # .train() обязан оставить их в eval, иначе backbone «уплывает»
    model = TransferResNet18(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
    model.train()
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bns and all(not bn.training for bn in bns)

    before = bns[0].running_mean.clone()
    model(torch.randn(4, 3, 224, 224))
    assert torch.equal(bns[0].running_mean, before)


def test_build_from_arch_dispatch():
    assert isinstance(build_from_arch("simple_cnn", NUM_CLASSES), SimpleCNN)
    model = build_from_arch("resnet18_transfer", NUM_CLASSES, pretrained=False)
    assert isinstance(model, TransferResNet18)
