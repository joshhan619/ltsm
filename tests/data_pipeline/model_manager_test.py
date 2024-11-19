import pytest
import argparse
import torch
from ltsm.data_pipeline import ModelManager 

@pytest.fixture
def mock_args():
    return argparse.Namespace(
        output_dir="./output",
        batch_size=8,
        train_epochs=3,
        learning_rate=5e-5,
        gradient_accumulation_steps=1,
        eval=False,
        lora=False,
        freeze=False,
        tmax=50,
        lora_dim=16
    )

@pytest.fixture
def model_manager(mock_args, mocker):
    mocker.patch("ltsm.models.get_model")
    return ModelManager(mock_args)

def test_print_trainable_parameters(model_manager, mocker):
    mock_model = mocker.MagicMock()
    mock_model.named_parameters.return_value = [("param1", torch.nn.Parameter(torch.randn(2, 2, requires_grad=True)))]
    model_manager.model = mock_model
    mock_print = mocker.patch("builtins.print")

    model_manager.print_trainable_parameters()
    mock_print.assert_called_once_with("param1 is trainable...")

def test_freeze_parameters(model_manager, mocker):
    mock_model = mocker.MagicMock()
    mock_model.named_parameters.return_value = [
        ("gpt2.weight", torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))),
        ("ln.weight", torch.nn.Parameter(torch.randn(2, 2, requires_grad=True)))
    ]
    model_manager.model = mock_model
    model_manager.freeze_parameters()
    
    assert mock_model.named_parameters()[0][1].requires_grad is False
    assert mock_model.named_parameters()[1][1].requires_grad is True

def test_create_model_lora_enabled(model_manager, mocker):
    model_manager.args.lora = True
    mocker.patch("ltsm.data_pipeline.model_manager.get_model", return_value=None)
    mock_get_peft_model = mocker.patch("ltsm.data_pipeline.model_manager.get_peft_model")

    model = model_manager.create_model()
    mock_get_peft_model.assert_called_once
