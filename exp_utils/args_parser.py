import argparse


def args_parser():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model_name", type=str, help="train config file basename")
    parser.add_argument(
        "--model",
        default="configs/model/seresnext50.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument("--batch_size", type=int, help="batch_size")
    parser.add_argument("--lr_default", type=float, help="lr_default")
    parser.add_argument("--lr_step_1", type=int, help="lr_step")
    parser.add_argument("--max_epochs", type=int, help="max_epochs")
    parser.add_argument("--data_version", type=str, help="data_version")
    parser.add_argument("--seed", type=int, help="data_version")
    parser.add_argument("--max_norm", type=float, help="max_nrom")

    args = parser.parse_args()
    return args
