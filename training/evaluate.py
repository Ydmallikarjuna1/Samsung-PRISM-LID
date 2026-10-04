import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from configs.train_config import (
    VALID_MANIFEST,
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_DURATION,
    INPUT_DIM,
    CHANNELS,
    EMBEDDING_DIM,
    NUM_CLASSES,
    AAM_MARGIN,
    AAM_SCALE,
    DEVICE,
)
from models.lid_model import LIDModel
from training.dataloader import LIDDataLoader


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint


def evaluate(
    model,
    loader,
    device,
    max_batches=None,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        progress = tqdm(
            enumerate(loader),
            total=(
                min(max_batches, len(loader))
                if max_batches is not None
                else len(loader)
            ),
            desc="Validation",
        )

        for batch_index, batch in progress:

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            features = batch["features"].to(
                device
            )

            labels = batch["labels"].to(
                device
            )

            logits, loss, embeddings = model(
                features,
                labels,
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

            average_loss = (
                total_loss / total_samples
            )

            accuracy = (
                total_correct / total_samples
            )

            progress.set_postfix(
                loss=f"{average_loss:.4f}",
                accuracy=f"{accuracy:.4f}",
            )

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum validation batches.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ECAPA-TDNN VALIDATION")
    print("=" * 60)

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    print(f"Device     : {DEVICE}")

    # --------------------------------------------------------
    # Validation DataLoader
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    data_builder = LIDDataLoader(
        manifest_path=VALID_MANIFEST,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_duration=MAX_DURATION,
        shuffle=False,
    )

    loader = data_builder.get_loader()

    print(
        f"Validation samples : "
        f"{len(data_builder.dataset)}"
    )

    print(
        f"Batches            : "
        f"{len(loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating model...")

    model = LIDModel(
        input_dim=INPUT_DIM,
        channels=CHANNELS,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        margin=AAM_MARGIN,
        scale=AAM_SCALE,
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    print(
        f"\nLoading checkpoint:"
    )

    print(checkpoint_path)

    checkpoint = load_checkpoint(
        model,
        checkpoint_path,
        DEVICE,
    )

    print(
        f"Checkpoint epoch : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint loss  : "
        f"{checkpoint['loss']:.6f}"
    )

    print(
        f"Checkpoint acc   : "
        f"{checkpoint['accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nStarting validation...")

    loss, accuracy = evaluate(
        model=model,
        loader=loader,
        device=DEVICE,
        max_batches=args.max_batches,
    )

    print("\nValidation results:")

    print(
        f"Validation Loss     : "
        f"{loss:.6f}"
    )

    print(
        f"Validation Accuracy : "
        f"{accuracy:.4f}"
    )

    print("\nVALIDATION TEST COMPLETED")


if __name__ == "__main__":
    main()