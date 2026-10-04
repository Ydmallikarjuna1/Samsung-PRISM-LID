import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from configs.train_config import (
    TRAIN_MANIFEST,
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_DURATION,
    INPUT_DIM,
    CHANNELS,
    EMBEDDING_DIM,
    NUM_CLASSES,
    AAM_MARGIN,
    AAM_SCALE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    GRADIENT_CLIP,
    CHECKPOINT_DIR,
)
from models.lid_model import LIDModel
from training.dataloader import LIDDataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    max_batches=None,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        enumerate(loader),
        total=(
            min(max_batches, len(loader))
            if max_batches is not None
            else len(loader)
        ),
        desc="Training",
    )

    for batch_index, batch in progress:

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits, loss, embeddings = model(
            features,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

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

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    accuracy,
):
    checkpoint_dir = Path(
        CHECKPOINT_DIR
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / f"ecapa_epoch_{epoch}.pt"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
        },
        checkpoint_path,
    )

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches for a smoke test.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to run.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ECAPA-TDNN TRAINING")
    print("=" * 60)

    set_seed()

    print(f"Device : {DEVICE}")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    data_builder = LIDDataLoader(
        manifest_path=TRAIN_MANIFEST,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_duration=MAX_DURATION,
        shuffle=True,
    )

    loader = data_builder.get_loader()

    print(
        f"Training samples : "
        f"{len(data_builder.dataset)}"
    )

    print(
        f"Batches per epoch : "
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

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting training...")

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        print(
            f"\nEpoch {epoch}/{args.epochs}"
        )

        loss, accuracy = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=DEVICE,
            max_batches=args.max_batches,
        )

        print(
            f"\nEpoch {epoch} results:"
        )

        print(
            f"Loss     : {loss:.6f}"
        )

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
        )

        print(
            f"Checkpoint saved to:"
        )

        print(checkpoint_path)

    print("\nTRAINING TEST COMPLETED")


if __name__ == "__main__":
    main()