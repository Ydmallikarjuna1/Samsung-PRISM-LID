import time

import torch

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
    DEVICE,
)
from models.lid_model import LIDModel
from training.dataloader import LIDDataLoader


def benchmark_data_loading(loader, num_batches=10):
    print("\n--- DATA PIPELINE BENCHMARK ---")

    start_time = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        if batch_index >= num_batches:
            break

        _ = batch["features"]
        _ = batch["labels"]

    elapsed = time.perf_counter() - start_time

    print(f"Batches tested : {num_batches}")
    print(f"Total time     : {elapsed:.3f} sec")
    print(
        f"Time / batch   : "
        f"{elapsed / num_batches:.3f} sec"
    )

    return elapsed


def benchmark_model(
    model,
    batch,
    device,
    num_iterations=10,
):
    print("\n--- MODEL BENCHMARK ---")

    features = batch["features"].to(device)
    labels = batch["labels"].to(device)

    model.train()

    # Warm-up
    logits, loss, embeddings = model(
        features,
        labels,
    )

    loss.backward()

    model.zero_grad(set_to_none=True)

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for _ in range(num_iterations):

        model.zero_grad(set_to_none=True)

        logits, loss, embeddings = model(
            features,
            labels,
        )

        loss.backward()

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    print(
        f"Iterations tested : "
        f"{num_iterations}"
    )

    print(
        f"Total time        : "
        f"{elapsed:.3f} sec"
    )

    print(
        f"Time / iteration  : "
        f"{elapsed / num_iterations:.3f} sec"
    )

    return elapsed


def main():

    print("=" * 60)
    print("TRAINING PIPELINE BENCHMARK")
    print("=" * 60)

    print(f"Device : {DEVICE}")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    data_builder = LIDDataLoader(
        manifest_path=TRAIN_MANIFEST,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_duration=MAX_DURATION,
        shuffle=False,
    )

    loader = data_builder.get_loader()

    print(
        f"Dataset size : "
        f"{len(data_builder.dataset)}"
    )

    # --------------------------------------------------------
    # Data pipeline benchmark
    # --------------------------------------------------------

    benchmark_data_loading(
        loader,
        num_batches=10,
    )

    # --------------------------------------------------------
    # Get one batch for model benchmark
    # --------------------------------------------------------

    batch = next(iter(loader))

    print(
        f"\nBenchmark batch shape : "
        f"{batch['features'].shape}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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
    # Model benchmark
    # --------------------------------------------------------

    benchmark_model(
        model=model,
        batch=batch,
        device=DEVICE,
        num_iterations=10,
    )

    print(
        "\nPIPELINE BENCHMARK COMPLETED"
    )


if __name__ == "__main__":
    main()