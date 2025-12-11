#!/usr/bin/env python3
"""Demonstration of the sample_rows functionality."""

import asyncio
from datasets_server import DatasetsServerClient, AsyncDatasetsServerClient


def sync_sampling_examples():
    """Demonstrate synchronous random sampling."""
    print("=== Synchronous Client Examples ===\n")

    # Initialize client
    client = DatasetsServerClient()

    # Example 1: Basic random sampling
    print("1. Basic random sampling from IMDB dataset:")
    samples = client.sample_rows("stanfordnlp/imdb", "plain_text", "train", n_samples=100, max_requests=2)
    print(samples)
    for i, row in enumerate(samples.rows):
        text = row["row"]["text"][:100] + "..." if len(row["row"]["text"]) > 100 else row["row"]["text"]
        label = row["row"]["label"]
        print(f"   Sample {i + 1}: [{label}] {text}")
    print(f"   Total dataset size: {samples.num_rows_total} rows\n")

    # Example 2: Reproducible sampling with seed
    print("2. Reproducible sampling with seed=42:")
    samples1 = client.sample_rows("stanfordnlp/imdb", "plain_text", "train", n_samples=2, seed=42)
    samples2 = client.sample_rows("stanfordnlp/imdb", "plain_text", "train", n_samples=2, seed=42)

    # Extract text from first sample of each
    text1 = samples1.rows[0]["row"]["text"][:50] + "..." if samples1.rows else "N/A"
    text2 = samples2.rows[0]["row"]["text"][:50] + "..." if samples2.rows else "N/A"

    print(f"   First call, sample 1 text: {text1}")
    print(f"   Second call, sample 1 text: {text2}")
    print(f"   Same results: {text1 == text2}\n")

    # Example 3: Sampling from smaller dataset
    print("3. Sampling from smaller dataset (iris):")
    try:
        # First, let's check available configs and splits
        splits = client.list_splits("scikit-learn/iris")
        if splits:
            config = splits[0].config
            split = splits[0].split
            print(f"   Using config='{config}', split='{split}'")

            samples = client.sample_rows("scikit-learn/iris", config, split, n_samples=5, seed=123)
            print(f"   Sampled {len(samples.rows)} rows from {samples.num_rows_total} total rows")

            # Show feature names
            if samples.features:
                feature_names = [f["name"] for f in samples.features]
                print(f"   Features: {', '.join(feature_names)}")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # Example 4: API-efficient sampling with max_requests
    print("4. API-efficient sampling with max_requests:")

    # Sample from large dataset with limited API calls
    print("   Sampling from large dataset with max_requests=1 (single API call):")
    efficient_samples = client.sample_rows(
        "stanfordnlp/imdb", "plain_text", "train", n_samples=5, seed=42, max_requests=1
    )
    print(f"   Got {len(efficient_samples.rows)} samples using only 1 API call")
    print("   This is less random but more API-friendly for large datasets\n")

    # Compare with default behavior
    print("   Comparing max_requests strategies:")

    # With max_requests=1
    samples_1_req = client.sample_rows(
        "stanfordnlp/imdb", "plain_text", "train", n_samples=10, seed=42, max_requests=1
    )

    # With max_requests=3
    samples_3_req = client.sample_rows(
        "stanfordnlp/imdb", "plain_text", "train", n_samples=10, seed=42, max_requests=3
    )

    # Default (no limit)
    samples_default = client.sample_rows("stanfordnlp/imdb", "plain_text", "train", n_samples=10, seed=42)

    print(f"   max_requests=1: All samples from single batch (less random)")
    print(f"   max_requests=3: Samples from up to 3 different parts of dataset")
    print(f"   max_requests=None: True random sampling (may use many API calls)")
    print()

    # Example 5: Edge cases
    print("5. Edge cases:")

    # Zero samples
    empty = client.sample_rows("stanfordnlp/imdb", "plain_text", "test", n_samples=0)
    print(f"   Zero samples requested: {len(empty.rows)} rows returned")

    # Error handling - requesting too many samples
    try:
        # First get the actual size
        size_info = client.get_rows("stanfordnlp/imdb", "plain_text", "test", offset=0, length=1)
        total_rows = size_info.num_rows_total
        print(f"   Dataset has {total_rows} rows")

        # Try to sample more than available
        client.sample_rows("stanfordnlp/imdb", "plain_text", "test", n_samples=total_rows + 100)
    except ValueError as e:
        print(f"   Error when requesting too many samples: {e}")


async def async_sampling_examples():
    """Demonstrate asynchronous random sampling."""
    print("\n=== Asynchronous Client Examples ===\n")

    # Initialize async client
    async with AsyncDatasetsServerClient() as client:
        # Example 1: Concurrent sampling from multiple datasets
        print("1. Concurrent sampling from multiple datasets:")

        # Sample from multiple datasets concurrently
        tasks = [
            client.sample_rows("stanfordnlp/imdb", "plain_text", "train", n_samples=2, seed=42),
            client.sample_rows("fka/awesome-chatgpt-prompts", "plain_text", "train", n_samples=2, seed=42),
            client.sample_rows("SetFit/ag_news", "default", "train", n_samples=2, seed=42),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        dataset_names = ["stanfordnlp/imdb", "fka/awesome-chatgpt-prompts", "SetFit/ag_news"]
        for name, result in zip(dataset_names, results):
            if isinstance(result, Exception):
                print(f"   {name}: Error - {result}")
            else:
                print(f"   {name}: Sampled {len(result.rows)} rows from {result.num_rows_total} total")

        print()

        # Example 2: API-efficient sampling with max_requests
        print("2. API-efficient async sampling with max_requests:")

        # Sample from multiple large datasets with limited API calls
        tasks = [
            client.sample_rows("stanfordnlp/imdb", "plain_text", "train", n_samples=5, seed=42, max_requests=1),
            client.sample_rows("SetFit/ag_news", "default", "train", n_samples=5, seed=42, max_requests=2),
            client.sample_rows("davanstrien/haiku_dpo", "default", "train", n_samples=5, seed=42, max_requests=3),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        configs = [
            ("stanfordnlp/imdb", "max_requests=1"),
            ("SetFit/ag_news", "max_requests=2"),
            ("davanstrien/haiku_dpo", "max_requests=3"),
        ]

        for (dataset, config), result in zip(configs, results):
            if isinstance(result, Exception):
                print(f"   {dataset} ({config}): Error - {result}")
            else:
                print(f"   {dataset} ({config}): Got {len(result.rows)} samples")

        print()

        # Example 3: Sampling with different seeds
        print("3. Multiple samples with different seeds:")
        seeds = [42, 123, 456]
        tasks = [
            client.sample_rows("stanfordnlp/imdb", "plain_text", "train", n_samples=1, seed=seed) for seed in seeds
        ]

        results = await asyncio.gather(*tasks)

        for seed, result in zip(seeds, results):
            if result.rows:
                label = result.rows[0]["row"]["label"]
                text_preview = result.rows[0]["row"]["text"][:50] + "..."
                print(f"   Seed {seed}: [{label}] {text_preview}")


def main():
    """Run all examples."""
    # Run synchronous examples
    sync_sampling_examples()

    # Run asynchronous examples
    asyncio.run(async_sampling_examples())

    print("\n=== Summary ===")
    print("The sample_rows method provides:")
    print("- Random sampling from any dataset/config/split")
    print("- Reproducible results with optional seed parameter")
    print("- API-efficient sampling with max_requests parameter")
    print("- Trade-off control: true randomness vs API efficiency")
    print("- Efficient batching for optimal API usage")
    print("- Support for both sync and async operations")
    print("- Proper error handling for edge cases")


if __name__ == "__main__":
    main()
