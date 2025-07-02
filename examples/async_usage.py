"""Async usage examples for the Datasets Server Python client."""

import asyncio
from typing import List

from datasets_server import AsyncDatasetsServerClient


async def explore_dataset(dataset_name: str):
    """Explore a dataset using async client."""
    async with AsyncDatasetsServerClient() as client:
        # Check validity
        validity = await client.is_valid(dataset_name)
        print(f"Dataset '{dataset_name}' features:")
        print(f"  Preview: {validity.preview}")
        print(f"  Search: {validity.search}")
        print(f"  Statistics: {validity.statistics}")

        # Get splits
        splits = await client.list_splits(dataset_name)
        print(f"\nAvailable splits: {len(splits)}")
        for split in splits[:3]:  # Show first 3
            print(f"  - {split.config}/{split.split}")

        if splits and validity.preview:
            # Preview first split
            first_split = splits[0]
            preview = await client.preview(dataset_name, config=first_split.config, split=first_split.split)
            print(f"\nFirst row from {first_split.config}/{first_split.split}:")
            if preview.rows:
                row = preview.rows[0]["row"]
                for key, value in row.items():
                    value_str = str(value)[:100]
                    if len(str(value)) > 100:
                        value_str += "..."
                    print(f"  {key}: {value_str}")


async def search_multiple_datasets(datasets: List[str], query: str):
    """Search multiple datasets concurrently."""
    async with AsyncDatasetsServerClient() as client:
        print(f"\nSearching for '{query}' in multiple datasets...")

        # Create search tasks for all datasets
        tasks = []
        for dataset in datasets:
            # First check if dataset is valid and has search
            validity = await client.is_valid(dataset)
            if validity.search:
                # Get first split
                splits = await client.list_splits(dataset)
                if splits:
                    first_split = splits[0]
                    task = client.search(
                        dataset=dataset,
                        query=query,
                        config=first_split.config,
                        split=first_split.split,
                        length=3,
                    )
                    tasks.append((dataset, task))

        if not tasks:
            print("No datasets support search!")
            return

        # Execute all searches concurrently
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Display results
        for (dataset, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                print(f"\nError searching {dataset}: {result}")
            else:
                print(f"\nResults from {dataset} ({result.num_rows_total} total matches):")
                for i, row in enumerate(result.rows[:2]):  # Show first 2
                    print(f"  Match {i + 1}:")
                    # Show a sample of the row data
                    row_data = row["row"]
                    for key, value in list(row_data.items())[:2]:  # Show first 2 fields
                        value_str = str(value)[:80]
                        if len(str(value)) > 80:
                            value_str += "..."
                        print(f"    {key}: {value_str}")


async def concurrent_dataset_info(datasets: List[str]):
    """Get information about multiple datasets concurrently."""
    async with AsyncDatasetsServerClient() as client:
        print("\nFetching dataset information concurrently...")

        # Create tasks to get info for all datasets
        tasks = [(dataset, client.get_info(dataset)) for dataset in datasets]

        # Execute all requests concurrently
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Display results
        for (dataset, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                print(f"\n{dataset}: Error - {result}")
            else:
                info = result.dataset_info
                print(f"\n{dataset}:")
                print(f"  Description: {info.get('description', 'N/A')[:100]}...")
                if "features" in info:
                    print(f"  Features: {len(info['features'])} columns")
                if "splits" in info:
                    total_examples = sum(split.get("num_examples", 0) for split in info["splits"].values())
                    print(f"  Total examples: {total_examples:,}")


async def main():
    """Run async examples."""
    # Example 1: Explore a single dataset
    print("=== Example 1: Exploring a single dataset ===")
    await explore_dataset("squad")

    # Example 2: Search multiple datasets
    print("\n\n=== Example 2: Searching multiple datasets ===")
    search_datasets = ["squad", "glue", "imdb"]
    await search_multiple_datasets(search_datasets, "what")

    # Example 3: Get info from multiple datasets concurrently
    print("\n\n=== Example 3: Concurrent dataset information ===")
    info_datasets = ["squad", "glue", "mnist"]
    await concurrent_dataset_info(info_datasets)


if __name__ == "__main__":
    # Run the async examples
    asyncio.run(main())
