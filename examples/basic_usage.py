"""Basic usage examples for the Datasets Server Python client."""

from datasets_server import DatasetsServerClient


def main():
    """Demonstrate basic usage of the sync client."""
    # Initialize client
    client = DatasetsServerClient()

    # Check if dataset is valid
    dataset = "stanfordnlp/imdb"
    print(f"Checking dataset: {dataset}")
    validity = client.is_valid(dataset)
    print(f"Dataset '{dataset}' validity:")
    print(f"  Preview available: {validity.preview}")
    print(f"  Search available: {validity.search}")
    print(f"  Filter available: {validity.filter}")
    print(f"  Statistics available: {validity.statistics}")
    print()

    # List available splits
    splits = client.list_splits(dataset)
    print(f"Available splits for '{dataset}':")
    for split in splits:
        print(f"  - Config: {split.config}, Split: {split.split}")
    print()

    # Preview first rows
    if validity.preview and splits:
        first_split = splits[0]
        print(f"Previewing {first_split.config}/{first_split.split}:")
        preview = client.preview(dataset, config=first_split.config, split=first_split.split)

        # Show features
        print("Features:")
        for feature in preview.features[:3]:
            print(f"  - {feature}")

        # Show first few rows
        print("\nFirst 3 rows:")
        for i, row in enumerate(preview.rows[:3]):
            print(f"\nRow {i}:")
            for key, value in row["row"].items():
                # Truncate long values for display
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"  {key}: {value_str}")

    # Get dataset size
    print("\nDataset size information:")
    size_info = client.get_size(dataset)
    if size_info.size and "dataset" in size_info.size:
        ds_size = size_info.size["dataset"]
        print(f"  Total size: {ds_size.get('num_bytes_original_files', 0):,} bytes")
        print(f"  Parquet size: {ds_size.get('num_bytes_parquet_files', 0):,} bytes")
        print(f"  Total rows: {ds_size.get('num_rows', 0):,}")

    # List Parquet files
    print("\nParquet files (first 3):")
    parquet_files = client.list_parquet_files(dataset)
    for pf in parquet_files[:3]:
        print(f"  - {pf.filename} ({pf.size:,} bytes)")
        print(f"    Config: {pf.config}, Split: {pf.split}")


if __name__ == "__main__":
    main()
