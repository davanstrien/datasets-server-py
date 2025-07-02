"""Examples of searching and filtering datasets."""

from datasets_server import DatasetsServerClient


def search_example():
    """Demonstrate dataset search functionality."""
    client = DatasetsServerClient()
    dataset = "stanfordnlp/imdb"

    # First, check if search is available
    validity = client.is_valid(dataset)
    if not validity.search:
        print(f"Search is not available for dataset '{dataset}'")
        return

    # Get splits to search
    splits = client.list_splits(dataset)
    if not splits:
        print("No splits found!")
        return

    # Use the first split
    first_split = splits[0]
    config = first_split.config
    split = first_split.split

    # Search for a query
    query = "artificial intelligence"
    print(f"Searching for '{query}' in {dataset}/{config}/{split}...")

    results = client.search(
        dataset=dataset,
        query=query,
        config=config,
        split=split,
        length=5,  # Get 5 results
    )

    print(f"\nFound {results.num_rows_total} total matches")
    print(f"Showing first {len(results.rows)} results:\n")

    for i, row in enumerate(results.rows):
        print(f"Result {i + 1}:")
        row_data = row["row"]
        # Display relevant fields (adjust based on dataset structure)
        for key, value in row_data.items():
            if isinstance(value, str) and len(value) > 200:
                value = f"{value[:200]}..."
            print(f"  {key}: {value}")
        print()


def filter_example():
    """Demonstrate dataset filtering functionality."""
    client = DatasetsServerClient()
    dataset = "stanfordnlp/imdb"

    # Check if filter is available
    validity = client.is_valid(dataset)
    if not validity.filter:
        print(f"Filter is not available for dataset '{dataset}'")
        return

    # Get splits
    splits = client.list_splits(dataset)
    if not splits:
        print("No splits found!")
        return

    # Use validation split if available, otherwise first split
    split_to_use = None
    for split in splits:
        if split.split == "validation":
            split_to_use = split
            break
    if not split_to_use:
        split_to_use = splits[0]

    config = split_to_use.config
    split = split_to_use.split

    print(f"Filtering {dataset}/{config}/{split}...")

    # Example filter: questions longer than 50 characters
    # Note: Column names must be in double quotes for SQL compatibility
    where_clause = 'LENGTH("question") > 50'

    try:
        filtered = client.filter(
            dataset=dataset,
            config=config,
            split=split,
            where=where_clause,
            length=5,
        )

        print(f"\nFilter: {where_clause}")
        print(f"Found {filtered.num_rows_total} matching rows")
        print(f"Showing first {len(filtered.rows)} results:\n")

        for i, row in enumerate(filtered.rows):
            row_data = row["row"]
            print(f"Row {i + 1}:")
            # Show the question field and its length
            if "question" in row_data:
                question = row_data["question"]
                print(f"  Question ({len(question)} chars): {question}")
            # Show other fields briefly
            for key, value in row_data.items():
                if key != "question":
                    value_str = str(value)[:100]
                    if len(str(value)) > 100:
                        value_str += "..."
                    print(f"  {key}: {value_str}")
            print()

    except Exception as e:
        print(f"Filter error: {e}")
        print("Note: Filter requires column names in double quotes and valid SQL syntax")


def statistics_example():
    """Demonstrate dataset statistics functionality."""
    client = DatasetsServerClient()
    dataset = "stanfordnlp/imdb"

    # Check if statistics are available
    validity = client.is_valid(dataset)
    if not validity.statistics:
        print(f"Statistics are not available for dataset '{dataset}'")
        return

    # Get splits
    splits = client.list_splits(dataset)
    if not splits:
        print("No splits found!")
        return

    # Use the first split
    first_split = splits[0]
    config = first_split.config
    split = first_split.split

    print(f"Getting statistics for {dataset}/{config}/{split}...")

    stats = client.get_statistics(
        dataset=dataset,
        config=config,
        split=split,
    )

    print("\nDataset statistics:")
    print(f"Number of examples: {stats.num_examples:,}")

    if stats.statistics:
        print(f"\nColumn statistics ({len(stats.statistics)} columns):")
        for col_stat in stats.statistics[:5]:  # Show first 5 columns
            print(f"\n  Column: {col_stat.get('column_name', 'unknown')}")
            print(f"  Type: {col_stat.get('column_type', 'unknown')}")

            # Display different statistics based on column type
            if "nan_count" in col_stat:
                print(f"  NaN count: {col_stat['nan_count']}")
            if "min" in col_stat and "max" in col_stat:
                print(f"  Range: [{col_stat['min']}, {col_stat['max']}]")
            if "mean" in col_stat:
                print(f"  Mean: {col_stat['mean']:.2f}")
            if "histogram" in col_stat:
                hist = col_stat["histogram"]
                if "bins" in hist:
                    print(f"  Histogram bins: {len(hist['bins'])}")


def pagination_example():
    """Demonstrate pagination through dataset rows."""
    client = DatasetsServerClient()
    dataset = "stanfordnlp/imdb"

    # Get splits
    splits = client.list_splits(dataset)
    if not splits:
        print("No splits found!")
        return

    first_split = splits[0]
    config = first_split.config
    split = first_split.split

    print(f"Paginating through {dataset}/{config}/{split}...")

    # Get rows in pages
    page_size = 10
    total_pages = 3  # Get 3 pages

    for page in range(total_pages):
        offset = page * page_size
        print(f"\nPage {page + 1} (rows {offset}-{offset + page_size - 1}):")

        rows = client.get_rows(
            dataset=dataset,
            config=config,
            split=split,
            offset=offset,
            length=page_size,
        )

        print(f"Retrieved {len(rows.rows)} rows")
        print(f"Total rows in dataset: {rows.num_rows_total}")

        # Show first row of each page
        if rows.rows:
            first_row = rows.rows[0]["row"]
            print("First row sample:")
            for key, value in list(first_row.items())[:3]:  # Show first 3 fields
                value_str = str(value)[:80]
                if len(str(value)) > 80:
                    value_str += "..."
                print(f"  {key}: {value_str}")


def main():
    """Run all examples."""
    print("=== Search Example ===")
    search_example()

    print("\n\n=== Filter Example ===")
    filter_example()

    print("\n\n=== Statistics Example ===")
    statistics_example()

    print("\n\n=== Pagination Example ===")
    pagination_example()


if __name__ == "__main__":
    main()
