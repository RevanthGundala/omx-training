"""
inspect_rrd.py — Read and summarize a Rerun .rrd log file.

Usage:
    uv run inspect_rrd.py                          # reads latest file in outputs/rerun/
    uv run inspect_rrd.py outputs/rerun/omx_eval.rrd
"""

import sys
from pathlib import Path

import rerun.dataframe as rrdf


def inspect(rrd_path: Path) -> None:
    print(f"Loading: {rrd_path}")
    recording = rrdf.load_recording(str(rrd_path))

    schema = recording.schema()
    print(f"\n=== Schema ({len(schema.component_columns)} component columns) ===")

    # Group columns by entity path
    entities: dict[str, list[str]] = {}
    for col in schema.component_columns:
        entity = col.entity_path
        component = col.component_name
        entities.setdefault(entity, []).append(component)

    for entity, components in sorted(entities.items()):
        print(f"\n  {entity}")
        for c in components:
            print(f"    - {c}")

    print(f"\nIndex columns: {[c.name for c in schema.index_columns]}")

    # Read all data as a view
    view = recording.view(index="step", contents="/**")
    table = view.select().read_all()
    print(f"\nTotal rows: {len(table)}")

    # Show sample of scalar data (joints, metrics)
    print("\n=== Sample Data (first & last 3 rows) ===")
    # Convert to pandas for easy display
    try:
        import pandas as pd
        df = table.to_pandas()
        # Filter to scalar columns only (skip images, metadata)
        scalar_cols = [c for c in df.columns if df[c].dtype in ("float64", "float32", "int64", "int32")]
        if scalar_cols:
            display_df = df[scalar_cols]
            print(display_df.head(3).to_string())
            print("...")
            print(display_df.tail(3).to_string())
        else:
            print("(No scalar columns found — try viewing in Rerun Viewer)")
    except ImportError:
        print("(Install pandas for tabular display: pip install pandas)")
        print(f"Columns: {table.column_names}")


def main():
    if len(sys.argv) > 1:
        rrd_path = Path(sys.argv[1])
    else:
        rrd_dir = Path("outputs/rerun")
        if not rrd_dir.exists():
            print(f"No rerun logs found at {rrd_dir}/")
            print("Run eval.py or replay.py first to generate logs.")
            sys.exit(1)
        rrds = sorted(rrd_dir.glob("*.rrd"), key=lambda p: p.stat().st_mtime)
        if not rrds:
            print(f"No .rrd files in {rrd_dir}/")
            sys.exit(1)
        rrd_path = rrds[-1]  # most recent

    if not rrd_path.exists():
        print(f"File not found: {rrd_path}")
        sys.exit(1)

    inspect(rrd_path)


if __name__ == "__main__":
    main()
