import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from bench import run_rust_core_benchmark


def main():
    scenarios = [
        {"document_count": 100, "query_count": 10, "top_k": 3},
        {"document_count": 1000, "query_count": 25, "top_k": 5},
    ]

    for scenario in scenarios:
        result = run_rust_core_benchmark(**scenario)
        print(result)


if __name__ == "__main__":
    main()
