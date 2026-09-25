import csv
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent
CSV_FILE = RESULTS_DIR / "final_results.csv"


def load_results():
    with CSV_FILE.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))

def generate_graphs(results):
    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    # -------------------------------
    # Graph 1: Overall performance
    # -------------------------------
    labels = []
    times = []

    for row in results:
        if row["implementation"] == "cpu":
            labels.append("CPU")
            times.append(float(row["average_ms"]))

        elif (
            row["implementation"] == "naive_gpu"
            and row["matrix_size"] == "1024x1024"
        ):
            labels.append("Naive GPU\n8x8")
            times.append(float(row["average_ms"]))

        elif (
            row["implementation"] == "tiled_gpu"
            and row["tile_size"] == "8"
        ):
            labels.append("Tiled GPU\n8x8")
            times.append(float(row["average_ms"]))

    plt.figure(figsize=(8, 5))
    plt.bar(labels, times)
    plt.ylabel("Average Kernel Time (ms)")
    plt.title("Matrix Multiplication Performance")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "overall_performance.png",
        dpi=200
    )
    plt.close()

    # -------------------------------
    # Graph 2: Tile-size sweep
    # -------------------------------
    tile_sizes = []
    tile_times = []

    for row in results:
        if row["implementation"] == "tiled_gpu":
            tile_sizes.append(
                int(row["tile_size"])
            )
            tile_times.append(
                float(row["average_ms"])
            )

    plt.figure(figsize=(8, 5))
    plt.plot(
        tile_sizes,
        tile_times,
        marker="o"
    )
    plt.xlabel("Tile Size")
    plt.ylabel("Average Kernel Time (ms)")
    plt.title("Tile Size vs Kernel Execution Time")
    plt.xticks(tile_sizes)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "tile_size_sweep.png",
        dpi=200
    )
    plt.close()

    print("\nGraphs generated:")
    print(
        figures_dir / "overall_performance.png"
    )
    print(
        figures_dir / "tile_size_sweep.png"
    )

def main():
    results = load_results()

    cpu = next(
        row for row in results
        if row["implementation"] == "cpu"
    )

    naive = next(
        row for row in results
        if row["implementation"] == "naive_gpu"
    )

    tiled_8 = next(
        row for row in results
        if row["implementation"] == "tiled_gpu"
        and row["tile_size"] == "8"
    )

    cpu_ms = float(cpu["average_ms"])
    naive_ms = float(naive["average_ms"])
    tiled_ms = float(tiled_8["average_ms"])

    cpu_to_naive = cpu_ms / naive_ms
    cpu_to_tiled = cpu_ms / tiled_ms
    naive_to_tiled = naive_ms / tiled_ms

    tiled_improvement = (
        (naive_ms - tiled_ms) / naive_ms
    ) * 100

    print("\n========== PERFORMANCE ANALYSIS ==========\n")

    print(f"CPU average time:        {cpu_ms:.3f} ms")
    print(f"Naive GPU average time:  {naive_ms:.3f} ms")
    print(f"Tiled GPU average time:  {tiled_ms:.3f} ms")

    print("\nSpeedups:")
    print(f"CPU -> Naive GPU:        {cpu_to_naive:.2f}x")
    print(f"CPU -> Tiled GPU:        {cpu_to_tiled:.2f}x")
    print(f"Naive -> Tiled GPU:      {naive_to_tiled:.2f}x")

    print(
        f"\nTiled improvement over naive: "
        f"{tiled_improvement:.2f}%"
    )

    print("\nTile-size results:")

    for row in results:
        if row["implementation"] == "tiled_gpu":
            print(
                f"Tile {row['tile_size']}x{row['tile_size']}: "
                f"{float(row['average_ms']):.3f} ms"
            )
    generate_graphs(results)

    print("\n==========================================\n")


if __name__ == "__main__":
    main()