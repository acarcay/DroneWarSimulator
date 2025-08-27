import json
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd

def save_report(old_results, new_results, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # DataFrame birleştir
    df_old = pd.DataFrame({"Drone": old_results["drone"], "Path_old": old_results["path"]})
    df_new = pd.DataFrame({"Drone": new_results["drone"], "Path_new": new_results["path"]})
    df = pd.merge(df_old, df_new, on="Drone")

    # Grafik 1: Paths
    plt.figure(figsize=(8,5))
    plt.plot(df["Drone"], df["Path_old"], marker="o", label="Old paths")
    plt.plot(df["Drone"], df["Path_new"], marker="s", label="New paths")
    plt.title("Drone Path Lengths Comparison")
    plt.xlabel("Drone ID")
    plt.ylabel("Path length (m)")
    plt.legend()
    plt.grid(True)
    path_img = os.path.join(out_dir, f"{timestamp}_paths.png")
    plt.savefig(path_img)
    plt.close()

    # Grafik 2: Metrics
    labels = ["Mean RMSE (m)", "Settling Time (s)"]
    old_vals = [old_results["mean_rmse"], old_results["settling_time"]]
    new_vals = [new_results["mean_rmse"], new_results["settling_time"]]
    x = range(len(labels))

    plt.figure(figsize=(7,5))
    plt.bar([i-0.2 for i in x], old_vals, width=0.4, label="Old")
    plt.bar([i+0.2 for i in x], new_vals, width=0.4, label="New")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title("RMSE & Settling Time Comparison")
    plt.legend()
    plt.grid(axis="y")
    metrics_img = os.path.join(out_dir, f"{timestamp}_metrics.png")
    plt.savefig(metrics_img)
    plt.close()

    # JSON summary
    summary = {
        "timestamp": timestamp,
        "old_results": old_results,
        "new_results": new_results,
        "files": {"paths": path_img, "metrics": metrics_img}
    }
    json_path = os.path.join(out_dir, f"{timestamp}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Basit HTML rapor
    html_path = os.path.join(out_dir, f"{timestamp}_report.html")
    with open(html_path, "w") as f:
        f.write(f"""
        <html><head><title>Simulation Report {timestamp}</title></head>
        <body>
        <h1>Simulation Report</h1>
        <p><b>Old Mean RMSE:</b> {old_results['mean_rmse']} | <b>New:</b> {new_results['mean_rmse']}</p>
        <p><b>Old Settling Time:</b> {old_results['settling_time']} | <b>New:</b> {new_results['settling_time']}</p>
        <h2>Path Comparison</h2>
        <img src="{os.path.basename(path_img)}" width="600">
        <h2>Metrics Comparison</h2>
        <img src="{os.path.basename(metrics_img)}" width="600">
        </body></html>
        """)
    return {"json": json_path, "html": html_path}
