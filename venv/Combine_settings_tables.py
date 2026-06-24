import os
import pandas as pd

files = {
    1: os.path.join("data", "subspace", "setting1", "final_results.csv"),
    2: os.path.join("data", "subspace", "setting2", "final_results.csv"),
    3: os.path.join("data", "subspace", "setting3", "final_results.csv"),
    4: os.path.join("data", "subspace", "setting4", "final_results.csv"),
}

all_dfs = []

for setting, path in files.items():
    print(f"Reading: {path}")
    df = pd.read_csv(path)
    df["setting"] = setting
    all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True)
final_df = final_df[
    ["setting", "group", "method", "split", "stationary_error", "nonstationary_error"]
]

final_df = final_df.sort_values(
    ["setting", "group", "method", "split"]
).reset_index(drop=True)

print(final_df)

final_df.to_csv("all_settings_results.csv", index=False)
print("\nSaved as all_settings_results.csv")