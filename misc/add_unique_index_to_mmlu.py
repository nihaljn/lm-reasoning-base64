"""Read mmlu.csv and add a column corresponding to a unique index"""

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("/mnt/d/projects/encoded-prompting-llm/data/mmlu.csv")
    df["UniqueIndex"] = range(len(df))
    df.to_csv("/mnt/d/projects/encoded-prompting-llm/data/mmlu_with_index.csv", index=False)
