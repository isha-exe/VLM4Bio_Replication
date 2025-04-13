import pandas as pd
from functools import reduce

DATASET_DIR = "/raid/gurukul/vlm4bio/Dataset/task1/10K"

# All task-specific files (change to fish.csv / butterfly.csv as needed)
files = {
    "main": "bird.csv",
    "cot": "cot/bird_cot.csv",
    "dense": "dense/bird_dense.csv",
    "fct": "fct/bird_fct.csv",
    "nota": "nota/bird_nota.csv",
}


columns_to_keep = {
    "main": ["fileNameAsDelivered", "scientificName", "MCQ_Prompt"],
    "cot": ["fileNameAsDelivered", "option_A_reason", "option_B_reason", "option_C_reason", "option_D_reason"],
    "dense": ["fileNameAsDelivered", "Dense Caption"],
    "fct": ["fileNameAsDelivered", "Answer"],
    "nota": ["fileNameAsDelivered", "NOTA_Prompt"],
}


dfs = []
for name, file in files.items():
    path = f"{DATASET_DIR}/{file}"
    df = pd.read_csv(path)
    df = df[columns_to_keep[name]]  
    dfs.append(df)


merged_df = reduce(lambda left, right: pd.merge(left, right, on="fileNameAsDelivered", how="outer"), dfs)


merged_df.rename(columns={
    "Dense Caption": "denseCaption",
    "Answer": "fct",
    "NOTA_Prompt": "nota",
    "option_A_reason": "optionA",
    "option_B_reason": "optionB",
    "option_C_reason": "optionC",
    "option_D_reason": "optionD"
}, inplace=True)


output_path = f"{DATASET_DIR}/main/bird.csv"  # Change to fish.csv / butterfly.csv as needed
merged_df.to_csv(output_path, index=False)

print(f"Merged dataset saved to {output_path}")

