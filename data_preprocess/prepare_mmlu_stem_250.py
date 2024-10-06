"""Select 250 random samples from the STEM subset of the MMLU benchmark"""

import pandas as pd


subject2category = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
} # from OpenAI simple-evals


def main(data_path: str, output_root_dir: str):
    # load the data
    df = pd.read_csv(data_path)
    # filter the data
    df = df[df["Subject"].map(subject2category) == "stem"]
    # select 250 + 5 random samples
    df = df.sample(n=255, random_state=0)
    few_shot_df = df.tail(5)
    df = df.head(250)
    # save the data
    df.to_csv(f"{output_root_dir}/mmlu_stem_250.csv", index=False)
    few_shot_df.to_csv(f"{output_root_dir}/mmlu_stem_250_few_shot.csv", index=False)


if __name__ == "__main__":
    data_path = "/mnt/d/projects/encoded-prompting-llm/data/mmlu/mmlu_with_index.csv"
    output_root_dir = "/mnt/d/projects/encoded-prompting-llm/data/mmlu"
    main(data_path, output_root_dir)
