"""Prepare the arithmetic benchmarks"""

import numpy as np
import pandas as pd


def generate_addition_samples(max_digits: int, n: int) -> list[dict]:
    """Generate n samples with up to max_digits digits in the operands"""
    samples = []
    store = set()
    for i in range(n):
        x1, x2 = np.random.randint(0, 10 ** max_digits, 2)
        while (x1, x2) in store:
            x1, x2 = np.random.randint(0, 10 ** max_digits, 2)
        store.add((x1, x2))
        prompt = f"{x1} + {x2} = "
        samples.append({
            "index": i, 
            "prompt": prompt, 
            "target": x1 + x2
        })
    return samples


def main(output_root_dir: str):

    max_digits = [1, 2, 4, 8]
    num_samples = [75, 100, 200, 200]
    for m, n in zip(max_digits, num_samples):
        # generate the data
        samples = generate_addition_samples(m, n + 5)
        # separate the last 5 for few shot
        few_shot_samples = samples[-5:]
        samples = samples[:-5]
        # save the data
        df = pd.DataFrame(samples)
        df.to_csv(f"{output_root_dir}/arithmetic/addition_{m}digits.csv", index=False)
        df = pd.DataFrame(few_shot_samples)
        df.to_csv(f"{output_root_dir}/arithmetic/addition_{m}digits_few_shot.csv", index=False)


if __name__ == "__main__":
    np.random.seed(0)
    output_root_dir = "/mnt/d/projects/encoded-prompting-llm/data/"
    main(output_root_dir)
