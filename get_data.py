from datasets import load_dataset, load_from_disk, Dataset

import pandas as pd
# Load the filtered dataset if it exists
import os
import numpy as np

def get_filtered_tulu_mix():
    # if os.path.exists("filtered_tulu_mix_dataset"):
    #     print("Loading filtered tulu_mix dataset from disk...")
    #     return load_from_disk("filtered_tulu_mix_dataset")
    # else:
    #     print("Filtered tulu_mix dataset not found on disk.")
    return None
filtered_tulu_mix = get_filtered_tulu_mix()
if filtered_tulu_mix is None or True:
    wildjailbreak_train = pd.read_csv("hf://datasets/allenai/wildjailbreak/train/train.tsv", sep="\t")
    wildjailbreak_train_vanilla = set(wildjailbreak_train['vanilla'])
    wildjailbreak_train_adversarial = set(wildjailbreak_train['adversarial'])
    wildjailbreak_test = pd.read_csv("hf://datasets/allenai/wildjailbreak/eval/eval.tsv", sep="\t")
    wildjailbreak_test_adversarial = set(wildjailbreak_test['adversarial'])
    
    wildjailbreak_train_harmful = wildjailbreak_train[wildjailbreak_train['data_type'].str.contains('harmful')]
    wildjailbreak_test_harmful = wildjailbreak_test[wildjailbreak_test['data_type'].str.contains('harmful')]
    wildjailbreak_train_harmful_vanilla = set(wildjailbreak_train_harmful['vanilla'])
    wildjailbreak_train_harmful_adversarial = set(wildjailbreak_train_harmful['adversarial'])
    wildjailbreak_test_harmful_adversarial = set(wildjailbreak_test_harmful['adversarial'])


    wildguard_mix_train = load_dataset("allenai/wildguardmix", "wildguardtrain")
    wildguard_mix_train = wildguard_mix_train['train']
    wildguard_mix_test = load_dataset("allenai/wildguardmix", "wildguardtest")
    wildguard_mix_test = wildguard_mix_test['test']
    wildguard_danger_filter_fn = lambda x: 'benign' not in x['subcategory']
    wildguard_train_danger_filter = wildguard_mix_train.filter(wildguard_danger_filter_fn)
    wildguard_test_danger_filter = wildguard_mix_test.filter(wildguard_danger_filter_fn)

    coconot = load_dataset("allenai/coconot", "original")
    coconot_train = coconot['train']
    coconot_test = coconot['test']
    coco_danger_filter_fn = lambda x: 'danger' in x['subcategory']
    coco_train_danger_filter = coconot_train.filter(coco_danger_filter_fn)
    coco_test_danger_filter = coconot_test.filter(coco_danger_filter_fn)

    tulu_mix = load_dataset("allenai/llama-3.1-tulu-3-8b-preference-mixture")
    tulu_mix_train = tulu_mix['train']



    # Extract prompts from wildguard and coconot datasets
    wildguard_train_prompts = set(wildguard_mix_train['prompt'])
    wildguard_test_prompts = set(wildguard_mix_test['prompt'])
    wildguard_train_danger_filter_prompts = set(wildguard_train_danger_filter['prompt'])
    wildguard_test_danger_filter_prompts = set(wildguard_test_danger_filter['prompt'])

    coconot_train_prompts = set(coconot_train['prompt'])
    coconot_test_prompts = set(coconot_test['prompt'])
    coconot_train_danger_filter_prompts = set(coco_train_danger_filter['prompt'])
    coconot_test_danger_filter_prompts = set(coco_test_danger_filter['prompt'])

    # Combine all prompts to filter out
    all_prompts_to_filter = wildguard_train_prompts.union(wildguard_train_prompts).union(wildguard_test_prompts).union(coconot_train_prompts).union(coconot_test_prompts)
    all_prompts_to_filter = all_prompts_to_filter.union(wildjailbreak_train_vanilla).union(wildjailbreak_train_adversarial).union(wildjailbreak_test_adversarial)
    print("Made prompts to filter, filtering tulu_mix_train...")

    danger_prompts = wildguard_train_danger_filter_prompts.union(wildguard_test_danger_filter_prompts).union(coconot_train_danger_filter_prompts).union(coconot_test_danger_filter_prompts)
    danger_prompts = danger_prompts.union(wildjailbreak_train_harmful_vanilla).union(wildjailbreak_train_harmful_adversarial).union(wildjailbreak_test_harmful_adversarial)
    # Add category column based on whether prompt is in danger_prompts
    tulu_mix_train = tulu_mix_train.add_column(
        "safety_category",
        ["dangerous" if prompt in danger_prompts else "benign" for prompt in tulu_mix_train["prompt"]]
    )

    safety_dataset = tulu_mix_train.filter(
        lambda example: example['prompt'] in all_prompts_to_filter
    )
    
    # 2. no_safety: only includes prompts NOT IN the filter list
    no_safety_dataset = tulu_mix_train.filter(
        lambda example: example['prompt'] not in all_prompts_to_filter
    )

    # Add label column to each dataset
    safety_dataset = safety_dataset.add_column("label", ["safety"] * len(safety_dataset))
    no_safety_dataset = no_safety_dataset.add_column("label", ["no_safety"] * len(no_safety_dataset))

    print(f"Original tulu_mix_train size: {len(tulu_mix_train)}")
    print(f"Safety dataset size: {len(safety_dataset)}")
    print(f"No safety dataset size: {len(no_safety_dataset)}")

    # # Combine both datasets
    # from datasets import concatenate_datasets
    # filtered_tulu_mix = concatenate_datasets([safety_dataset, no_safety_dataset])
    


    # # Add messages column
    # def create_messages(row):
    #     if type(row['chosen']) == list or type(row['chosen']) == np.ndarray:
    #         return list(row['chosen'])
    #     return [
    #         {"content": row["prompt"], "role": "user"},
    #         {"content": row["chosen"], "role": "user"}
    #     ]

    # # Convert to pandas DataFrame
    # df = filtered_tulu_mix.to_pandas()

    # # Apply the function to create the messages column
    # df['messages'] = df.apply(create_messages, axis=1)

    # # Convert back to Hugging Face dataset
    # filtered_tulu_mix = Dataset.from_pandas(df)

    # # Save the filtered dataset to disk
    # filtered_tulu_mix.save_to_disk("filtered_tulu_mix_dataset")
    # print("Filtered dataset saved to 'filtered_tulu_mix_dataset' directory")

    # filtered_tulu_mix.push_to_hub("KJha02/filtered_tulu")
    # print("Filtered dataset pushed to hub")


    '''
    VICTORIA's DATASET
    '''
    victoria_ds = load_dataset("VGraf/safe_responses_dev")
    victoria_train = victoria_ds['train']

    def last_user_turn(messages):
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['role'] == 'user':
                return messages[i]['content']
        return ''
    
    victoria_train = victoria_train.add_column(
        "safety_category",
        ["dangerous" if last_user_turn(messages) in danger_prompts else "benign" for messages in victoria_train["messages"]]
    )
    victoria_ds['train'] = victoria_train
    victoria_ds.push_to_hub("KJha02/victoria_ds")
