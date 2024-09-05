import pandas as pd

csv_file = '/cs/home/alykc8/vit_train_dataset.csv'
train_df = pd.read_csv(csv_file)

no_pain_df = train_df[train_df['pspi_score'] == 0]
pain_df = train_df[train_df['pspi_score'] != 0]

next_majority_non_zero_pain_class = pain_df['pspi_score'].value_counts().idxmax()

next_majority_non_zero_pain_count = pain_df['pspi_score'].value_counts().max()

undersampled_no_pain_df = no_pain_df.sample(n=next_majority_non_zero_pain_count * 2, random_state=42)

undersampled_train_df = pd.concat([undersampled_no_pain_df, pain_df])

undersampled_train_df = undersampled_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

undersampled_train_df.to_csv('/cs/home/alykc8/vit_undersampled_train_dataset.csv', index=False)

print("Training dataset undersampled and saved successfully.")
