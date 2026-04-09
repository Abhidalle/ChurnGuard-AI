import pandas as pd

df = pd.read_csv(
    '/content/ebd_NP_smp_relFeb-2026.txt',
    sep='\t',
    encoding='utf-8',
    on_bad_lines='skip'
)

df.to_csv('nepal_birds.csv', index=False)
print(f"Done! {len(df)} rows, {len(df.columns)} columns")
df.head()