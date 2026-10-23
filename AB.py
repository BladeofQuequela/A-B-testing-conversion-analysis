import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

a = pd.read_csv(r"C:\Users\mshan\Desktop\ab_test_dataset.csv")
a_clean = a.dropna(subset=['group']).copy()
a_clean['group'] = a_clean['group'].str.strip().str.lower().replace({'a': 'con'})
a_clean = a_clean[a_clean['group'].isin(['exp', 'con'])]

stats = a_clean.groupby('group')['click'].agg(
    clicks='sum',
    users='count'
)
stats['ctr'] = stats['clicks'] / stats['users']

print(stats)

clicks = stats['clicks'].values
users = stats['users'].values

z_stat, p_value = proportions_ztest(count=clicks, nobs=users)
print(f"\nZ-statistic: {z_stat:.2f}")
print(f"P-value: {p_value}")

a_clean['device_type'] = a_clean['device_type'].str.strip().str.lower()


sns.set_theme(style="whitegrid")

groups = stats.index
ctr_values = stats['ctr']

plt.figure(figsize=(8, 5))

bars = plt.bar(groups, ctr_values, color=['#808080', '#2ecc71'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.1%}', ha='center', va='bottom', fontweight='bold')

plt.title('Conversion Rate (CTR) by Group', fontsize=14, fontweight='bold')
plt.ylabel('CTR')
plt.ylim(0, 0.6)
plt.savefig(BASE_DIR / 'images' / 'сonversion.png', dpi=150, bbox_inches='tight')
plt.show()

pivot = a_clean.pivot_table(index='device_type', columns='group', values='click', aggfunc='mean')
print(pivot)

plt.figure(figsize=(10, 6))
pivot.plot(kind='bar', color=['#808080', '#2ecc71'], edgecolor='black', ax=plt.gca())

plt.title('CTR by Device Type and Group', fontsize=14, fontweight='bold')
plt.xlabel('Device Type')
plt.ylabel('CTR')
plt.xticks(rotation=0)
plt.legend(title='Group', loc='upper left')


for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.1%}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points', fontsize=10)

plt.tight_layout()
plt.savefig(BASE_DIR / 'images' / 'device_conv.png', dpi=150, bbox_inches='tight')
plt.show()

