import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load fraud and non-fraud data
rawData1 = pd.read_csv('visualization.csv', nrows=3)
cols = rawData1.columns
rawData2 = pd.read_csv('visualization.csv', skiprows=40204)
rawData2.columns = cols
data = pd.concat([rawData1, rawData2], ignore_index=True)

# --- Plot 1: Consumers With Fraud ---
fig, axs = plt.subplots(2, 1, figsize=(10, 6), dpi=400)
fig.suptitle('Consumers With Fraud', fontsize=18)
plt.subplots_adjust(hspace=0.5)

data.loc[0].plot(ax=axs[0], color='firebrick', grid=True)
axs[0].set_title('Consumer 0', fontsize=14)
axs[0].set_xlabel('Dates')
axs[0].set_ylabel('Consumption')

data.loc[2].plot(ax=axs[1], color='firebrick', grid=True)
axs[1].set_title('Consumer 1', fontsize=14)
axs[1].set_xlabel('Dates')
axs[1].set_ylabel('Consumption')

fig.tight_layout()
fig.savefig('fraud_consumers.png', bbox_inches='tight', pad_inches=0.2)

# --- Plot 2: Consumers Without Fraud ---
fig, axs = plt.subplots(2, 1, figsize=(10, 6), dpi=400)
fig.suptitle('Consumers Without Fraud', fontsize=18)
plt.subplots_adjust(hspace=0.5)

data.loc[3].plot(ax=axs[0], color='teal', grid=True)
axs[0].set_title('Consumer 40255', fontsize=14)
axs[0].set_xlabel('Dates')
axs[0].set_ylabel('Consumption')

data.loc[4].plot(ax=axs[1], color='teal', grid=True)
axs[1].set_title('Consumer 40256', fontsize=14)
axs[1].set_xlabel('Dates')
axs[1].set_ylabel('Consumption')

fig.tight_layout()
fig.savefig('nonfraud_consumers.png', bbox_inches='tight', pad_inches=0.2)

# --- Plot 3: Statistics for Consumers with Fraud ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=400)
fig.suptitle('Statistics for Consumer with Fraud', fontsize=18)
plt.subplots_adjust(hspace=0.4)

data.loc[0].plot(ax=axs[0, 0], color='firebrick', grid=True)
axs[0, 0].set_title('Time Series')

data.loc[0].hist(ax=axs[0, 1], color='firebrick', grid=True)
axs[0, 1].set_title('Histogram')

data.loc[0].plot.kde(ax=axs[1, 0], color='firebrick', grid=True)
axs[1, 0].set_title('KDE')

data.loc[0].describe().drop(['count']).plot(kind='bar', ax=axs[1, 1], color='firebrick', grid=True)
axs[1, 1].set_title('Stats Summary')

fig.tight_layout()
fig.savefig('stats_fraud.png', bbox_inches='tight', pad_inches=0.2)

# --- Plot 4: Statistics for Consumers without Fraud ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=400)
fig.suptitle('Statistics for Consumer without Fraud', fontsize=18)
plt.subplots_adjust(hspace=0.4)

data.loc[4].plot(ax=axs[0, 0], color='teal', grid=True)
axs[0, 0].set_title('Time Series')

data.loc[4].hist(ax=axs[0, 1], color='teal', grid=True)
axs[0, 1].set_title('Histogram')

data.loc[4].plot.kde(ax=axs[1, 0], color='teal', grid=True)
axs[1, 0].set_title('KDE')

data.loc[4].describe().drop(['count']).plot(kind='bar', ax=axs[1, 1], color='teal', grid=True)
axs[1, 1].set_title('Stats Summary')

fig.tight_layout()
fig.savefig('stats_nonfraud.png', bbox_inches='tight', pad_inches=0.2)

# --- Plot 5: Four Week Consumption Comparison ---
fig, axs = plt.subplots(2, 1, figsize=(10, 6), dpi=400)
fig.suptitle('Four Week Consumption Comparison', fontsize=16)
plt.subplots_adjust(hspace=0.4)

for i in range(59, 83, 7):
    axs[0].plot(data.iloc[0, i:i+7].to_numpy(), marker='o', label=f'Week {(i-59)//7 + 1}')
axs[0].legend()
axs[0].set_title('Fraudulent Consumer')
axs[0].set_ylabel('Consumption')
axs[0].grid(True)

for i in range(59, 83, 7):
    axs[1].plot(data.iloc[4, i:i+7].to_numpy(), marker='o', label=f'Week {(i-59)//7 + 1}')
axs[1].legend()
axs[1].set_title('Non-Fraudulent Consumer')
axs[1].set_ylabel('Consumption')
axs[1].grid(True)

fig.tight_layout()
fig.savefig('weekly_comparison.png', bbox_inches='tight', pad_inches=0.2)

# --- Plot 6: Correlation Matrix ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=400)

# Correlation for fraud
weeks = [data.iloc[0, i:i+7].to_numpy() for i in range(59, 83, 7)]
cor_fraud = pd.DataFrame(weeks).T.corr()
cax = axs[0].matshow(cor_fraud, cmap='coolwarm')
for (i, j), val in np.ndenumerate(cor_fraud):
    axs[0].text(j, i, f'{val:.1f}', ha='center', va='center', color='white')
axs[0].set_title('Fraud Consumer Correlation')
axs[0].set_xticks(range(4))
axs[0].set_yticks(range(4))
axs[0].set_xticklabels(['Week 1', 'Week 2', 'Week 3', 'Week 4'])
axs[0].set_yticklabels(['Week 1', 'Week 2', 'Week 3', 'Week 4'])

# Correlation for non-fraud
weeks = [data.iloc[4, i:i+7].to_numpy() for i in range(59, 83, 7)]
cor_nonfraud = pd.DataFrame(weeks).T.corr()
cax = axs[1].matshow(cor_nonfraud, cmap='coolwarm')
for (i, j), val in np.ndenumerate(cor_nonfraud):
    axs[1].text(j, i, f'{val:.1f}', ha='center', va='center', color='white')
axs[1].set_title('Non-Fraud Consumer Correlation')
axs[1].set_xticks(range(4))
axs[1].set_yticks(range(4))
axs[1].set_xticklabels(['Week 1', 'Week 2', 'Week 3', 'Week 4'])
axs[1].set_yticklabels(['Week 1', 'Week 2', 'Week 3', 'Week 4'])

fig.colorbar(cax, ax=axs.ravel().tolist(), shrink=0.6)
fig.tight_layout()
fig.savefig('correlation_matrix.png', bbox_inches='tight', pad_inches=0.2)

plt.close('all')
