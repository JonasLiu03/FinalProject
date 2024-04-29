import pandas as pd
import matplotlib.pyplot as plt


# df1 = pd.read_csv('crop64batch8mish.txt')
# df2 = pd.read_csv('crop64batch8prelu.txt')
# df3 = pd.read_csv('crop96batch16prelu.txt')
#
# df1['Model Label'] = 'Model 1'
# df2['Model Label'] = 'Model 2'
# df3['Model Label'] = 'Model 3'
#

# dfs = [df1, df2, df3]
# for df in dfs:
#     df['Epoch'] = df['Model'].apply(lambda x: int(x.split('_')[2]))
#

# metrics = ['PSNR', 'SSIM', 'MSE']
# test_sets = df1['Test Data'].unique()
#
#
# def plot_metric(df_list, metric):

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
#     for ax, test_set in zip(axes, test_sets):
#         for df in df_list:
#             data = df[df['Test Data'] == test_set].sort_values(by='Epoch')
#             ax.plot(data['Epoch'], data[metric], marker='o', linestyle='-', label=df['Model Label'].iloc[0])
#             ax.set_title(f'{test_set} - {metric}')
#             ax.set_xlabel('Epoch')
#             ax.set_ylabel(metric)
#             ax.grid(True)
#             ax.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#

# for metric in metrics:
#     plot_metric(dfs, metric)
df = pd.read_csv('crop64batch8mishALL.txt')
df['Model Label'] = 'Model 1'
df['Epoch'] = df['Model'].apply(lambda x: int(x.split('_')[2]))
metrics = ['PSNR', 'SSIM', 'MSE']
test_sets = df['Test Data'].unique()
def plot_metric(df, metric):
    fig, axes = plt.subplots(1, len(test_sets), figsize=(18, 6), sharey=True)

    for ax, test_set in zip(axes, test_sets):
        data = df[df['Test Data'] == test_set].sort_values(by='Epoch')
        ax.plot(data['Epoch'], data[metric], marker='o', linestyle='-')
        ax.set_title(f'{test_set} - {metric}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.grid(True)
        last_point = data.iloc[-1]
        ax.annotate(f'{last_point[metric]:.3f}',
                    (last_point['Epoch'], last_point[metric]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    plt.tight_layout()
    plt.show()



for metric in metrics:
    plot_metric(df, metric)