import pandas as pd
import matplotlib.pyplot as plt

# Read the evaluation results from the file
df = pd.read_csv('evaluation_results2.txt')

# Extract the epoch numbers from the model names
df['Epoch'] = df['Model'].apply(lambda x: int(x.split('_')[2]))

# Sort the DataFrame by the Epoch column
df_sorted = df.sort_values(by='Epoch')

def plot_metric_for_test_set(df, test_set, metric):
    # Filter the DataFrame for the current test set and metric
    data = df[df['Test Data'] == test_set].sort_values(by='Epoch')

    # Plot the metric
    plt.figure(figsize=(6, 4))
    plt.plot(data['Epoch'], data[metric], marker='o', linestyle='-')

    # Annotate the first and last data points
    plt.annotate(f'{data[metric].iloc[0]:.2f}', (data['Epoch'].iloc[0], data[metric].iloc[0]),
                 textcoords="offset points", xytext=(-10, 10), ha='center')
    plt.annotate(f'{data[metric].iloc[-1]:.2f}', (data['Epoch'].iloc[-1], data[metric].iloc[-1]),
                 textcoords="offset points", xytext=(-10, 10), ha='center')
    plt.title(f'{test_set} - {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

metrics = ['PSNR', 'SSIM', 'MSE']
for test_set in df_sorted['Test Data'].unique():
    for metric in metrics:
        plot_metric_for_test_set(df_sorted, test_set, metric)
