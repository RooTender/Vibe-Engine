import matplotlib.pyplot as plt
import numpy as np

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1 - luminość) by the given amount.

    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((0.3, 0.55, 0.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = mc.to_rgb(c)
    c = colorsys.rgb_to_hls(*c)
    new_color = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    return new_color

def create_bar_chart(data, group_labels, bar_labels):
    """
    Create a bar chart with the specified data and display values in percentiles.

    Parameters:
    - data: 2D numpy array of shape (num_groups, bars_per_group) with values for the bars.
    - group_labels: List of labels for each group.
    - bar_labels: List of labels for each bar within the groups.
    """
    num_groups, bars_per_group = data.shape
    bar_width = 0.15
    x = np.arange(num_groups)  # The label locations

    fig, ax = plt.subplots(figsize=(12, 6))

    # Define base colors for each transformer
    base_colors = {
        'STFT': '#1f77b4',  # Blue
        'CWT': '#2ca02c',   # Green
        'SLT': '#ff7f0e'    # Orange
    }

    # Create a color map for the bars
    colors = []
    for label in bar_labels:
        if 'STFT' in label:
            color = base_colors['STFT']
        elif 'CWT' in label:
            color = base_colors['CWT']
        elif 'SLT' in label:
            color = base_colors['SLT']
        else:
            color = '#7f7f7f'  # Default gray

        if 'power of 3' in label:
            # Lighten the base color for the 'power 9' version
            color = lighten_color(color, 0.6)

        if 'power of 6' in label:
            color = lighten_color(color, 0.4)

        colors.append(color)

    # Plot each set of bars
    for i in range(bars_per_group):
        bars = ax.bar(x + i * bar_width, data[:, i], bar_width, label=bar_labels[i], color=colors[i])

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, height + 0.5,
                f'{height:.1f}%',  # Format as percentage
                ha='center', va='bottom', fontsize=8
            )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Variance (%)')
    ax.set_title('PCA results (higher is better)')
    ax.set_xticks(x + bar_width * (bars_per_group / 2 - 0.5))
    ax.set_xticklabels(group_labels)

    # Adjust y-axis limit to make room for data labels
    max_value = np.max(data)
    ax.set_ylim(0, max_value + 10)

    # Adjust legend to be at the bottom of the plot
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=5)
    

    # Show the plot
    plt.tight_layout()
    plt.savefig('pca_chart.png', bbox_inches='tight')

# data = np.array([
#     [36.81, 39.67, 57.59, 37.58, 37.27, 41.98, 41.12],
#     [45.69, 51.24, 66.62, 42.11, 47.29, 46.80, 54.23],
#     [51.07, 56.44, 69.34, 45.71, 53.20, 50.22, 63.51] 
# ])

data = np.array([
    [50.04, 48.72, 48.43, 53.95, 56.90],
    [65.53, 64.79, 61.13, 62.73, 65.04],
    [71.84, 70.97, 66.64, 68.27, 72.79]
])

group_labels = ['1 component', '2 components', '3 components']
bar_labels = ['STFT', 'STFT to the power of 3', 'STFT to the power of 6', 'CWT', 'SLT']

# Create the bar chart
create_bar_chart(data, group_labels, bar_labels)
