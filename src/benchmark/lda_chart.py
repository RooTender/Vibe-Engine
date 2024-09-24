import matplotlib.pyplot as plt
import numpy as np

def create_bar_chart(data, bar_labels):
    """
    Create a bar chart with the specified data and display values in percentiles.
    Bars with a value of 10% will be red, and those greater than 20% will be green.

    Parameters:
    - data: 1D numpy array of shape (num_bars,) with values for the bars.
    - bar_labels: List of labels for each bar.
    """
    num_bars = len(data)
    x = np.arange(num_bars)  # The label locations

    fig, ax = plt.subplots(figsize=(12, 6))

    # Define the colors for the bars
    colors = []
    for value in data:
        if value == 10:
            colors.append('red')
        elif value > 20:
            colors.append('green')
        else:
            colors.append('grey')  # Default color for other bars

    # Create the bar chart with custom colors
    bars = ax.bar(x, data, width=0.5, color=colors)

    # Add data labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.25,
            f'{height}%',  # Format as percentage
            ha='center', va='bottom', fontsize=8
        )

    # Set labels, title, and ticks
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('LDA results (higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=90, ha='center', fontsize=10)

    # Adjust y-axis limit to make room for data labels
    max_value = np.max(data)
    ax.set_ylim(0, max_value + 5)

    # Show the plot
    plt.tight_layout()
    plt.savefig('lda_chart_colored.png', bbox_inches='tight')

# Data and labels
data = np.array([
    10, 11, 
    24,
    13, 13, 10, 13, 13,
    14,
    18, 18, 16,
    11, 10, 20, 
    20, 28,
    11, 10, 11, 14
])

bar_labels = [
    'Dominant frequencies', 'Fundamental frequencies',
    'STFT',
    'Mel energy', 'Bark energy', 'CQT energy', 'ERB energy', 'Gammatone energy',
    'LPC',
    'MFCC', 'BFCC', 'GFCC',
    'RMS', 'ZCR', 'TEO',
    'PSD', 'ASD',
    'Spectral entropy', 'Spectral centroid', 'Spectral flux', 'Spectral contrast'
]

# Create the bar chart
create_bar_chart(data, bar_labels)
