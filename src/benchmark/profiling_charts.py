import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

processing_times = {
    "dominant_freq": {
        "min_time_ms": 0.37097930908203125,
        "avg_time_ms": 0.6779885011265265,
        "max_time_ms": 22.18031883239746,
        "std_time_ms": 0.2663363486286387
    },
    "fundamental_freq": {
        "min_time_ms": 0.021457672119140625,
        "avg_time_ms": 0.8109184627846581,
        "max_time_ms": 12.711763381958008,
        "std_time_ms": 0.21048316002676984
    },
    "stft": {
        "min_time_ms": 0.164031982421875,
        "avg_time_ms": 45.25460156100093,
        "max_time_ms": 94.81096267700195,
        "std_time_ms": 21.691203592702877
    },
    "mel_energy": {
        "min_time_ms": 0.32711029052734375,
        "avg_time_ms": 0.6526286030486694,
        "max_time_ms": 16.56818389892578,
        "std_time_ms": 0.508751264126959
    },
    "bark_energy": {
        "min_time_ms": 4.271030426025391,
        "avg_time_ms": 10.70452329424123,
        "max_time_ms": 33.89406204223633,
        "std_time_ms": 3.144099354949731
    },
    "cqt_energy": {
        "min_time_ms": 10.834932327270508,
        "avg_time_ms": 18.284448352091317,
        "max_time_ms": 736.59348487854,
        "std_time_ms": 5.631203496253409
    },
    "erb_energy": {
        "min_time_ms": 0.8304119110107422,
        "avg_time_ms": 1.906084907590013,
        "max_time_ms": 26.92103385925293,
        "std_time_ms": 1.5645846567471833
    },
    "gammatone_energy": {
        "min_time_ms": 0.4036426544189453,
        "avg_time_ms": 0.6773021584749895,
        "max_time_ms": 4.79578971862793,
        "std_time_ms": 0.1016656840402155
    },
    "lpc": {
        "min_time_ms": 0.016450881958007812,
        "avg_time_ms": 1.0996135249921752,
        "max_time_ms": 17.405986785888672,
        "std_time_ms": 0.2036263963613974
    },
    "mfcc": {
        "min_time_ms": 0.6377696990966797,
        "avg_time_ms": 1.1711491460872296,
        "max_time_ms": 5.311250686645508,
        "std_time_ms": 0.17441365804405412
    },
    "bfcc": {
        "min_time_ms": 5.343198776245117,
        "avg_time_ms": 13.125934050736866,
        "max_time_ms": 41.579484939575195,
        "std_time_ms": 2.278227271605086
    },
    "gfcc": {
        "min_time_ms": 0.7464885711669922,
        "avg_time_ms": 1.4441459367402354,
        "max_time_ms": 13.610124588012695,
        "std_time_ms": 0.48859754849309134
    },
    "rms": {
        "min_time_ms": 0.13566017150878906,
        "avg_time_ms": 0.23069812135789347,
        "max_time_ms": 12.150764465332031,
        "std_time_ms": 0.1481420530258339
    },
    "zcr": {
        "min_time_ms": 0.09894371032714844,
        "avg_time_ms": 0.16352753295278372,
        "max_time_ms": 10.91146469116211,
        "std_time_ms": 0.10654473593345196
    },
    "teo": {
        "min_time_ms": 0.022411346435546875,
        "avg_time_ms": 0.038877362236461876,
        "max_time_ms": 8.986234664916992,
        "std_time_ms": 0.055161353642805985
    },
    "psd": {
        "min_time_ms": 0.1652240753173828,
        "avg_time_ms": 0.27700446011470464,
        "max_time_ms": 8.39853286743164,
        "std_time_ms": 0.061381839071156005
    },
    "asd": {
        "min_time_ms": 0.15020370483398438,
        "avg_time_ms": 0.2535992229918251,
        "max_time_ms": 12.346267700195312,
        "std_time_ms": 0.15538091549309457
    },
    "spectral_entropy": {
        "min_time_ms": 0.16069412231445312,
        "avg_time_ms": 0.27189809339383875,
        "max_time_ms": 12.111663818359375,
        "std_time_ms": 0.12654724388815633
    },
    "spectral_centroid": {
        "min_time_ms": 0.25463104248046875,
        "avg_time_ms": 0.42770097023989445,
        "max_time_ms": 12.386798858642578,
        "std_time_ms": 0.22729997510597827
    },
    "spectral_flux": {
        "min_time_ms": 0.04601478576660156,
        "avg_time_ms": 0.0737978300858276,
        "max_time_ms": 9.421348571777344,
        "std_time_ms": 0.05789947402139572
    },
    "spectral_contrast": {
        "min_time_ms": 0.4634857177734375,
        "avg_time_ms": 0.7686473788878502,
        "max_time_ms": 16.910314559936523,
        "std_time_ms": 0.2734813933728258
    }
}

df = pd.DataFrame.from_dict(processing_times, orient='index')
df.index.name = 'Feature'
df.reset_index(inplace=True)


df['cv'] = df['std_time_ms'] / df['avg_time_ms'] * 100

total_avg_time = df['avg_time_ms'].sum()

df_rounded = df.copy()
numeric_cols = ['min_time_ms', 'avg_time_ms', 'cv', 'std_time_ms', 'max_time_ms']
df_rounded[numeric_cols] = df_rounded[numeric_cols].round(2)

df_heatmap = df_rounded.set_index('Feature')

df_heatmap = df_heatmap.transpose()

sns.set_theme(style="whitegrid")


plt.figure(figsize=(20, 8))  
from matplotlib.colors import LogNorm

numeric_data = df_heatmap.select_dtypes(include=[float, int])
norm = LogNorm(vmin=numeric_data.min().min(), vmax=numeric_data.max().max())

custom_labels_list = [
    'Min. time (ms)', 
    'Avg. time (ms)', 
    'Max. time (ms)',
    'Std. deviation (ms)',
    'Flow coefficient (%)', 
]

heatmap = sns.heatmap(
	df_heatmap, 
	annot=True,
	fmt=".2f", 
	cmap="RdYlGn_r", 
	linewidths=.5, 
	norm=norm, 
	cbar_kws={'label': 'Value'},
	yticklabels=custom_labels_list
)

plt.title('Heatmap of Audio Feature Processing Time Statistics', fontsize=16, pad=20)
plt.xlabel('Statistical Features', fontsize=14)
plt.ylabel('Sound Features', fontsize=14)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()

plt.savefig("statistics_heatmap.png", dpi=300)


# Dane rozmiarów
data_sizes = {
    "dominant_freq": {"size": 2},
    "fundamental_freq": {"size": 4},
    "stft": {"size": 1809},
    "mel_energy": {"size": 24},
    "bark_energy": {"size": 24},
    "cqt_energy": {"size": 4},
    "erb_energy": {"size": 42},
    "gammatone_energy": {"size": 24},
    "lpc": {"size": 28},
    "mfcc": {"size": 26},
    "bfcc": {"size": 26},
    "gfcc": {"size": 26},
    "rms": {"size": 4},
    "zcr": {"size": 4},
    "teo": {"size": 1764},
    "psd": {"size": 883},
    "asd": {"size": 883},
    "spectral_entropy": {"size": 1},
    "spectral_centroid": {"size": 4},
    "spectral_flux": {"size": 1},
    "spectral_contrast": {"size": 28}
}

df_sizes = pd.DataFrame.from_dict(data_sizes, orient='index')
df_sizes.reset_index(inplace=True)
df_sizes.rename(columns={'index': 'Feature', 'size': 'Size'}, inplace=True)

# Sortowanie danych dla lepszej czytelności (najmniejsze na dole)
df_sizes_sorted = df_sizes.sort_values(by='Size', ascending=True)

# Ustawienia stylu seaborn
sns.set_theme(style="whitegrid")

# Tworzenie wykresu słupkowego
plt.figure(figsize=(14, 10))  # Dostosuj rozmiar w razie potrzeby

ax = sns.barplot(
    x='Size',
    y='Feature',
    data=df_sizes_sorted,
    palette="viridis"
)

# Ustawienie skali logarytmicznej na osi X
ax.set_xscale('log')

# Dodanie etykiet na słupkach
for p in ax.patches:
    width = p.get_width()
    y = p.get_y() + p.get_height() / 2
    # Formatowanie etykiet z separatorami tysięcy
    label = f'{int(width):,}'
    
    # Ustawienie pozycji etykiety na 2% szerszej niż szerokość słupka
    ax.text(
        width * 1.02,  # Pozycja X
        y,              # Pozycja Y
        label,          # Tekst etykiety
        ha='left',      # Wyrównanie poziome
        va='center',    # Wyrównanie pionowe
        fontsize=10,
        color='black'
    )

# Dodanie tytułu i etykiet osi
plt.title('Size of data received as a result of processing', fontsize=18, pad=20)
plt.xlabel('NUmber of elements in data', fontsize=14)
plt.ylabel('Sound Feature', fontsize=14)

# Dostosowanie zakresu osi X, aby etykiety nie były obcięte
max_size = df_sizes_sorted['Size'].max()
ax.set_xlim(0, max_size * 1.1)  # Dodanie 10% marginesu

# Rotacja etykiet osi Y dla lepszej czytelności (opcjonalnie)
plt.yticks(fontsize=12)

# Dostosowanie układu
plt.tight_layout()

# Zapisanie wykresu do pliku (opcjonalnie)
plt.savefig("size_statistics.png", dpi=300, bbox_inches='tight')
