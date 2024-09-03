import convert_baum as baum
from audio_dataset import AudioDataset

baum.convert_videos_to_audio("../data/Annotations_BAUM1a.csv")

dataset = AudioDataset('../data/output')