import convert_baum as baum
import audio_dataset

baum.convert_videos_to_audio("../data/Annotations_BAUM1a.csv")
baum_db = audio_dataset.AudioDataset('../data/output')