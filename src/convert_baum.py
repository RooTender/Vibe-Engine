import os
import pandas as pd
from moviepy.editor import VideoFileClip

def is_video_file(filename):
    """Check if a file is in a video format based on the file extension."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def convert_videos_to_audio(csv_file):
    annotations = pd.read_csv(csv_file)
    unique_emotions = annotations['Emotion'].unique()
    output_dir = '../data/output'

    if all(os.path.exists(os.path.join(output_dir, emotion)) for emotion in unique_emotions):
        print("All emotion folders already exist. No conversion necessary.")
        return
    
    for _, row in annotations.iterrows():
        clip_name = row['Clip Name']
        emotion = row['Emotion']
        
        video_path = None
        for root, _, files in os.walk('../data/input'):
            for file in files:
                if file.startswith(clip_name) and is_video_file(file):
                    video_path = os.path.join(root, file)
                    break

            if video_path:
                break
        
        if video_path is None:
            print(f"Video file for {clip_name} not found.")
            continue
        
        emotion_folder = os.path.join('../data/output', emotion)
        if not os.path.exists(emotion_folder):
            os.makedirs(emotion_folder)
        
        audio_output_path = os.path.join(emotion_folder, f"{clip_name}.wav")
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_output_path)
            video.close()
            print(f"Converted {video_path} to {audio_output_path}")
        except Exception as e:
            print(f"Error converting {video_path}: {e}")
