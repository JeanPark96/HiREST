from os import path
from pydub import AudioSegment
import argparse
import os
from tqdm import tqdm

if __name__ == '__main__':
    print('Extracting audio from videos...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # files  
    for root, dir, files in os.walk(args.audio_path):
        file_names = [f.replace(".mp3", "") for f in files] 

        for fname in tqdm(file_names):                                                                    
            src = os.path.join(root, fname) + ".mp3"
            dst = os.path.join(args.save_path, fname) + ".wav"

            # convert wav to mp3 
            if not os.path.exists(dst):                                                           
                sound = AudioSegment.from_mp3(src)
                sound.export(dst, format="wav")