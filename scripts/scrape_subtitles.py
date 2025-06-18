import os 
import re
import csv


def clean_text(line):
    # Remove narrator brackets 《...》 and (...) and (...)
    line = re.sub(r'[《（(].*?[）》)]', '', line)
    
    # Remove special symbols
    line = re.sub(r'[♪※◆★☆【】→←◯△×●]', '', line)

    # Remove full-width spaces and normalize whitespace
    line = line.replace('\u3000', ' ')  # Japanese wide space to normal space
    line = re.sub(r'\s+', ' ', line)

    # remove Romaji
    line = re.sub(r'[a-zA-Z0-9]', '', line)

    return line.strip()

def extract_subtitles(file_path):
    subtitles = []
    current_block = []
    all_blocks = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip().replace('\ufeff1', '')
            line = line.replace('～', '')
            if not line or line.isdigit() or re.search(r"\d{2}:\d{2}:\d{2},\d{3}", line):
                if current_block:
                    joined = " ".join(current_block)
                    all_blocks.append(joined)
                    current_block = []
                continue
            current_block.append(line)
            cleaned_blocks = [clean_text(block) for block in all_blocks if clean_text(block)]
    return cleaned_blocks


path = r"politeness-classifier-jp\data\raw\subtitles"
subtitles = []
count = 0

for genre in os.listdir(path):
    genre_path = os.path.join(path, genre)
    for episode in os.listdir(genre_path):
        episode_path = os.path.join(genre_path, episode)
        lines = extract_subtitles(episode_path)
        for line in lines:
            subtitles.append(line)
        if count == 3:          # For now we just grab the first 3 episodes of Romance Anime
            break
        count += 1

print(len(subtitles))

output_path = r"politeness-classifier-jp/data/processed/unlabeled_subtitles.csv"
with open(output_path, mode='w', encoding='utf=8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["text", "label"]) # header

    for line in subtitles:
        writer.writerow([line, ""])