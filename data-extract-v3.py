import os
import lzma
from tqdm import tqdm
import concurrent.futures
import random

def process_file(args):
    directory, filename, output_file, vocab = args
    file_path = os.path.join(directory, filename)
    with lzma.open(file_path, "rt", encoding="utf-8") as infile:
        text = infile.read()
    with open(output_file, "a", encoding="utf-8") as outfile:
        outfile.write(text)
    characters = set(text)
    return characters

def xz_files_in_dir(directory):
    return [filename for filename in os.listdir(directory) if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename))]

def process_files_in_parallel(files, folder_path, output_file):
    vocab = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        args = [(folder_path, filename, output_file, vocab) for filename in files]
        for characters in tqdm(executor.map(process_file, args), total=len(files)):
            vocab.update(characters)
    return vocab


# if you don't put this guard (if __name__ == '__main__':)  ---- > code start a new process before the current process has finished its bootstrapping phase ((error))
if __name__ == '__main__':
    
    folder_path = "D:\Code_store\LLM_build\data\openwebtext"
    output_file_train = "output_train.txt"
    output_file_val = "output_val.txt"
    vocab_file = "vocab.txt"

    files = xz_files_in_dir(folder_path)
    total_files = len(files)
    print(total_files)

    split_index = int(total_files * 0.8)  # 90% for training
    files_train = files[:split_index]
    files_val = files[split_index:]

    # Sampling a hundredth of the files for each split
    sample_rate = 0.1
    files_train_sampled = random.sample(files_train, max(1, int(len(files_train) * sample_rate)))
    files_val_sampled = random.sample(files_val, max(1, int(len(files_val) * sample_rate)))

    # Ensure output files are empty before appending
    open(output_file_train, 'w').close()
    open(output_file_val, 'w').close()

    # Process the sampled training files
    vocab_train = process_files_in_parallel(files_train_sampled, folder_path, output_file_train)

    # Process the sampled validation files
    vocab_val = process_files_in_parallel(files_val_sampled, folder_path, output_file_val)

    # Combine vocabularies (if needed) and write to vocab.txt
    vocab = vocab_train.union(vocab_val)
    with open(vocab_file, "w", encoding="utf-8") as vfile:
        for char in sorted(vocab):
            vfile.write(char + '\n')
