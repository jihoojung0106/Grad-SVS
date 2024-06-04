import bisect
import os

import torch
from torch.utils.data import DataLoader, Dataset


class SingleVoiceKoreanDataset(Dataset):
    def __init__(self, mode, data_path, chunk_size=200):
        assert mode in {"train", "valid","test"}
        super().__init__()
        self.chunk_size = chunk_size
        self.data_path = os.path.join(data_path, mode)
        self.file_list = os.listdir(self.data_path)
        self.file_list.sort()
        self.lengths = get_cumul_len(self.data_path, chunk_size)

    def __len__(self):
        return self.lengths[-1]

    def __getitem__(self, idx):
        if idx < 0:
            return self[len(self) + idx]
        data_idx = bisect.bisect_right(self.lengths, idx)
        start = idx if data_idx == 0 else idx - self.lengths[data_idx - 1]
        end = start + self.chunk_size
        notes, phonemes, mel = torch.load(
            os.path.join(self.data_path, self.file_list[data_idx])
        )
        return notes[start:end], phonemes[start:end], mel[start:end]


def make_loaders(mode):
    data_path = "data_preprocess/bin"
    chunk_size = 192
    dataset = SingleVoiceKoreanDataset(mode, data_path, chunk_size)
    num_workers = get_num_workers(1)
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=2,
        shuffle=True
    )
    # valid_dataset = SingleVoiceKoreanDataset("valid", data_path, chunk_size)
    # valid_loader = DataLoader(
    #     valid_dataset,
    #     batch_size=384,
    #     num_workers=num_workers,
    #     shuffle=False,
    # )
    return loader #, valid_loader


def get_num_workers(num_workers_config):
    try:
        return int(num_workers_config)
    except ValueError:
        assert num_workers_config == "all"
        return os.cpu_count()


def get_cumul_len(data_path, chunk_size=128):
    lengths = []
    files = os.listdir(data_path)
    files.sort()
    for file_ in files:
        notes, _, _ = torch.load(os.path.join(data_path, file_))
        lengths.append(len(notes))
    lengths = [count_chunks(length, chunk_size) for length in lengths]
    cumul_sum_(lengths)
    return lengths


def count_chunks(length, chunk_size):
    return length - chunk_size + 1


def cumul_sum_(lengths):
    for i in range(len(lengths)):
        if i == 0:
            continue
        lengths[i] += lengths[i - 1]


# if __name__ == '__main__':
#     print("Hello")  # 여기서 브레이크포인트를 설정하세요
#     train_loader = make_loaders()
#     it = iter(train_loader)
#     for _ in range(10):
#         item = next(it)
#         print(item)
#         print()
