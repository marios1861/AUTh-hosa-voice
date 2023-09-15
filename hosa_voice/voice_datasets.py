import random

import torch
import torchvision
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


class DatasetModule(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=8, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.spread = torch.tensor([57, 72, 40, 38])
        self.train_idxs = [
            10,
            42,
            47,
            51,
            8,
            22,
            56,
            38,
            53,
            14,
            6,
            29,
            2,
            49,
            31,
            55,
            17,
            3,
            28,
            41,
            1,
            19,
            35,
            11,
            4,
            21,
            46,
            32,
            97,
            62,
            105,
            101,
            98,
            69,
            84,
            65,
            114,
            119,
            89,
            75,
            85,
            83,
            108,
            58,
            80,
            99,
            115,
            120,
            77,
            125,
            57,
            81,
            59,
            111,
            64,
            128,
            127,
            102,
            61,
            109,
            74,
            106,
            116,
            103,
            137,
            134,
            141,
            166,
            157,
            153,
            136,
            160,
            151,
            139,
            131,
            143,
            132,
            146,
            133,
            168,
            156,
            158,
            164,
            147,
            171,
            204,
            188,
            189,
            190,
            194,
            170,
            198,
            180,
            176,
            183,
            203,
            187,
            181,
            178,
            193,
            202,
            205,
            179,
        ]
        self.val_idxs = [
            13,
            36,
            12,
            24,
            15,
            9,
            40,
            18,
            48,
            43,
            20,
            44,
            37,
            5,
            94,
            117,
            73,
            66,
            91,
            95,
            118,
            112,
            113,
            121,
            93,
            100,
            107,
            71,
            63,
            126,
            60,
            96,
            162,
            148,
            155,
            159,
            135,
            165,
            154,
            130,
            138,
            150,
            184,
            206,
            196,
            185,
            173,
            191,
            172,
            182,
            200,
        ]
        self.test_idxs = [
            25,
            0,
            52,
            54,
            33,
            26,
            23,
            30,
            7,
            39,
            16,
            45,
            50,
            34,
            27,
            88,
            123,
            92,
            122,
            124,
            68,
            82,
            79,
            104,
            78,
            86,
            110,
            90,
            70,
            67,
            76,
            87,
            72,
            149,
            161,
            163,
            167,
            129,
            140,
            142,
            145,
            152,
            144,
            175,
            201,
            169,
            192,
            199,
            195,
            177,
            197,
            174,
            186,
        ]
        self.train_targets = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
        ]
        self.val_targets = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
        ]
        self.test_targets = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
        ]

    def setup(self, stage: str):
        train_transforms = v2.Compose(
            [
                v2.Resize(size=(256, 256), antialias=True),
                v2.RandomCrop(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToImageTensor(),
                v2.ConvertDtype(torch.float),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        test_transforms = v2.Compose(
            [
                v2.Resize(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToImageTensor(),
                v2.ConvertDtype(torch.float),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.train_dataset = ImageFolder(
            self.kwards["data_dir"], transform=train_transforms
        )
        self.test_dataset = ImageFolder(
            self.kwards["data_dir"], transform=test_transforms
        )
        class_weights = 1.0 / self.spread.float()
        train_samples_weight = torch.tensor(
            [class_weights[t] for t in self.train_target[self.train_idxs]]
        )
        self.sampler = WeightedRandomSampler(
            train_samples_weight, len(train_samples_weight)
        )
        self.train_dataset = Subset(self.train_dataset, self.train_idxs)
        self.val_dataset = Subset(self.test_dataset, self.val_idxs)
        self.test_dataset = Subset(self.test_dataset, self.test_idxs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


# choose train, val, test splits
def main():
    torchvision.disable_beta_transforms_warning()
    transforms = v2.Compose(
        [
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImageTensor(),
            v2.ConvertDtype(torch.float),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageFolder(
        "./datasets/bicepstrum_image/bicepstrum_ml_normalized_imagesc_100_100/",
        transform=transforms,
    )
    pairs = [(idx, dataset[idx][1]) for idx in range(len(dataset))]
    labels = [pair[1] for pair in pairs]
    spread = [labels.count(label) for label in range(4)]
    random_per_label_pairs = [
        list(filter(lambda pair: pair[1] == label, pairs)) for label in range(4)
    ]
    [
        random.shuffle(random_per_label_pair)
        for random_per_label_pair in random_per_label_pairs
    ]
    train_pairs = [
        random_per_label_pair[: spread[label] // 2]
        for label, random_per_label_pair in enumerate(random_per_label_pairs)
    ]
    val_pairs = [
        random_per_label_pair[
            spread[label] // 2 : spread[label] // 2 + spread[label] // 4
        ]
        for label, random_per_label_pair in enumerate(random_per_label_pairs)
    ]
    test_pairs = [
        random_per_label_pair[spread[label] // 2 + spread[label] // 4 :]
        for label, random_per_label_pair in enumerate(random_per_label_pairs)
    ]
    train_idxs = [pair[0] for label_list in train_pairs for pair in label_list]
    val_idxs = [pair[0] for label_list in val_pairs for pair in label_list]
    test_idxs = [pair[0] for label_list in test_pairs for pair in label_list]
    train_targets = [pair[1] for label_list in train_pairs for pair in label_list]
    val_targets = [pair[1] for label_list in val_pairs for pair in label_list]
    test_targets = [pair[1] for label_list in test_pairs for pair in label_list]
    print(
        spread,
        train_idxs,
        val_idxs,
        test_idxs,
        train_targets,
        val_targets,
        test_targets,
    )


if __name__ == "__main__":
    main()
