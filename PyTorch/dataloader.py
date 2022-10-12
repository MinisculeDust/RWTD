# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications

from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import os
import random
import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import DepthNorm
from torch.utils.data.dataloader import default_collate

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


def my_collate_fn(batch):
    #  fileter NoneType data
    batch = list(filter(lambda x:x['depth'] is not None and x['image'] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)

class DepthDataLoader(object):
    def __init__(self, args, mode):
        transform_data = transforms.Compose([
            transforms.ToTensor(),
        ]
        )
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=transform_data)
            self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'target':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=transform_data)
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, args.batch_size,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=transform_data)
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('error')


class DepthDataLoader_evaluate(object):
    def __init__(self, args, mode):
        transform_data = transforms.Compose([
            transforms.ToTensor(),
        ]
        )
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=transform_data)
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'target':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=transform_data)
            if args.distributed:  # redundant. here only for readability and to be more explicit
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, args.batch_size,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=transform_data)
            if args.distributed:  # redundant. here only for readability and to be more explicit
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'target' or mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])

        if self.mode == 'train':
            if self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
                image_path = os.path.join(self.args.data_path, remove_leading_slash(sample_path.split()[3]))
                depth_path = os.path.join(self.args.gt_path, remove_leading_slash(sample_path.split()[4]))
            else:
                image_path = os.path.join(self.args.data_path, sample_path.split()[0])
                depth_path = os.path.join(self.args.gt_path, sample_path.split()[1])

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    image = image.transpose(2, 0, 1)
            else:
                image = None

            # for exr file, 3 channels are the same
            if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval, doNorm=False)
            else:
                depth_gt = None

            sample = {'image': image, 'depth': depth_gt, 'focal': focal}

        if self.mode == 'target':
            if self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
                image_path = os.path.join(self.args.data_path, remove_leading_slash(sample_path.split()[3]))
                depth_path = os.path.join(self.args.gt_path, remove_leading_slash(sample_path.split()[4]))
            else:
                image_path = os.path.join(self.args.data_path, sample_path.split()[0])
                depth_path = os.path.join(self.args.gt_path, sample_path.split()[1])

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    image = image.transpose(2, 0, 1)
            else:
                image = None

            # for exr file, 3 channels are the same
            if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval, doNorm=False)
            else:
                depth_gt = None

            sample = {'image': image, 'depth': depth_gt, 'focal': focal}

        if self.mode == 'online_eval':
            image_path = os.path.join(self.args.data_path, sample_path.split()[0])
            depth_path = os.path.join(self.args.gt_path, sample_path.split()[1])

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    image = image.transpose(2, 0, 1)
            else:
                image = None

            # for exr file, 3 channels are the same
            if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                     doNorm=False)
            else:
                depth_gt = None

            sample = {'image': image, 'depth': depth_gt, 'focal': focal}

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        transforms.Normalize(mean=[0.53607797, 0.53617338, 0.53618207], std=[0.31895092, 0.31896688, 0.31896867]) # for SUNCG training dataset

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
                    'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
