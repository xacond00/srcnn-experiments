import os
import requests
import zipfile
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import random
from concurrent.futures import ThreadPoolExecutor

class RandomRotate90(torch.nn.Module):
    def __init__(self, enable = True):
        super().__init__()
        self.enable = enable
        self.rotates = [0, 1, 2, 3]

    def forward(self, img):
        if not self.enable: return img
        k = random.choice(self.rotates)
        return torch.rot90(img, k, dims=(1, 2))

class ImageDataset(Dataset):
    def __init__(self, dataset_name="DIV2K", train : bool = True, scale : int= 4, downscale : int = 1, crop : int = 1024, cache_size = 'full', pre_crop = None, download_only : bool = False):
        """
        Args:
            dataset_name (str): Either 'DIV2K' or 'Flickr2K'.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.train = train
        self.crop = crop
        self.downscale = downscale
        self.scale = scale
        self.dataset_name = dataset_name
        self.dataset_urls = {
            "CUSTOM" : "CUSTOM", 
            "DIV2K": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
            "DIV2KVal": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
            "Flickr2K": "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar",
            'COCO2017V' : 'http://images.cocodataset.org/zips/val2017.zip',
            'COCO2017T' : 'http://images.cocodataset.org/zips/test2017.zip'
        }
        if dataset_name not in self.dataset_urls:
            print("Invalid dataset !")
            exit(0)
        if dataset_name == "Flickr2K":
            self.dataset_folder = dataset_name
        elif dataset_name == "DIV2KVal":
            self.dataset_folder = "DIV2KVal/DIV2K_valid_HR/" 
        elif dataset_name == "COCO2017V":
            self.dataset_folder = "COCO2017V/val2017" 
        elif dataset_name == "COCO2017T":
            self.dataset_folder = "COCO2017T/test2017" 
        elif dataset_name == "DIV2K":
            self.dataset_folder = "DIV2K/DIV2K_train_HR/"
        else:
            self.dataset_folder = "CUSTOM"

        print(self.dataset_folder)
        self.pre_crop = pre_crop
        self.v2cvt = v2.ToImage()
        # Ensure dataset is available
        if not os.path.exists(self.dataset_folder):
            self.download_and_extract()

        # if totally custom dataloder such as ResShift, end
        if download_only:
            return

        self.images = [os.path.join(self.dataset_folder, f) 
                            for f in sorted(os.listdir(self.dataset_folder))
                            if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        if(cache_size == 'full'): cache_size = len(self.images)
        elif(cache_size == 'half'): cache_size = len(self.images) // 2
        elif(cache_size == 'quar'): cache_size = len(self.images) // 4
        elif(cache_size == 'eigh'): cache_size = len(self.images) // 8
        else:  cache_size = min(cache_size, len(self.images))
        self.en_cache = cache_size > 0
        if self.en_cache:
            self.cache = [None] * cache_size
            self.cache_size = cache_size
            def load_and_convert_image(i):
                with Image.open(self.images[i]) as img:
                    self.cache[i] = self.img_cvt(img.convert('RGB'))
                    
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(load_and_convert_image, i) for i in range(cache_size)]
                for future in futures:
                    future.result()

    def img_cvt(self, img):
        if self.pre_crop:
            p = self.pre_crop
            size = (min(img.size(1), p), min(img.size(2), p))
            tran = v2.Compose(self.v2cvt, v2.CenterCrop(size)) 
            return tran(img)
        else: 
            return self.v2cvt(img)

    def cached_img(self, i):
        if self.en_cache and i < self.cache_size:
            return self.cache[i]
        else:
            with Image.open(self.images[i]) as img:
                return self.img_cvt(img.convert('RGB'))     
    
    def get_transforms(self, train, osize, dims, crop : int = 0):
        if(osize[0] < crop and osize[1] < crop):
            fn_size = v2.Resize(size=(crop, crop))
        elif(osize[0] < crop):
            fn_size = v2.Resize(size=(crop, osize[1]))
        elif(osize[1] < crop):
            fn_size = v2.Resize(size=(osize[0], crop))
        else:
            fn_size = v2.Identity()
        if(crop):
            fn_crop = v2.RandomCrop(size=(crop, crop)) if train else v2.CenterCrop(size=(crop,crop))
        else:
            fn_crop = v2.Identity()
        if(dims):
            fn_resize = v2.Resize(size=dims)
        else:
            fn_resize = v2.Identity()
        fn_flip = v2.Compose([RandomRotate90(), v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()]) if train else v2.Identity()
        return v2.Compose([
            fn_size,
            fn_crop,
            fn_resize,
            fn_flip,
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __getitem__(self, i):
        return self.load_img(i, self.scale, self.downscale, self.crop, self.train)

    def load_img(self, i, scale : int = 4, downscale : int = 1, crop : int = 1024, train = False):
        img = self.cached_img(i)        
        in_size = (img.size(1), img.size(2))
        size = (crop, crop) if crop else in_size
        size = [s // downscale for s in size]
        dims = None if downscale == 1 else size
        trafo = self.get_transforms(train, in_size, dims, crop)
        hr = trafo(img)

        lr_size = [s // scale for s in size]
        lr_scale = v2.Compose([v2.Resize(size=lr_size)])
        return lr_scale(hr), hr

    def download_and_extract(self):
        """Downloads and extracts the dataset if not found."""
        os.makedirs(self.dataset_name, exist_ok=True)
        url = self.dataset_urls.get(self.dataset_name)

        if not url:
            raise ValueError(f"Dataset {self.dataset_name} not supported!")

        filename = os.path.join(self.dataset_name, url.split("/")[-1])

        # Download dataset if not exists
        if not os.path.exists(filename):
            print(f"Downloading {self.dataset_name} dataset...")
            self._download_file(url, filename)

        # Extract dataset
        print(f"Extracting {self.dataset_name} dataset...")
        self._extract_file(filename)

    def _download_file(self, url, filename):
        """Helper method to download a file with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB

        with open(filename, "wb") as file, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

    def _extract_file(self, filepath):
        """Extracts ZIP or TAR files."""
        if filepath.endswith(".zip"):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_name)
        elif filepath.endswith(".tar"):
            os.system(f"tar -xf {filepath} -C {self.dataset_name}")
        else:
            raise ValueError("Unsupported file format!")
        
    def __len__(self):
        return len(self.images)
