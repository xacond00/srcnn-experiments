import os
import requests
import zipfile
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import threading

class ImageDataset(Dataset):
    def __init__(self, dataset_name="DIV2K", train : bool = True, scale : int= 4, downscale : int = 1, crop : int = 1024, cache_size = 1000):
        """
        Args:
            root_dir (str): Directory to store/download datasets.
            dataset_name (str): Either 'DIV2K' or 'Flickr2K'.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.train = train
        self.crop = crop
        self.downscale = downscale
        self.scale = scale

        self.dataset_name = dataset_name

        if dataset_name == "Flickr2K":
            self.dataset_folder = dataset_name
        elif dataset_name == "DIV2KVal":
            self.dataset_folder = "DIV2KVal/DIV2K_valid_HR/" 
        else:
            self.dataset_folder = "DIV2K/DIV2K_train_HR/"

        print(self.dataset_folder)
         
        self.dataset_urls = {
            "DIV2K": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
            "DIV2KVal": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
            "Flickr2K": "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"
        }
        # Ensure dataset is available
        if not os.path.exists(self.dataset_name):
            self.download_and_extract()

        self.images = [os.path.join(self.dataset_folder, f) 
                            for f in os.listdir(self.dataset_folder) 
                            if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.en_cache = cache_size > 0
        if(self.en_cache):
            self.cache = dict()
            self.cache_size = cache_size  # Max cached images
            self.lock = threading.Lock()  # Thread safety

    def cached_img(self, i):
        if not self.en_cache:
            with Image.open(self.images[i]) as img:
                return img.convert('RGB')

        with self.lock:  # Ensure thread safety
            if i in self.cache:
                return self.cache[i]  # Return cached image

        # Load image from disk if not cached
        with Image.open(self.images[i]) as img:
            img = img.convert('RGB')

        with self.lock:
            if len(self.cache) < self.cache_size:
                self.cache[i] = img

        return img
    
    def get_transforms(self, train, osize, dims, crop : int = 0):
        if(osize[0] < crop):
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
        return v2.Compose([
            v2.ToImage(),
            fn_size,
            fn_crop,
            fn_resize,
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __getitem__(self, i):
        return self.load_img(i, self.scale, self.downscale, self.crop, self.train)

    def load_img(self, i, scale : int = 4, downscale : int = 1, crop : int = 1024, train = False):
        img = self.cached_img(i)        
        in_size = (img.height, img.width)
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
