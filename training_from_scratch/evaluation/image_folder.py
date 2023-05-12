
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS


class ImageSubfolder(DatasetFolder):
    """Extend ImageFolder to support fold subsets
    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        class_to_idx (dict): Dict with items (class_name, class_index).
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        class_to_idx: Optional[Dict] = None,
    ):
        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        if class_to_idx is not None:
            classes = class_to_idx.keys()
        else:
            classes, class_to_idx = self.find_classes(self.root)
        extensions = IMG_EXTENSIONS if is_valid_file is None else None,
        samples = self.make_dataset(self.root, class_to_idx, extensions[0], is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples
