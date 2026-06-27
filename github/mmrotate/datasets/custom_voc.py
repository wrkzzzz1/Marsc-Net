from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets import VOCDataset

@ROTATED_DATASETS.register_module()
class CustomVOCDataset(VOCDataset):
    def get_img_info(self, idx):
        img_id = self.data_infos[idx]['id']
        filename = f'{img_id}.bmp'  # bmp后缀
        return dict(filename=filename)