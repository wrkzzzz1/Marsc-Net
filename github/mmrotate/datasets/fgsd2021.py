import os.path as osp
import numpy as np
import xml.etree.ElementTree as ET
from mmrotate.datasets.dota import DOTADataset
from mmdet.datasets.builder import DATASETS

@DATASETS.register_module()
class FGSD2021Dataset(DOTADataset):
    """FGSD2021 Dataset (VOC XML) - 中文类别名"""

    CLASSES = (
        '仁慈级', '佩里级', '供应级', '其他', '凯泽级', '刘易斯和克拉克级',
        '圣安东尼奥级', '塔瓦拉级', '复仇者级', '奥斯汀级', '惠特贝岛级',
        '提康德罗加级', '新港级', '潜艇', '独立级', '自由级',
        '航母', '阿利·伯克级', '霍普级', '黄蜂级'
    )

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_prefix,
                 img_subdir='images',
                 ann_subdir='annotations',
                 version='le90',
                 **kwargs):
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.version = version
        super(FGSD2021Dataset, self).__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            img_prefix=img_prefix,
            version=version,
            **kwargs)

    def load_annotations(self, ann_file):
        img_infos = []
        with open(ann_file) as f:
            img_ids = [x.strip() for x in f if x.strip()]

        for img_id in img_ids:
            # 优先按 img_subdir / ann_subdir 路径找
            filename = osp.join(self.img_prefix, self.img_subdir, f'{img_id}.jpg')
            xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')

            # 如果子文件夹不存在，就直接在 img_prefix 根目录找
            if not osp.exists(filename) or not osp.exists(xml_path):
                alt_filename = osp.join(self.img_prefix, f'{img_id}.jpg')
                alt_xml_path = osp.join(self.img_prefix, f'{img_id}.xml')
                if osp.exists(alt_filename) and osp.exists(alt_xml_path):
                    filename = alt_filename
                    xml_path = alt_xml_path
                else:
                    print(f'[WARN] Missing {img_id} -> {filename} / {xml_path}')
                    continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            bboxes, labels, polygons = [], [], []

            for obj in root.findall('object'):
                cname = obj.find('name').text.strip()
                if cname not in self.CLASSES:
                    continue
                label = self.CLASSES.index(cname)

                robndbox = obj.find('robndbox')
                if robndbox is not None:
                    cx = float(robndbox.find('cx').text)
                    cy = float(robndbox.find('cy').text)
                    w = float(robndbox.find('w').text)
                    h = float(robndbox.find('h').text)
                    angle = float(robndbox.find('angle').text)
                    bboxes.append([cx, cy, w, h, angle])
                    polygons.append(self.rbox2poly(cx, cy, w, h, angle))
                    labels.append(label)
                    continue

                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    w = xmax - xmin
                    h = ymax - ymin
                    angle = 0.0
                    bboxes.append([cx, cy, w, h, angle])
                    polygons.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
                    labels.append(label)

            if bboxes:
                bboxes = np.array(bboxes, dtype=np.float32)
                labels = np.array(labels, dtype=np.int64)
                polygons = np.array(polygons, dtype=np.float32)
            else:
                bboxes = np.zeros((0, 5), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
                polygons = np.zeros((0, 8), dtype=np.float32)

            img_infos.append(
                dict(
                    id=img_id,
                    filename=filename,
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=bboxes,
                        labels=labels,
                        polygons=polygons
                    )
                )
            )
        return img_infos

    @staticmethod
    def rbox2poly(cx, cy, w, h, angle):
        import math
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        pts = [
            (-w / 2, -h / 2),
            ( w / 2, -h / 2),
            ( w / 2,  h / 2),
            (-w / 2,  h / 2)
        ]
        poly = []
        for x, y in pts:
            px = cx + cos_a * x - sin_a * y
            py = cy + sin_a * x + cos_a * y
            poly.extend([px, py])
        return poly
