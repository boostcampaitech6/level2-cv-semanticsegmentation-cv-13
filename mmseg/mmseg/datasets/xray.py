from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp


@DATASETS.register_module()
class XRayDataset(BaseSegDataset):

    METAINFO = dict(
        classes=['finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
                'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
                'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
                'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
                'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
                'Triquetrum', 'Pisiform', 'Radius', 'Ulna',],
        palette=[(120, 203, 228), (145, 42, 177), (210, 71, 77), (193, 223, 159), (139, 26, 26), (209, 146, 117),
                (205, 0, 0), (255, 99, 71), (159, 95, 159), (238, 221, 139), (255, 248, 220), (238, 238, 209),
                (250, 235, 215), (205, 149, 140), (51, 153, 204), (205, 133, 63), (240, 140, 130), (255, 193, 193),
                (168, 168, 168), (0, 0, 255), (0, 0, 128), (132, 112, 255), (47, 79, 47), (255, 0, 0),
                (112, 219, 219), (122, 103, 238), (205, 51, 51), (127, 255, 0), (0, 255, 0)])

    def __init__(self, data_root, data_prefix, pipeline):
        super(XRayDataset, self).__init__(
            data_root=data_root, data_prefix=data_prefix, pipeline=pipeline,)
            # img_suffix='.png', seg_map_suffix='.png')