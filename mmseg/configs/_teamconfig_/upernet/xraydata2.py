# dataset settings
dataset_type = 'XRayDataset2'
# data_root = '/data/ephemeral/home/data'
crop_size = (1024, 1024)
train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadXRayAnnotations'),
            dict(type='Resize', scale=(512, 512)),
            # dict(type='Albu', transforms=albu_train_transforms),
            dict(type='TransposeAnnotations'),
            dict(type='PackSegInputs')
        ]
test_pipeline = [
            dict(type='LoadImageFromFile'),
            # dict(type='Resize', scale=(1450, 1450)),
            dict(type='LoadXRayAnnotations'),
            dict(type='TransposeAnnotations'),
            dict(type='PackSegInputs')
        ]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    # persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        is_train = True,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        is_train = False,
        pipeline=test_pipeline))
test_dataloader = val_dataloader


val_evaluator = dict(type='DiceMetric')
test_evaluator = dict(type='DiceMetric')
