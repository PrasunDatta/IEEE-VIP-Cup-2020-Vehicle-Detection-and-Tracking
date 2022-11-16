_base_ = '/content/vip-cup-2020-team-zodiac/configs/detr/detr_r50_8x4_150e_coco.py'


dataset_type = 'CocoDataset'
classes = ('vehicle',)
data_root = '/content/gdrive/My Drive/data/'


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/content/gdrive/My Drive/data/train/labels_cocoformat.json',
        img_prefix=data_root + 'train/images/'
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/content/gdrive/My Drive/data/test/labels_cocoformat.json',
        img_prefix=data_root + 'test/images/'
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/content/gdrive/My Drive/data/test/labels_cocoformat.json',
        img_prefix=data_root + 'test/images/'
    )
)

custom_imports=dict(
    imports=['mmdet.models.backbones.day_night_backbone'])


model = dict(
        bbox_head = dict(
            num_classes=1
        )
)


optimizer = dict(
    lr=0.00025
)


total_epochs = 1


lr_config = dict(
    gamma=0.1,
    warmup_iters = 500,
    step=[3, 4]
)


test_cfg = dict(
    score_thr=0.05
)


log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)


checkpoint_config = dict(
    create_symlink=False
)


load_from = "/content/gdrive/My Drive/detr_mmdet/detr_r50_8x4_150e_coco.pth"


work_dir = '/content/gdrive/My Drive/detr_mmdet/checkpoints'