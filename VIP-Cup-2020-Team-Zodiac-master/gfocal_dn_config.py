_base_ = '/content/vip-cup-2020-team-zodiac/configs/gfl/gfl_r50_fpn_mstrain_2x_coco.py'


dataset_type = 'CocoDataset'
classes = ('vehicle',)
data_root = '/content/gdrive/My Drive/data/'


data = dict(
    samples_per_gpu=8,
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
        backbone = dict(
            type = 'DayNight'
            ),
        bbox_head = dict(
            num_classes=1
        )
)


optimizer = dict(
    lr=0.005,
    #paramwise_cfg=dict(
    #    custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)})
)


total_epochs = 7


lr_config = dict(
    gamma=0.1,
    warmup_iters = 2500,
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


load_from = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/gfl/gfl_r50_fpn_mstrain_2x_coco/gfl_r50_fpn_mstrain_2x_coco_20200629_213802-37bb1edc.pth"


work_dir = '/content/gdrive/My Drive/Gfocal_dn/checkpoints'