_base_ = '/content/vip-cup-2020-team-zodiac/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'


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
          roi_head = dict(
                bbox_head = dict(
                    num_classes=1
                )
          )
)


optimizer = dict(
    lr=0.005
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


load_from = "/content/gdrive/My Drive/frcnn_mmdet/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth.pth"


work_dir = '/content/gdrive/My Drive/frcnn_mmdet/checkpoints'