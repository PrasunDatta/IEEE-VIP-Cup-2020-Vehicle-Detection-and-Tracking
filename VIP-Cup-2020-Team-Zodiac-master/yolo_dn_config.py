_base_ = '/content/vip-cup-2020-team-zodiac/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'


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
    imports=['mmdet.models.backbones.darknet_dn_backbone'])


model = dict(
        backbone = dict(
            type = 'DarkDNnet'
            ),
        bbox_head = dict(
            num_classes=1
        )
)


optimizer = dict(
    lr=0.0005
)


total_epochs = 1


lr_config = dict(
    gamma=0.1,
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


load_from = "/content/gdrive/My Drive/yolo_mmdet/yolov3_d53_mstrain-608_273e_coco-139f5633.pth"


work_dir = '/content/gdrive/My Drive/yolo_mmdet/checkpoints'