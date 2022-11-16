_base_ = '/content/vip-cup-2020-team-zodiac/configs/gfl/gfl_r50_fpn_mstrain_2x_coco.py'


dataset_type = 'CocoDataset'
classes = ('day_vehicle','night_vehicle')
data_root = '/content/gdrive/My Drive/data/'


data = dict(
    samples_per_gpu=8, 
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/content/gdrive/My Drive/Gfocal_dn_correct/trainlabels_dn.json',
        img_prefix=data_root + 'train/images/'
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/content/gdrive/My Drive/Gfocal_dn_correct/testlabels_dn.json',
        img_prefix=data_root + 'test/images/'
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/content/gdrive/My Drive/Gfocal_dn_correct/testlabels_dn.json',
        img_prefix=data_root + 'test/images/'
    )
)


model = dict(
        bbox_head = dict(
            num_classes=2
        )
)


optimizer = dict(
    lr=0.005
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


load_from = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/gfl/gfl_r50_fpn_mstrain_2x_coco/gfl_r50_fpn_mstrain_2x_coco_20200629_213802-37bb1edc.pth"


work_dir = '/content/gdrive/My Drive/Gfocal_dn_correct/checkpoints'