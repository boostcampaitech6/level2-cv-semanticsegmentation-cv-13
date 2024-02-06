default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='MMSegWandbHook',
            by_epoch=False,
            interval=1,
            with_step=False,
            init_kwargs=dict(
                entity='frostings',
                project='Boost Camp Lv2-3',
                name='mmsegtest'),
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=10)
    ])

tta_model = dict(type='SegTTAModel')
