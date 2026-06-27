from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class EpochSyncHook(Hook):
    def after_train_epoch(self, runner):
        # step1: 兼容DDP和单卡
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        # step2: 你的OrientedStandardRoIHeadWithContrast是在roi_head
        if hasattr(model, 'roi_head') and hasattr(model.roi_head, 'contrastive_head'):
            # 这里的epoch就是刚结束的epoch，比如第一轮刚完就是1
            model.roi_head.current_epoch = runner.epoch + 1
            model.roi_head.contrastive_head.set_epoch(runner.epoch + 1)
            print(f"[HOOK DEBUG] Set contrastive_head cur_epoch={runner.epoch+1}")
