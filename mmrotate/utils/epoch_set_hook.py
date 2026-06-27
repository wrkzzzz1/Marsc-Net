from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class GradientAccumulationHook(Hook):
    """实现梯度累计的小hook。【仅适用于mmrotate0.3.4及mmdet2.x】"""
    def __init__(self, cumulative_iters=16):
        self.cumulative_iters = cumulative_iters

    def after_train_iter(self, runner):
        # 注意：这里逻辑只适用于 epoch-based runner，iter-based runner也可用但末尾处理稍调整
        if (runner.iter + 1) % self.cumulative_iters == 0 or (runner.inner_iter + 1) == len(runner.data_loader):
            runner.optimizer.step()
            runner.model.zero_grad()
        else:
            pass    # 不step