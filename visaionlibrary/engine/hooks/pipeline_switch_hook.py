from mmcv.transforms import Compose
from mmengine.hooks import Hook

from mmengine.registry import HOOKS


@HOOKS.register_module()
class VisaionPipelineSwitchHook(Hook):
    """Switch data pipeline at switch_iter.

    Args:
        switch_iter (int): switch pipeline at this iter.
        switch_pipeline (list[dict]): the pipeline to switch to.
    """

    def __init__(self, switch_iter, switch_pipeline):
        self.switch_iter = switch_iter
        self.switch_pipeline = switch_pipeline
        self._restart_dataloader = False

    def before_train_iter(self, runner, batch_idx, data_batch):
        """switch pipeline."""
        iter = runner.iter
        train_loader = runner.train_dataloader
        if iter == self.switch_iter:
            runner.logger.info('Switch pipeline now!')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            train_loader.dataset.pipeline = Compose(self.switch_pipeline)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True

        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True
