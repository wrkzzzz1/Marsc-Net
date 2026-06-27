import torch
import mmrotate.models  # 注册模型
from mmdet.apis import init_detector
from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader

cfg = Config.fromfile('')
model = init_detector(cfg, '', device='cuda:0')
model.eval()

dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, shuffle=False)

device = 'cuda:0'

results = []
for data in data_loader:
    with torch.no_grad():
        imgs = [data['img'][0].data[0].unsqueeze(0).to(device)]  # 转移到GPU
        img_metas = [data['img_metas'][0].data[0]]  # img_metas通常是CPU上的list/dict，保持不动
        result = model(return_loss=False, rescale=True, img=imgs, img_metas=img_metas)
    results.append(result)



# 评估
eval_results = dataset.evaluate(results, metric='mAP')
print(eval_results)