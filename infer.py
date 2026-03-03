import torch
import yaml

from model import MiniUnet
from rectified_flow import RectifiedFlow
import cv2
import os
import numpy as np


def infer(config):
    """flow matching模型推理
    """
    config = yaml.load(open(config, 'rb'), Loader=yaml.FullLoader)
    checkpoint_path = config.get('checkpoint_path', './checkpoints/v1.1-cfg/miniunet_49.pth')
    base_channels = config.get('base_channels', 16)
    step = config.get('step', 50)  # 采样步数（Euler方法的迭代次数） 10步效果就很好 1步效果不好
    num_imgs = config.get('num_imgs', 5)
    y = config.get('y', 'None')
    cfg_scale = config.get('cfg_scale', 7.0)
    save_path = config.get('save_path', './results')
    save_noise_path = config.get('save_noise_path', 'None')
    device = config.get('device', 'mps')
    os.makedirs(save_path, exist_ok=True)
    if save_noise_path != 'None':
        os.makedirs(save_noise_path, exist_ok=True)

    if y != 'None':
        assert len(y.shape) == 1 or len(
            y.shape) == 2, 'y must be 1D or 2D tensor'
        assert y.shape[0] == num_imgs or y.shape[
            0] == 1, 'y.shape[0] must be equal to num_imgs or 1'
        if y.shape[0] == 1:
            y = y.repeat(num_imgs, 1).reshape(num_imgs)
        y = y.to(device)
    # 生成一些图片
    # 加载模型
    model = MiniUnet(base_channels=base_channels)
    model.to(device)
    model.eval()

    # 加载RectifiedFlow
    rf = RectifiedFlow()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    # with torch.no_grad():  # 无需梯度，加速，降显存
    with torch.no_grad():
        # 无条件或有条件生成图片
        for i in range(num_imgs):
            print(f'Generating {i}th image...')
            # Euler法间隔
            dt = 1.0 / step

            # 初始的x_t就是x_0，标准高斯噪声
            x_t = torch.randn(1, 1, 28, 28).to(device)
            noise = x_t.detach().cpu().numpy()

            # 提取第i个图像的标签条件y_i
            if y != 'None':
                y_i = y[i].unsqueeze(0)

            for j in range(step):
                if j % 10 == 0:
                    print(f'Generating {i}th image, step {j}...')
                t = j * dt
                t = torch.tensor([t]).to(device)

                if y != 'None':
                    # classifier-free guidance需要同时预测有条件和无条件的输出
                    # 利用CFG的公式：x = x_uncond + cfg_scale * (x_cond - x_uncond)
                    # 为什么用score推导的公式放到预测向量场v的情形可以直接用？ SDE ODE
                    v_pred_uncond = model(x=x_t, t=t)
                    v_pred_cond = model(x=x_t, t=t, y=y_i)
                    v_pred = v_pred_uncond + cfg_scale * (v_pred_cond -
                                                          v_pred_uncond)
                else:
                    v_pred = model(x=x_t, t=t)

                # 使用Euler法计算下一个时间的x_t
                x_t = rf.euler(x_t, v_pred, dt)

            # 最后一步的x_t就是生成的图片
            # 先去掉batch维度
            x_t = x_t[0]
            # 归一化到0到1
            # x_t = (x_t / 2 + 0.5).clamp(0, 1)
            x_t = x_t.clamp(0, 1)
            img = x_t.detach().cpu().numpy()
            img = img[0] * 255
            img = img.astype('uint8')
            cv2.imwrite(os.path.join(save_path, f'{i}.png'), img)
            if save_noise_path != 'None':
                # 保存为一个.npy格式的文件
                np.save(os.path.join(save_noise_path, f'{i}.npy'), noise)


if __name__ == '__main__':
    # 每个条件生成10张图像
    # label一个数字出现十次
    y = []
    for i in range(10):
        y.extend([i] * 10)
    # v1.1 1-RF
    infer(config='./config/infer_config.yaml')

    # v1.2 2-RF
    # infer(checkpoint_path='./checkpoints/v1.2-reflow-cfg/miniunet_19.pth',
    #       base_channels=64,
    #       step=2,
    #       num_imgs=100,
    #       y=torch.tensor(y),
    #       cfg_scale=5.0,
    #       save_path='./results/reflow-cfg',
    #       device='cuda')
