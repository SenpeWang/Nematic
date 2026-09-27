# -*- coding: utf-8 -*-
"""轻量级 TensorBoard 日志记录器.

TBLogger: 基类 — SummaryWriter 封装, 可继承扩展
NPRStats: 向列相残差统计 — 按 stage 累积, epoch 结束 flush
"""


class TBLogger:
    """轻量 TensorBoard 记录器基类.

    封装 SummaryWriter, 提供标量写入 + 步数管理.
    可继承扩展自定义记录逻辑.
    """

    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def scalar(self, tag, value):
        self.writer.add_scalar(tag, float(value), self.step)

    def set_step(self, step):
        self.step = step

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


class NPRStats(TBLogger):
    """向列相残差统计记录器 (按 stage).

    forward 中只累积, epoch 结束调 flush_epoch() 写入 TB.
    每 stage 记录:
        Q/{name}/S             — Q-tensor 序参量
        Q/{name}/gate          — 门控均值
        Q/{name}/scale         — 全局缩放
    """

    def __init__(self, log_dir):
        super().__init__(log_dir)
        self._buf = {}
        self._nsg_buf = {}

    def log_q_update(self, name, gate_mean, scale, S_after):
        """累积单次 Q-tensor 更新统计 (不立即写 TB)."""
        buf = self._buf.setdefault(name, {'gate': [], 'scale': [], 'S': []})
        buf['gate'].append(gate_mean)
        buf['scale'].append(scale)
        buf['S'].append(S_after)

    def log_nsg(self, name, gate_mean):
        """累积 NSG 门控值."""
        self._nsg_buf.setdefault(name, []).append(gate_mean)

    def flush_epoch(self):
        """写入本 epoch 累积的均值到 TB, 然后清空."""
        for name, buf in self._buf.items():
            if buf['gate']:
                n = len(buf['gate'])
                self.scalar(f'Q/{name}/S',
                            sum(buf['S']) / n)
                self.scalar(f'Q/{name}/gate',
                            sum(buf['gate']) / n)
                self.scalar(f'Q/{name}/scale',
                            sum(buf['scale']) / n)
        self._buf.clear()
        for name, vals in self._nsg_buf.items():
            if vals:
                self.scalar(f'NSG/{name}/gate', sum(vals) / len(vals))
        self._nsg_buf.clear()
        self.flush()
