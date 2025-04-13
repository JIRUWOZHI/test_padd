import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class MM(nn.Layer):
  def __init__(self, out_channels=8):
      super(MM, self).__init__()
      self.conv = nn.Conv2D(3, out_channels, 3, 2, 1)

  def forward(self, x):
    out = self.conv(x)
    out = F.relu(out)
    
    return out