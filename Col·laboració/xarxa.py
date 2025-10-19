class ResBlock(nn.Module):
    def __init__(self,  in_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return self.relu(res + x)
class SRNet(nn.Module):
    def __init__(self, sampling, features=64, kernel_size=3, blocks=3, channels=2):
        super(SRNet, self).__init__()
        self.add_features = nn.Conv2d(channels, features, kernel_size=kernel_size, padding=kernel_size//2)
        self.residual = nn.ModuleList([ResBlock(features, kernel_size=kernel_size) for _ in range(blocks)])
        self.obtain_channels = nn.Conv2d(features, channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, low):
        features = self.add_features(low)
        for res_block in self.residual:
            features = res_block(features)
        res = self.obtain_channels(features)
        return res