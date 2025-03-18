import torch
import torch.nn as nn
import torch.nn.functional as F

class KeyPointDetector(nn.Module):
    def __init__(self):
        super(KeyPointDetector, self).__init__()
        self.conv1 = nn.Conv2d(33, 64,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64, 64,3)
        self.fc1 = nn.Linear(64*62*62, 64)
        self.fc2 = nn.Linear(64, 6)

        self.conv1_md = nn.Conv2d(256, 128, 3,padding=1)
        self.conv2_md = nn.Conv2d(128, 64, 3,padding=1)
        self.conv3_md = nn.Conv2d(64, 32, 3,padding=1)


    def forward(self, features,masks):
        device = torch.device('cuda')
        # masks = torch.zeros(2,1,256,256).to(device)
        features = F.interpolate(features, size=(64, 64), mode='nearest')
        x = nn.functional.relu(self.conv1_md(features))
        x = F.interpolate(x, size=(128, 128), mode='nearest')
        x = nn.functional.relu(self.conv2_md(x))
        x = F.interpolate(x, size=(256, 256), mode='nearest')
        x = nn.functional.relu(self.conv3_md(x))
        # x = self.pool_md(nn.functional.relu(self.conv4_md(x)))
        # x = F.interpolate(x, size=(256, 256), mode='nearest')
        x_ = torch.cat((x, masks), dim=1) # (2,33,256,256)

        x = self.pool(nn.functional.relu(self.conv1(x_)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64*62*62)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# model = KeyPointDetector()
