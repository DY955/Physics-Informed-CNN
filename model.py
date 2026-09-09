import torch
import torch.nn as nn

class PhysicsInformedCNN(nn.Module):
    def __init__(self):
        super(PhysicsInformedCNN, self).__init__()
        # Image Processing Branch
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Feature Fusion & Prediction Head
        self.fc_layer = nn.Sequential(
            nn.Linear(16 * 32 * 32 + 2, 64), # Combined spatial + crystallographic features
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, image_data, crystal_data):
        spatial_features = self.conv_layer(image_data)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)
        fused_features = torch.cat((spatial_features, crystal_data), dim=1)
        return self.fc_layer(fused_features)

def physics_informed_loss(predictions, targets, misorientations, schmid_factors, lambda1=0.1, lambda2=0.1):
    # L_data: Standard binary cross-entropy
    bce_loss = nn.BCELoss()(predictions, targets)
    # L_misorient & L_Schmid: Physics penalties
    physics_penalty = lambda1 * torch.mean(torch.relu(misorientations - 15.0)) + \
                      lambda2 * torch.mean(torch.relu(0.5 - schmid_factors))
    return bce_loss + physics_penalty
