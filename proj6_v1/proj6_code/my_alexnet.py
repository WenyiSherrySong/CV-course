import torch
import torch.nn as nn

from torchvision.models import alexnet


class MyAlexNet(nn.Module):
    def __init__(self):
        '''
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Ready Pytorch documention
        to understand what it means

        Note: Do not forget to freeze the layers of alexnet except the last one

        Download pretrained alexnet using pytorch's API (Hint: see the import
        statements)
        '''
        # super(MyAlexNet, self).__init__()
        super().__init__()

        self.cnn_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ###########################################################################
        # Student code begin
        ###########################################################################

        # freezing the layers by setting requires_grad=False
        # example: self.cnn_layers[idx].weight.requires_grad = False

        # take care to turn off gradients for both weight and bias
        model = alexnet(pretrained=True)
        self.cnn_layers = nn.Sequential(
            *list(model.children())[0][:],
            # *list(model.children())[1][:]
        )
        # self.cnn_layers.weight.requires_grad = False
        # self.cnn_layers.bias.requires_grad = False
        for param in self.cnn_layers.parameters():
            param.requires_grad = False
        self.fc_layers = nn.Sequential(
            *list(model.children())[2][:-1],
            nn.Linear(4096, 15)
        )
        self.fc_layers[1].weight.requires_grad = False
        self.fc_layers[1].bias.requires_grad = False
        self.fc_layers[4].weight.requires_grad = False
        self.fc_layers[4].bias.requires_grad = False
        self.loss_criterion = nn.CrossEntropyLoss(
            size_average=None, reduce=None, reduction='sum')

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        '''

        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images

        ###########################################################################
        # Student code begin
        ###########################################################################
        # self.cnn_layers.weight.requires_grad = False
        # self.cnn_layers.bias.requires_grad = False
        # self.fc_layers[0].weight.requires_grad = False
        # self.fc_layers[0].bias.requires_grad = False
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        model_output = self.fc_layers(x)

        ###########################################################################
        # Student code end
        ###########################################################################
        return model_output
