import torch
import torchvision.models as models 

class FeaturesExtractor(object):
  def __init__(self):
    backbone = models.resnet50(pretrained=True)
    modules=list(backbone.children())[:-1]
    self.model =torch.nn.Sequential(*modules)
    for p in self.model.parameters():
        p.requires_grad = False

  def getFeatures(self, im):
    im = torch.FloatTensor(im).permute(2,0,1)
    im = im.view(1,im.size(0),im.size(1),im.size(2))
    return self.model(im).squeeze()

  def getFeaturesOfList(self, ims):
    ims = torch.FloatTensor(ims).permute(0,3,1,2)
    return self.model(ims).squeeze()