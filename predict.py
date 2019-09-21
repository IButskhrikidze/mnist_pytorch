import main
import torch
from main import pil_to_tensor
from torch.autograd import Variable
from PIL import Image

net = main.MyConvNet()

filename = './trained_model/weights.pth'

net.load_state_dict(torch.load(filename))


def Preprocess(img):
    img = pil_to_tensor(img)

    img = Variable(img, requires_grad=False)

    return img

def Predict():
    img = Image.open('./trained_model/input_image.jpg')
    pre = net(Preprocess(img))
    x = pre[0].max(0)[1]
    return int(x)
