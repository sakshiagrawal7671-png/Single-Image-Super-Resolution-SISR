import torch
import cv2
import matplotlib.pyplot as plt
from src.srcnn_model import SRCNN

model = SRCNN()
model.load_state_dict(torch.load("srcnn.pth", map_location="cpu"))
model.eval()

img = cv2.imread("sample_images/input_lr.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()

with torch.no_grad():
    sr = model(img).squeeze(0).permute(1,2,0).numpy()

plt.imshow(sr)
plt.title("SRCNN Output")
plt.axis("off")
plt.show()
plt.imsave("sample_images/output_srcnn.png", sr)
