import mmdet
from mmdet.apis import DetInferencer
print(mmdet.__version__)

inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco', device="cuda:0")

# Perform inference
inferencer(r"C:\Users\parvs\Downloads\WhatsApp Image 2024-02-22 at 00.29.09_30f01e74.jpg", 
           show=True, 
           # pred_score_thr=0.5, 
           # check for how enter string prompt, and what works, whatsapp image
           )