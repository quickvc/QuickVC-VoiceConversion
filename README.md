# QuickVC
This repository contains the open source code, audio samples and pretrained models of my paper: QuickVC: Any-to-many Voice Conversion Using Inverse Short-time Fourier Transform for Faster Conversion
## [Demo Page](https://quickvc.github.io/quickvc-demo)
<img src="qvcfinalwhite.png" width="100%">

## [Pretrained Model](https://drive.google.com/drive/folders/1DF6RgIHHkn2aoyyUMt4_hPitKSc2YR9d?usp=share_link)
Put pretrained model into logs/quickvc

## Inference with pretrained model
  ㄴpython convert.py

## Train
  ㄴpython train.py
 If you want to change the config and model name, change:
 
  parser.add_argument('-c', '--config', type=str, default="./configs/quickvc.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str,default="quickvc",
                      help='Model name')
                      
  in utils.py

## References
If you have any question about the decoder, refer to [MS-ISTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS).

If you have any question about the Hubert-soft, refer to [Soft-VC](https://github.com/bshall/hubert).

If you have any question about the data augumentation, refer to [FreeVC](https://github.com/OlaWod/FreeVC).
## If you meet any problem, welcome to contact with me.
