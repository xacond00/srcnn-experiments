RESULTS:
https://drive.google.com/drive/folders/16FCOaLuFvgzC1v0jG7NbsmuZq_6kACBz?hl=cs

SRCNN upscaling experiments

Has SRCNN class that allows creation of normal subpixel CNN with:
1) Residual blocks
2) Convolutional blocks
3) Auxiliary upscaler (improves performance)

Best performing so far is "c5x64x2_c3x64x5_aux" model with "c5x4" auxiliary upscaler.