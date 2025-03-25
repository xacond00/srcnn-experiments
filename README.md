SRCNN upscaling experiments

Has SRCNN class that allows creation of normal subpixel CNN with:
1) Residual blocks
2) Convolutional blocks
3) Auxiliary upscaler (improves performance)

Best performing so far is "c5x64x2_c3x64x5_aux" model with "c5x4" auxiliary upscaler.