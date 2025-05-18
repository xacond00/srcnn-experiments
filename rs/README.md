pip install -r requirements.txt

## inference
# pred spustenim inference je potreba rozjet trenovani a pockat nez zacne trening, aby s enacetly vsechny potrebne soubory jako vqgan atd!

# -ig: adresar s HR vstupy, rovnou spocita i SSIM/PSNR
# -i: adresar s LR vstupy
# -o: vysledek SR, defaultne ./results

!python inference_resshift.py -ig /content/drive/MyDrive/KNN-2025/hr_div2k -i /content/drive/MyDrive/KNN-2025/lr_div2k


## trenovani
# v konfiguracnim souboru v conf/ bud zakomentovat resume_url/resume/ema_resume/resume_ema_url nahore, nebo model.ckpt_path nastavit na ~ a zakomentovat model.ckpt_url . Model budto pokracuje v trenovani, nebo si stahne predtrenovany model a pokracuje s nim. 

# flops_only - pokud True, tak dochazi jen k vypoctu flops/px.

!python train_resshift.py