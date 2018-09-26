
##  Image Segmentation 
**Problem:** Perform image segmentation for given image using color and spetial similarity

**Approach:** Implemented mean-shift algorithm based on 4 dimensional features (rgb and distance) to partition image into segmentation.

[image-segmentation.py](image-segmentation.py)

**Input image:**

![Butterfly.jpg](Butterfly.jpg)

**Output Images:**

Segmented using h = 60 and shift = 1

![params_h-60_shift-1.PNG](output/params_h-60_shift-1.PNG)

Segmented using h = 90 and shift = 1

![params_h-90_shift-1.PNG](output/params_h-90_shift-1.PNG)

Segmented using h = 120 and shift = 1

![params_h-120_shift-1.PNG](output/params_h-120_shift-1.PNG)
