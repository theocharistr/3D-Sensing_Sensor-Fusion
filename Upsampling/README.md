Depth Image  
![alt text](https://github.com/theocharistr/3D-Sensing_Sensor-Fusion/blob/master/Upsampling/data/disp1.png)  
Guided by original RGB Image  
![alt text](https://github.com/theocharistr/3D-Sensing_Sensor-Fusion/blob/master/Upsampling/data/view0.png)  
Upsampled Image  
![alt text](https://github.com/theocharistr/3D-Sensing_Sensor-Fusion/blob/master/Upsampling/data/Upsampled.png)  

Bilateral filter(for the combinations of 4 different spatial, 4 different spectral sigmas)on the image of my choosing and the upsampled image after Guided Joint bilateral filter to upsample my depth image, guided by the coressponding RGB image.  
As the range parameter σr (range kernel, spectral sigma)  increases, the bilateral filter gradually approximates Gaussian convolution more closely because the range Gaussian Gσr widens and flattens, i.e., is nearly constant over the intensity interval of the image. While, Increasing the spatial parameter σs smooths larger features.
