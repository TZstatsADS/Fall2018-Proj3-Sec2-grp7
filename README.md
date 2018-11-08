# Project: Can you unscramble a blurry image? 
![image](figs/example.png)

### [Full Project Description](doc/project3_desc.md)

Term: Fall 2018

+ Team #7
+ Team members
	+ Guo, Yaoqi  yg2542@columbia.edu
	+ Loewenstein, Oded orl2108@columbia.edu
	+ Ma, Yunsheng  ym2650@columbia.edu
	+ Zhang, Yixin  yz3223@columbia.edu
	+ Zheng, Wanyi  wz2409@columbia.edu

+ Project summary: 
	+ **GBM model**: we created a classification engine for enhance the resolution of images. Our baseline model is GBM. Using cross-validation, we compare the performance of models with different specifications and based on the MSE we chose depth = 2. Then we test it on 1500 images and get `psnr = 24.67765`.
	+ **SRCNN model**: we implement a CNN for super resolution that consists of three layers of relu convolution. Using independent testing (500 test images and the rest for training), we get `psnr = 27.93`.
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement. 

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
