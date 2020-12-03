# SyntheticUrduDataset
Synthetic Urdu Dataset for Outdoor Text Detection and Recognition

Urdu text is a cursive script and belongs to a non-Latin family of other cursive scripts like Arabic, Chinese, and Hindi. Urdu text poses a challenge for detection/localization from natural scene images, and consequently recognition of individual ligatures in scene images. In this paper, a methodology is proposed that covers detection, orientation prediction, and recognition of Urdu ligatures in outdoor images. As a first step, the custom FasterRCNN algorithm has been used in conjunction with well-known CNNs like Squeezenet, Googlenet, Resnet18, and Resnet50 for detection and localization purposes for images of size 320 × 240 pixels. For ligature Orientation prediction, a custom Regression Residual Neural Network (RRNN) is trained/tested on datasets containing randomly oriented ligatures. Recognition of ligatures was done using Two Stream Deep Neural Network (TSDNN). In our experiments, five-set of datasets, containing 4.2K and 51K Urdu-text-embedded synthetic images were generated using the CLE annotation text to evaluate different tasks of detection, orientation prediction, and recognition of ligatures. These synthetic images contain 132, and 1600 unique ligatures corresponding to 4.2K and 51K images respectively, with 32 variations of each ligature (4-backgrounds and font 8-color variations). Also, 1094 real-world images containing more than 12k Urdu characters were used for TSDNN's evaluation. Finally, all four detectors were evaluated and used to compare them for their ability to detect/localize Urdu-text using average-precision (AP). Resnet50 features based FasterRCNN was found to be the winner detector with AP of.98. While Squeeznet, Googlenet, Resnet18 based detectors had testing AP of.65, .88, and .87 respectively. RRNN achieved and accuracy of 79% and 99% for 4k and 51K images respectively. Similarly, for characters classification in ligatures, TSDNN attained a partial sequence recognition rate of 94.90% and 95.20% for 4k and 51K images respectively. Similarly, a partial sequence recognition rate of 76.60% attained for real world-images.










Please cite our dataset as!!

Arafat, Syed Yasser, and Muhmmad Javed Iqbal. "Urdu-Text Detection and Recognition in Natural Scene Images Using Deep Learning." IEEE Access (2020).
