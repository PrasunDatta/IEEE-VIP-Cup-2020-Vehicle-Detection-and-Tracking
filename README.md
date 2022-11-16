# IEEE-VIP-Cup-2020-Vehicle-Detection-and-Tracking
This research work is mainly based on the proposed problem of IEEE VIP Cup where the main task was to detect and track vehicles at junction using a fish-eye camera.
## Introduction

<p align ="justify">
With the increasing growth of urbanization, it introduces traffic jams and congestion in several locations around the city. Apart from accidents, that may result in drastic average travel time increase from point A to point B in a city. Especially junctions are critical since delays and accidents tend to be concentrated at these places. Under these circumstances, intelligent traffic systems are unavoidable that are capable of tasks such as vehicle detection, tracking, violation detection and congestion control. </p>
<p align ="justify">
The 2020 VIP-cup challenge focuses on fisheye cameras mounted into street lamps at junctions and vehicle detection and tracking to be used for a junction management system to optimize the flow of traffic and synchronize with other junctions to obtain bottleneck performances throughout the city. Fisheye cameras are used since they tend to be promising in terms of reliability and scene coverage at a chosen junction. They provide 360 degrees of observation view, thus introducing key changes in traffic management.</p>
<p align ="justify">
Although fish eye cameras have a key role in junction management systems, accompanying challenges come with them as well, such as : High distortion ratios, Different scales of same target object moving in different parts of the image, Day/night views variance (night view suffers from low quality related to surrounding lightning conditions), Exposure introduced with vehicle lights (night view). A dataset of traffic videos from several junctions at different times during the day/night is provided with the annotation for training and validation (icip2020.issd.com.tr). The evaluation will be performed based on separate test datasets.
</p>
<img src = "https://github.com/PrasunDatta/IEEE-VIP-Cup-2020-Vehicle-Detection-and-Tracking/blob/main/Fish-eye%20Camera%20Images.jpg" align = "Center" />

## Abstract


<p align = "justify">
Traffic surveillance and monitoring using fisheye
cameras are gaining popularity because of the 360-degree wideangle view. The accuracy of vehicle detection from fisheye images
has been significantly improved by using high-performance deep
learning-based object detectors. However, one key area of improvement remains in detecting vehicles from night-time images
due to the low brightness and contrast against the background.
We have found that it is possible to make night images suitable
for the model to learn from by intentionally blurring out
selective portions of the images before training. In our proposed
technique termed as SelectBlur, we first divide a night image
into square grids and depending on whether a grid meets certain
conditions, we decide on blurring it. It is shown that blurring
out parts of the image that are known to not contain any vehicles
leads to significantly improved performance.<b> The SelectBlur, in
conjunction with state-of-the-art object detectors, such as Yolov5x
and Yolov5s, beats the baseline model without pre-processing
by 5.3% and 3.0% improvement in mean average precision,
respectively. An ablation study of our proposed algorithm is also
performed by considering different conditions for blurring, and
also varying the type and the size of the kernel used to perform
the blurring operation.</b> </p>

## Model Diagram
<img src ="https://github.com/PrasunDatta/IEEE-VIP-Cup-2020-Vehicle-Detection-and-Tracking/blob/main/ProposedModel.PNG" align = "center" />
<img src ="https://github.com/PrasunDatta/IEEE-VIP-Cup-2020-Vehicle-Detection-and-Tracking/blob/main/Proposed%20Model_1.PNG" align ="center" />


