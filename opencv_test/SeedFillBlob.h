//
//  SeedFillBlob.h
//  opencv_test
//
//  Created by vk on 15/7/22.
//  Copyright (c) 2015年 leisheng526. All rights reserved.
//

#ifndef __opencv_test__SeedFillBlob__
#define __opencv_test__SeedFillBlob__

#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
void icvprCcaBySeedFill(const cv::Mat & _binImg, cv::Mat& _lableImg,int &totalLabel);

//cv::Mat icvprCcaBySeedFill(const cv::Mat _binImg,int &totalLabel);
#endif /* defined(__opencv_test__SeedFillBlob__) */
