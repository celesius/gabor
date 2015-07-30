//
//  colorBalance.h
//  opencv_test
//
//  Created by vk on 15/5/11.
//  Copyright (c) 2015年 leisheng526. All rights reserved.
//

#ifndef __opencv_test__colorBalance__
#define __opencv_test__colorBalance__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "ColorBalance.h"
class ColorBalanceClass{
public:
    ColorBalanceClass();
    ~ColorBalanceClass();
    void getColorBalance(cv::Mat &destPR,double setValue);
    void getColorBalance2(cv::Mat &destPR,double setValue);
    void setSrcImage(const cv::Mat srcImage);
    void getImageWithSameConfig(cv::Mat srcImage, cv::Mat &destPR);
    void getRevertImage(cv::Mat &dst);  //取出没有计算的图像，并且回复设置
private:
    void color_balance_transfer_init ();
    void color_balance_init(ColorBalance *cb);
    void color_balance_create_lookup_tables(ColorBalance *cb);
    void setColorBalance (ColorBalance *cb , double setValue);
    void setColorBalance2 (ColorBalance *cb , double setValue);
   // void color_balance(ColorBalance *cb, cv::Mat &srcPR, cv::Mat &destPR);
    void mergeMat(const cv::Mat blueChannel,const cv::Mat greenChannel,const cv::Mat redChannel,cv::Mat &dst);
    double  highlights[256];
    double  midtones[256];
    double  shadows[256];
    bool transfer_initialized;
    ColorBalance m_colorBalance;
    cv::Mat m_image;
};

#endif /* defined(__opencv_test__colorBalance__) */
