//
//  colorBalance.cpp
//  opencv_test
//
//  Created by vk on 15/5/11.
//  Copyright (c) 2015年 leisheng526. All rights reserved.
//

#include "ColorBalanceClass.h"
#include "baseenum.h"
//#include "opencv2/highgui/highgui.hpp"
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

#define CLAMP(x,l,u) ((x)<(l)?(l):((x)>(u)?(u):(x)))
#define CLAMP0255(a) CLAMP(a,0,255)

ColorBalanceClass::ColorBalanceClass()
{
    transfer_initialized = false;
    this->color_balance_transfer_init();
    this->color_balance_init(&m_colorBalance);
}

void ColorBalanceClass::color_balance_transfer_init (void)
{
    int i;
    
    for (i = 0; i < 256; i++)
    {
        static const double a = 64, b = 85, scale = 1.785;
        double low = CLAMP ((i - b) / -a + .5, 0, 1) * scale;
        double mid = CLAMP ((i - b) /  a + .5, 0, 1) *
        CLAMP ((i + b - 255) / -a + .5, 0, 1) * scale;
        
        shadows[i]          = low;
        midtones[i]         = mid;
        highlights[255 - i] = low;
    }
}

void ColorBalanceClass::color_balance_init (ColorBalance *cb)
{
    GimpTransferMode range;
    // g_return_if_fail (cb != NULL);
    cb->cyan_red[0]      = 0.0;
    cb->magenta_green[0] = 0.0;
    cb->yellow_blue[0]   = 0.0;
    
    cb->cyan_red[1]      = 0.0;
    cb->magenta_green[1] = 0.0;
    cb->yellow_blue[1]   = 0.0;
    
    cb->cyan_red[2]      = 0.0;
    cb->magenta_green[2] = 0.0;
    cb->yellow_blue[2]   = 0.0;
    
    cb->preserve_luminosity = true;
}

void ColorBalanceClass::setColorBalance (ColorBalance *cb , double setValue)
{
    cb->preserve_luminosity = false;
    
    cb->cyan_red[0]      = 0;
    cb->magenta_green[0] = 0;
    cb->yellow_blue[0]   = 0;
    
    cb->cyan_red[GIMP_MIDTONES]      = setValue*(1);
    cb->magenta_green[GIMP_MIDTONES] = setValue*(1);
    cb->yellow_blue[GIMP_MIDTONES]   = setValue;
    
    cb->cyan_red[GIMP_HIGHLIGHTS]      = setValue*(1);
    cb->magenta_green[GIMP_HIGHLIGHTS] = setValue*(1);
    cb->yellow_blue[GIMP_HIGHLIGHTS]   = setValue;
}

void ColorBalanceClass::setColorBalance2 (ColorBalance *cb , double setValue)
{
    cb->preserve_luminosity = false;
    
    cb->cyan_red[0]      = 0;
    cb->magenta_green[0] = 0;
    cb->yellow_blue[0]   = 0;
    
    cb->cyan_red[GIMP_MIDTONES]      = setValue*(0.9);
    cb->magenta_green[GIMP_MIDTONES] = setValue*(0.7);
    cb->yellow_blue[GIMP_MIDTONES]   = setValue*1.5;
    
    cb->cyan_red[GIMP_HIGHLIGHTS]      = setValue*(0.9);
    cb->magenta_green[GIMP_HIGHLIGHTS] = setValue*(0.7);
    cb->yellow_blue[GIMP_HIGHLIGHTS]   = setValue*1.5;
}

void ColorBalanceClass::color_balance_create_lookup_tables (ColorBalance *cb)
{
    int     i;
    int   r_n, g_n, b_n;
    
    //g_return_if_fail (cb != NULL);
    
    if (! transfer_initialized)
    {
        color_balance_transfer_init ();
        transfer_initialized = true;
    }
    
    for (i = 0; i < 256; i++)
    {
        r_n = i;
        g_n = i;
        b_n = i;
        
        r_n += cb->cyan_red[GIMP_SHADOWS] * shadows[i];
        r_n += cb->cyan_red[GIMP_MIDTONES] * midtones[i];
        r_n += cb->cyan_red[GIMP_HIGHLIGHTS] * highlights[i];
        r_n = CLAMP0255 (r_n);
        
        g_n += cb->magenta_green[GIMP_SHADOWS] * shadows[i];
        g_n += cb->magenta_green[GIMP_MIDTONES] * midtones[i];
        g_n += cb->magenta_green[GIMP_HIGHLIGHTS] * highlights[i];
        g_n = CLAMP0255 (g_n);
        
        b_n += cb->yellow_blue[GIMP_SHADOWS] * shadows[i];
        b_n += cb->yellow_blue[GIMP_MIDTONES] * midtones[i];
        b_n += cb->yellow_blue[GIMP_HIGHLIGHTS] * highlights[i];
        b_n = CLAMP0255 (b_n);
        
        cb->r_lookup[i] = r_n;
        cb->g_lookup[i] = g_n;
        cb->b_lookup[i] = b_n;
    }
}

void ColorBalanceClass::mergeMat(const cv::Mat blueChannel,const cv::Mat greenChannel,const cv::Mat redChannel,cv::Mat &dst)
{
    std::vector<cv::Mat> array_to_merge;
    array_to_merge.push_back(blueChannel);
    array_to_merge.push_back(greenChannel);
    array_to_merge.push_back(redChannel);
    cv::merge(array_to_merge, dst);
}

void ColorBalanceClass::setSrcImage(const cv::Mat srcImage)
{
    srcImage.copyTo(m_image);  //数据拷贝 srcImage可以释放
}

//void ColorBalanceClass::getColorBalance(cv::Mat &destPR,double setValue)
//{
//    //    const unsigned char *src, *s;
//    //    unsigned char       *dest, *d;
//    //    bool      alpha;
//    unsigned char r, g, b;
//    unsigned char r_n, g_n, b_n;
//    int w, h;
//    h  = m_image.rows;
//    w  = m_image.cols;
//    
//    ColorBalance *cb = &m_colorBalance;
//    this->setColorBalance(cb, setValue);
//    this->color_balance_create_lookup_tables(cb);
//    
//    cv::Mat red = cv::Mat(m_image.rows,m_image.cols,CV_8UC1,cv::Scalar(0));
//    cv::Mat blue = cv::Mat(m_image.rows,m_image.cols,CV_8UC1,cv::Scalar(0));
//    cv::Mat green = cv::Mat(m_image.rows,m_image.cols,CV_8UC1,cv::Scalar(0));
//    
//    cv::Vec3b intensity;
//    
//    for(int y = 0;y<h;y++)
//        for(int x = 0;x<w;x++)
//        {
//            intensity = m_image.at<cv::Vec3b>(y, x);
//            b = intensity.val[0];
//            g = intensity.val[1];
//            r = intensity.val[2];
//            
//            r_n = cb->r_lookup[r];
//            g_n = cb->g_lookup[g];
//            b_n = cb->b_lookup[b];
//            
//            
//            blue.at<uchar>(y,x) = b_n;
//            green.at<uchar>(y,x) = g_n;
//            red.at<uchar>(y,x) = r_n;
//        }
//    
//    
//    mergeMat(blue,green,red,destPR);
//}

void ColorBalanceClass::getColorBalance(cv::Mat &destPR,double setValue)
{
    //    const unsigned char *src, *s;
    //    unsigned char       *dest, *d;
    //    bool      alpha;
    unsigned char r, g, b;
    unsigned char r_n, g_n, b_n;
    int w, h;
    h  = m_image.rows;
    w  = m_image.cols;
    
    ColorBalance *cb = &m_colorBalance;
    this->setColorBalance(cb, setValue);
    this->color_balance_create_lookup_tables(cb);
    
    cv::Mat red = cv::Mat(m_image.rows,m_image.cols,CV_8UC1,cv::Scalar(0));
    cv::Mat blue = cv::Mat(m_image.rows,m_image.cols,CV_8UC1,cv::Scalar(0));
    cv::Mat green = cv::Mat(m_image.rows,m_image.cols,CV_8UC1,cv::Scalar(0));
    
    cv::Vec3b intensity;
    
    for (int y = 0; y<h; y++) {
        uchar *data = m_image.ptr<uchar>(y);
        uchar *blueData = blue.ptr<uchar>(y);
        uchar *greenData = green.ptr<uchar>(y);
        uchar *redData = red.ptr<uchar>(y);
        for (int x = 0; x<w; x++) {
            b = *data;
            data++;
            g = *data;
            data++;
            r = *data;
            data++;
            
            r_n = cb->r_lookup[r];
            g_n = cb->g_lookup[g];
            b_n = cb->b_lookup[b];
            *blueData = b_n;
            blueData++;
            *greenData = g_n;
            greenData++;
            *redData = r_n;
            redData++;
        }
    }
    mergeMat(blue,green,red,destPR);
}

void ColorBalanceClass::getColorBalance2(cv::Mat &destPR,double setValue)
{
    //    const unsigned char *src, *s;
    //    unsigned char       *dest, *d;
    //    bool      alpha;
    unsigned char r, g, b;
    unsigned char r_n, g_n, b_n;
    int w, h;
    h  = m_image.rows;
    w  = m_image.cols;
    
    ColorBalance *cb = &m_colorBalance;
    this->setColorBalance2(cb, setValue);
    this->color_balance_create_lookup_tables(cb);
    
    cv::Mat red = cv::Mat(m_image.rows,m_image.cols,CV_8UC1,cv::Scalar(0));
    cv::Mat blue = cv::Mat(m_image.rows,m_image.cols,CV_8UC1,cv::Scalar(0));
    cv::Mat green = cv::Mat(m_image.rows,m_image.cols,CV_8UC1,cv::Scalar(0));
    
    cv::Vec3b intensity;
    
    for (int y = 0; y<h; y++) {
        uchar *data = m_image.ptr<uchar>(y);
        uchar *blueData = blue.ptr<uchar>(y);
        uchar *greenData = green.ptr<uchar>(y);
        uchar *redData = red.ptr<uchar>(y);
        for (int x = 0; x<w; x++) {
            b = *data;
            data++;
            g = *data;
            data++;
            r = *data;
            data++;
            
            r_n = cb->r_lookup[r];
            g_n = cb->g_lookup[g];
            b_n = cb->b_lookup[b];
            *blueData = b_n;
            blueData++;
            *greenData = g_n;
            greenData++;
            *redData = r_n;
            redData++;
        }
    }
    mergeMat(blue,green,red,destPR);
}


void ColorBalanceClass::getImageWithSameConfig(cv::Mat srcImage, cv::Mat &destPR)
{
    //    const unsigned char *src, *s;
    //    unsigned char       *dest, *d;
    //    bool      alpha;
    unsigned char r, g, b;
    unsigned char r_n, g_n, b_n;
    int w, h;
    h  = srcImage.rows;
    w  = srcImage.cols;
    
    ColorBalance *cb = &m_colorBalance;
    //this->setColorBalance2(cb, setValue);
    //this->color_balance_create_lookup_tables(cb);
    
    cv::Mat red = cv::Mat(srcImage.rows,srcImage.cols,CV_8UC1,cv::Scalar(0));
    cv::Mat blue = cv::Mat(srcImage.rows,srcImage.cols,CV_8UC1,cv::Scalar(0));
    cv::Mat green = cv::Mat(srcImage.rows,srcImage.cols,CV_8UC1,cv::Scalar(0));
    
    cv::Vec3b intensity;
    
    for (int y = 0; y<h; y++) {
        uchar *data = srcImage.ptr<uchar>(y);
        uchar *blueData = blue.ptr<uchar>(y);
        uchar *greenData = green.ptr<uchar>(y);
        uchar *redData = red.ptr<uchar>(y);
        for (int x = 0; x<w; x++) {
            b = *data;
            data++;
            g = *data;
            data++;
            r = *data;
            data++;
            
            r_n = cb->r_lookup[r];
            g_n = cb->g_lookup[g];
            b_n = cb->b_lookup[b];
            *blueData = b_n;
            blueData++;
            *greenData = g_n;
            greenData++;
            *redData = r_n;
            redData++;
        }
    }
    mergeMat(blue,green,red,destPR);
}




void ColorBalanceClass::getRevertImage(cv::Mat &dst)
{
    transfer_initialized = false;
    this->color_balance_transfer_init();
    this->color_balance_init(&m_colorBalance);
    m_image.copyTo(dst);
}

