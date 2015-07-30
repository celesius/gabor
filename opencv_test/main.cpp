//////
//////  main.cpp
//////  opencv_test
//////
//////  Created by leisheng526 on 14/10/31.
//////  Copyright (c) 2014年 leisheng526. All rights reserved.
//////
////
#include <iostream>
#include <opencv2/opencv.hpp>

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "ColorBalanceClass.h"
#include "LazySnapping.h"
#include "twoPassBlob.h"
#include "cloverLabelColor.h"
#include "SeedFillBlob.h"

using namespace std;
//using namespace cv;

#define CLAMP(x,l,u) ((x)<(l)?(l):((x)>(u)?(u):(x)))
#define CLAMP0255(a) CLAMP(a,0,255)

static bool transfer_initialized = false;
cv::Mat outframe;
cv::Mat filterFrame;
cv::Mat lsMat;
//cv::Mat lsResult;



//jiangbo
//void SimplestCB(Mat& in, Mat& out, float percent) {
//    assert(in.channels() == 3);
//    assert(percent > 0 && percent < 100);
//    
//    float half_percent = percent / 200.0f;
//    
//    vector<Mat> tmpsplit; split(in,tmpsplit);
//    for(int i=0;i<3;i++) {
//        //find the low and high precentile values (based on the input percentile)
//        Mat flat; tmpsplit[i].reshape(1,1).copyTo(flat);
//        cv::sort(flat,flat,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
//        int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
//        int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));
//        cout << lowval << " " << highval << endl;
//        
//        //saturate below the low percentile and above the high percentile
//        tmpsplit[i].setTo(lowval,tmpsplit[i] < lowval);
//        tmpsplit[i].setTo(highval,tmpsplit[i] > highval);
//        
//        //scale the channel
//        normalize(tmpsplit[i],tmpsplit[i],0,255,NORM_MINMAX);
//    }
//    merge(tmpsplit,out);
//}

//void opencvColorChannel(cv::Mat imgSrc,cv::Mat &blue,cv::Mat &green, cv::Mat &red,cv::Mat &dst)
//{
//    int rows = imgSrc.rows;
//    int cols = imgSrc.cols;
//    Vec3b intensity;
//    cv::Scalar outPixel;
//    
//    uchar bluep;
//    uchar greenp;
//    uchar redp;
//    
//    for(int y = 0;y<rows;y++)
//        for(int x = 0;x<cols;x++)
//        {
//            intensity = imgSrc.at<Vec3b>(y, x);
//            bluep = intensity.val[0];
//            greenp = intensity.val[1];
//            redp = intensity.val[2];
//            
//            blue.at<uchar>(y,x) = bluep;
//            green.at<uchar>(y,x) = greenp;
//            red.at<uchar>(y,x) = redp;
//        }
//    mergeMat(blue,green,red,dst);
//}

void myImageShow(const string &winname,cv::Mat showImage,cv::Size showSize)
{
    cv::Mat showMat;
    cv::resize(showImage, showMat , showSize);
    cv::imshow(winname, showMat);
}

void myImageShowScale(const string &winname,cv::Mat showImage)
{
    cv::Mat showMat;
    
    cv::Size showSize = cv::Size(showImage.cols/1,showImage.rows/1);
    
    cv::resize(showImage, showMat , showSize);
    cv::imshow(winname, showMat);
}


void filter_and_threshold(cv::Mat imFrame,cv::Mat & outFrame)
{
    
    /* Soften image */
    cv::Mat tmpMat;
    cv::GaussianBlur(imFrame, tmpMat, cv::Size(11,11), 0,0);
    //myImageShow("Gaussian", tmpMat, cv::Size(640,480));
    //cvSmooth(ctx->image, ctx->temp_image3, CV_GAUSSIAN, 11, 11, 0, 0);
    /* Remove some impulsive noise */
    cv::medianBlur(tmpMat, tmpMat,1);
//    cvSmooth(ctx->temp_image3, ctx->temp_image3, CV_MEDIAN, 11, 11, 0, 0);
    cv::cvtColor(tmpMat, tmpMat, CV_BGR2HSV);
//    cvCvtColor(ctx->temp_image3, ctx->temp_image3, CV_BGR2HSV);
//    
//    /*
//     * Apply threshold on HSV values to detect skin color
//     */
    
    cv::inRange(tmpMat, cv::Scalar(0,55,90,255), cv::Scalar(28,175,230,255), outFrame);
    
//    cvInRangeS(ctx->temp_image3,
//               cvScalar(0, 55, 90, 255),
//               cvScalar(28, 175, 230, 255),
//               ctx->thr_image);
//    
//    /* Apply morphological opening */
//    cvMorphologyEx(ctx->thr_image, ctx->thr_image, NULL, ctx->kernel,
//                   CV_MOP_OPEN, 1);
//    cvSmooth(ctx->thr_image, ctx->thr_image, CV_GAUSSIAN, 3, 3, 0, 0);
  
  //  IplConvKernel *kernel = cvCreateStructuringElementEx(9, 9, 4, 4, CV_SHAPE_RECT,NULL);
    cv::Mat kernelMat = getStructuringElement(CV_SHAPE_RECT, cv::Size(9,9),cv::Point(4,4));
    
    
    cv::morphologyEx(outFrame, outFrame, CV_MOP_OPEN,kernelMat);
    cv::GaussianBlur(outFrame, outFrame, cv::Size(3,3), 0,0);
    
}

void ergodicCricleMat(cv::Mat refImg,cv::Mat drawImg)
{
    
    
    int nr=refImg.rows;
    int nc=refImg.cols;
    //outImage.create(image.size(),image.type());
//    if(refImg.isContinuous()&&refImg.isContinuous())
//    {
//        nr=1;
//        nc=nc*refImg.rows*refImg.channels();
//    }
    for(int i=0;i<nr;i++)
    {
        const uchar* inData=refImg.ptr<uchar>(i);
        //uchar* outData=outImage.ptr<uchar>(i);
        for(int j=0;j<nc;j++)
        {
            if((int)inData[j] > 250)
            {
                cv::circle(drawImg, cv::Point(i,j), 5, cv::Scalar(0),2, 8, 0);
            }
        }
    }
    
    imshow( "dst", drawImg );
    
    
}


void imageProcess(cv::Mat srcImage)
{
    cv::Mat harrisMat;
    cv::Mat dst_norm, dst_norm_scaled;
    cv::Mat srcImage_gray;
    cv::cvtColor(srcImage, srcImage_gray, CV_BGRA2GRAY);
    
    cv::cornerHarris(srcImage_gray, harrisMat, 2, 3, 0.04);
    
    normalize( harrisMat, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    convertScaleAbs( dst_norm,dst_norm_scaled);
//    ergodicCricleMat(dst_norm,dst_norm_scaled);
    /// Drawing a circle around corners
//    for( int j = 0; j < dst_norm.rows ; j++ )
//    { for( int i = 0; i < dst_norm.cols; i++ )
//    {
//        if( (int) dst_norm.at<float>(j,i) > 250 )
//        {
//            circle( dst_norm_scaled, cv::Point( i, j ), 5,  cv::Scalar(0), 2, 8, 0 );
//            circle(srcImage,cv::Point( i, j ), 5,  cv::Scalar(255,0,0), -1, 8, 0 );
//        }
//    }
//    }
//    printf("dst_norm_scaled.channels() = %d\n",dst_norm_scaled.channels());
//    printf("dst_norm.channels() = %d\n",dst_norm.channels());
//    imshow( "dst", dst_norm_scaled );
    
    
    int nr=dst_norm.rows;
    int nc=dst_norm.cols;
    
    for(int y = 0;y<nr;y++){
        uchar *oneRowDst_normValid = dst_norm_scaled.ptr<uchar>(y);
        for(int x =0;x<nc;x++){
            if(oneRowDst_normValid[x] > 200)
            {
                cv::circle(dst_norm_scaled, cv::Point(x,y), 5, cv::Scalar(0));
            
            }
            
        }
    }
    
//    for(int i=0;i<nr;i++)
//    {
//        const uchar* inData=dst_norm.ptr<uchar>(i);
//        //uchar* outData=outImage.ptr<uchar>(i);
//        for(int j=0;j<nc;j++)
//        {
//            int value = inData[j];
//            if(value > 250)
//            {
//                //cv::circle(dst_norm_scaled, cv::Point(i,j), 5, cv::Scalar(0),2, 8, 0);
//            }
//        }
//    }
    
    
    cv::imshow( "dst_norm", dst_norm );
    cv::imshow("dst_norm_scaled",dst_norm_scaled);

}

void testImageProcess(cv::Mat srcImg)
{
    cv::Mat grayImg;
    cv::cvtColor(srcImg, grayImg, CV_BGRA2GRAY);
    //cv::Mat outImg =
    //myImageShow("grayImg", grayImg, cv::Size(640,480));
    cv::Mat testMat;
    testMat.create(grayImg.size(), grayImg.type());
    
    int nr = grayImg.rows;
    int nc = grayImg.cols;
    
    for(int y = 0;y<nr;y++){
        uchar *oneRowData = grayImg.ptr<uchar>(y);
        uchar *oneRowDataOut = testMat.ptr<uchar>(y);
        for(int x = 0;x<nc;x++)
        {
            oneRowDataOut[x] = oneRowData[x];
            if(oneRowData[x] > 200)
            {
                cv::circle(testMat, cv::Point(x,y), 10, cv::Scalar(0));
            
            }
        }
    }
    
    cv::imshow("src", grayImg);
    cv::imshow("out", testMat);
    
    //myImageShow("src", grayImg, cv::Size(640,480));
    //myImageShow("out", testMat, cv::Size(640,480));

}

void imageProcessDiffFrame(cv::Mat previousFrame , cv::Mat currentFrame , cv::Mat &resultFrame)
{
    cv::Mat previousGray;
    cv::Mat currentGray;
    
    int channels = previousFrame.channels();
    
    if(channels == 3){
        cv::cvtColor(previousFrame, previousGray, CV_RGB2GRAY);
        cv::cvtColor(currentFrame, currentGray, CV_RGB2GRAY);
    }
    else{
        cv::cvtColor(previousFrame, previousGray, CV_BGRA2GRAY);
        cv::cvtColor(currentFrame, currentGray, CV_BGRA2GRAY);
    }
    //cv::imshow("pGray",previousGray);
    //cv::imshow("cGray",currentGray);
    
    cv::absdiff(currentGray, previousGray, resultFrame);
}

//void on_mouse( int event, int x, int y, int flags, void* ustc)
//{
//    //CvFont font;
//    //cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);
//    
//    if( event == CV_EVENT_LBUTTONDOWN )
//    {
//        //CvPoint pt = cvPoint(x,y);
//        cv::Point pt = cv::Point(x,y);
//        uchar pixel = lsResult.at<uchar>(y, x);
//        //int dataVaild = outframe.at<int>(y,x);
//        
//        char temp[16];
//        //sprintf(temp,"(%d,%d)",pt.x,pt.y);
//        sprintf(temp, "(%d)",pixel);
//        cv::putText(lsResult, temp, pt, cv::FONT_HERSHEY_PLAIN, 0.75, cv::Scalar(255,0,0,0));
//        std::cout<<lsResult.channels()<<std::endl;
//        
//        //cvPutText(src,temp, pt, &font, cvScalar(255, 255, 255, 0));
//        //cvCircle( src, pt, 2,cvScalar(255,0,0,0) ,CV_FILLED, CV_AA, 0 );
//        cv::circle(lsResult, pt, 2, cv::Scalar(255));
//        cv::imshow( "lsResult", lsResult );
//    }
//}

void on_mouseColor( int event, int x, int y, int flags, void* ustc)
{
    //CvFont font;
    //cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);
    
    if( event == CV_EVENT_LBUTTONDOWN )
    {
        //CvPoint pt = cvPoint(x,y);
        cv::Point pt = cv::Point(x,y);
        cv::Vec3b pixel = filterFrame.at<cv::Vec3b>(y, x);
        //uchar pixel = outframe.at<uchar>(y, x);
        //int dataVaild = outframe.at<int>(y,x);
        
        char temp[16];
        //sprintf(temp,"(%d,%d)",pt.x,pt.y);
        sprintf(temp, "(%d,%d,%d)",pixel[0],pixel[1],pixel[2]);
        cv::putText(filterFrame, temp, pt, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
       // std::cout<<outframe.channels()<<std::endl;
        
        //cvPutText(src,temp, pt, &font, cvScalar(255, 255, 255, 0));
        //cvCircle( src, pt, 2,cvScalar(255,0,0,0) ,CV_FILLED, CV_AA, 0 );
        cv::circle(filterFrame, pt, 2, cv::Scalar(255));
        cv::imshow( "filterFrame", filterFrame );
    }
}


vector<CvPoint> forePts;
vector<CvPoint> backPts;
const int SCALE = 4;
//IplImage* image = NULL;
char winName[] = "lazySnapping";
IplImage* imageDraw = NULL;

cv::Mat imageProcessingLS( cv::Mat srcImage ){
    //cv::imshow("srcImage", srcImage);
    IplImage simage = srcImage;
    IplImage *image = &simage;
    
    //imageDraw = cvCloneImage(image);
    
    
    LasySnapping ls;

    IplImage* imageLS = cvCreateImage(cvSize(image->width/SCALE,image->height/SCALE),
                                      8,3);
    cvResize(image,imageLS);
    ls.setImage(imageLS);
    ls.setBackgroundPoints(backPts);
    ls.setForegroundPoints(forePts);
    double t = (double)cvGetTickCount();
    ls.runMaxflow();
    t = (double)cvGetTickCount() - t;
    printf( "run time = %gms\n", t/(cvGetTickFrequency()*1000) );
    IplImage* mask = ls.getImageMask();
    IplImage* gray = cvCreateImage(cvGetSize(image),8,1);
    cvResize(mask,gray);
    //cvShowImage("maskLS", mask);
    //cvShowImage("grayLS", gray);
    lsMat = cv::Mat(gray,true);
    // edge
//    cvCanny(gray,gray,50,150,3);
//    
//    IplImage* showImg = cvCloneImage(imageDraw);
//    for(int h =0; h < image->height; h ++){
//        unsigned char* pgray = (unsigned char*)gray->imageData + gray->widthStep*h;
//        unsigned char* pimage = (unsigned char*)showImg->imageData + showImg->widthStep*h;
//        for(int width  =0; width < image->width; width++){
//            if(*pgray++ != 0 ){
//                pimage[0] = 0;
//                pimage[1] = 255;
//                pimage[2] = 0;
//            }
//            pimage+=3;
//        }
//    }
   // cvSaveImage("t.bmp",showImg);
   // cvShowImage(winName,showImg);
    
    
    //cvReleaseImage(&image);
    cvReleaseImage(&imageLS);
    cvReleaseImage(&mask);
   // cvReleaseImage(&image);
   // cvReleaseImage(&showImg);
    cvReleaseImage(&gray);
    return lsMat;

}

void diffFrameAndFilterImgAndFrontPoint(const cv::Mat srcImage1,const cv::Mat srcImage2,cv::Mat &dtsImage, cv::Mat filterImg ,std::vector<CvPoint> &frontPoint,uchar thresh)
{
    cv::Mat gray1,gray2;
    cv::cvtColor(srcImage1, gray1, CV_BGR2GRAY);
    cv::cvtColor(srcImage2, gray2, CV_BGR2GRAY);
    int nr = gray1.rows;
    int nc = gray1.cols;
   // dtsImage = cv::Mat(nr,nc,CV_8UC1,cv::Scalar(0));
   // filterImg = cv::Mat(nr,nc,CV_8UC1,cv::Scalar(0));
    
    frontPoint.clear();
    
    for(int y = 0;y<nr;y++){
        uchar *gray1Data = gray1.ptr<uchar>(y);
        uchar *gray2Data = gray2.ptr<uchar>(y);
        uchar *dtsData = dtsImage.ptr<uchar>(y);
        uchar *filterData = filterImg.ptr<uchar>(y);
        for (int x = 0; x<nc; x++) {
            if(gray1Data[x]>gray2Data[x])
            {
                uchar data = gray1Data[x]-gray2Data[x];
                dtsData[x] = data;
                if(data > thresh){
                    filterData[x] = 255;
                    frontPoint.push_back(cvPoint(x/4, y/4));
                }
                else{
                    filterData[x] = 0;
                }
                    
            }
            else{
                dtsData[x] = 0;
            }
        }
    }
}

void filterBlob(cv::Mat blobImg,cv::Mat maskImg, cv::Mat &dstImg)
{
    int nr = blobImg.rows;
    int nc = blobImg.cols;
    std::map<int,int> blobMap;
    std::vector<int> blobValid;
    dstImg = cv::Mat(nr,nc,CV_8UC1,cv::Scalar(0));
    
    for(int y = 0;y<nr;y++){
        uchar *blobRowData = blobImg.ptr<uchar>(y);
        uchar *maskRowData = maskImg.ptr<uchar>(y);
        for (int x = 0; x<nc; x++) {
            uchar maskPixelData = maskRowData[x];
            uchar blobPixelData = blobRowData[x];
            if(maskPixelData != 0 && blobPixelData!= 0){
                blobMap[(int)blobPixelData]++;
                //std::cout<<"maskPixelData  = "<<(int)maskPixelData<<std::endl;
                //std::cout<<"blobPixelData  = "<<(int)blobPixelData<<std::endl;
            }
            
        }
    }
    
    std::map<int,int>::iterator it;
    for(it=blobMap.begin();it!=blobMap.end();++it)
    {
        std::cout<<it->first<<":"<<it->second<<std::endl;
        blobValid.push_back(it->first);
    }
    
    for(int y=0;y<nr;y++){
        uchar *blobRowData2 = blobImg.ptr<uchar>(y);
        uchar *dstImgRowData = dstImg.ptr<uchar>(y);
        for(int x = 0;x<nc;x++){
            uchar blobPixelData2 = blobRowData2[x];
            //uchar dstPixelData = dstImgRowData[x];
            bool pixelValid = false;
            for(int i = 0;i<blobValid.size();i++)
            {
                if(blobPixelData2 == blobValid[i])
                    pixelValid = true;
            }
            
            if(pixelValid)
                dstImgRowData[x] = blobPixelData2;
            else
                dstImgRowData[x] = 0;
        }
    
    }
}

float sigCalculate(float b , float lm)
{
    float sig;
    float pi = CV_PI;
    //printf("pi = %f\n",pi);
    sig = (1/pi) * sqrt(log(2)/2) * ((pow(2, b) + 1)/(pow(2, b) - 1)) * lm ;
    return sig;
}

double lmCalculate(double para)
{
    double lm = 0;
    double f = cv::sqrt(2.0);
    lm = cv::pow(f, para);
    return lm;
}

void getGaborImgWithKernel(cv::Mat kernel,cv::Mat srcMat,cv::Mat & dstMat)  //根据外部传进的kernel进行
{
    cv::Mat floatMat;
    cv::Mat src_f;
    srcMat.convertTo(src_f, CV_32F);
    cv::filter2D(src_f, floatMat, CV_32F, kernel);
    floatMat.convertTo(dstMat,CV_8U,0.1/25.0);
}

void getGaborImg(cv::Mat srcMat,cv::Mat & dstMat) //根据局部的kernel进行计算
{
    cv::Mat floatMat;
    cv::Mat src_f;
    int kernel_size = 101;
    //double sig = 1, th = 0, lm = 1.0, gm = 0.02, ps = 0;
    // double sig = 5, th = 0, lm = 0.0, gm = 0.5, ps = 1;
    double  th = 0, lm = 40.0 /*lmCalculate(5.0)*/, gm = 0.5, ps = 0, sig =sigCalculate(1,lm); //方向 波长 长宽比 相位偏移  sig = 0.56*lm 带宽与sig 和 lm有关 当满足 sig = 0.56*lm时候，带宽为1
    srcMat.convertTo(src_f, CV_32F);
    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);//
    cv::filter2D(src_f, floatMat, CV_32F, kernel);
    floatMat.convertTo(dstMat,CV_8U,1.0/255.0);
}

//void thre

int main(int ac, const char * av[]) {
    cv::Mat colorImg = cv::imread("/Users/vk/Downloads/textureTest/1.pic.jpg");          // load grayscale
    cv::Mat dest;
    cv::Mat src_f;
    cv::Mat in;
    cv::Mat kernel;
    //cv::threshold(<#InputArray src#>, <#OutputArray dst#>, <#double thresh#>, <#double maxval#>, <#int type#>)
    //cv::resize(colorImg, in, cv::Size(1280,720));
    in = colorImg;
    cv::VideoCapture cap(0);
    cv::Mat frame;
    //in.convertTo(src_f,CV_32F);
    double pi = CV_PI;
    int kernel_size = 69;
    //double sig = 1, th = 0, lm = 1.0, gm = 0.02, ps = 0;
   // double sig = 5, th = 0, lm = 0.0, gm = 0.5, ps = 1;
    double  th = 0, lm = 5 /*lmCalculate(5.0)*/, gm = 0.5, ps = 0, sig =sigCalculate(1,lm); //方向 波长 长宽比 相位偏移  sig = 0.56*lm 带宽与sig 和 lm有关 当满足 sig = 0.56*lm时候，带宽为1
    cv::Mat colorDect;
    while (cap.read(frame)) {
        cv::resize(frame, in , cv::Size(640,480));
       // in = frame;
        filter_and_threshold(in , colorDect );
        myImageShowScale("org",in);
        myImageShowScale("colorDect",colorDect);
        
//    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);//
//    cv::imshow("k1",kernel);
//    //cv::filter2D(src_f, dest, CV_32F, kernel);
//    //cerr << dest(cv::Rect(30,30,10,10)) << endl; // peek into the data
//    //cv::Mat viz;
//    //dest.convertTo(viz,CV_8U,1.0/255.0);     // move to proper[0..255] range to show it
//    getGaborImgWithKernel(kernel,in,dest);
//    myImageShowScale("d1",dest);
//    
//    //kernel_size = 69;
//    //lm = 40.0;
//    th = pi/4;
//    kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);//
//    cv::imshow("k2",kernel);
//    getGaborImgWithKernel(kernel,in,dest);
//    cv::threshold(dest, dest, 1, 255, CV_8UC1);
//    myImageShowScale("d2",dest);
//    for (int i = 0; i < 4; i++) {
//        th = pi*i/4;
//        
//        char name[100];
//        
//        sprintf(name, "CD%d" ,i);
//        
//        kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
//
//        getGaborImgWithKernel(kernel,in,dest);
//        cv::threshold(dest, dest, 1, 255, CV_8UC3);
//        myImageShowScale(name,dest);
//        
//        
//        //printf("i = %d",i);
//    }
    cv::cvtColor(in, in, CV_BGR2GRAY);
    cv::Mat integratMat;
    for (int i = 0; i < 4; i++) {
        th = pi*i/4;
        
        char name[100];
        char kname[100];
        sprintf(name, "GD%d" ,i);
        sprintf(kname, "kernel%d" ,i);
        
        
        kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
        
        getGaborImgWithKernel(kernel,in,dest);
        //cv::threshold(dest, dest, 10, 255, CV_8UC3);
        cv::Mat disp;
        //cv::adaptiveThreshold(dest, disp, 255, CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,8,8);
        cv::threshold(dest, disp, 1, 255, CV_8UC1);
        
        if(i == 0){
            integratMat = disp;
        }
        else{
            int nx = integratMat.cols;
            int ny = integratMat.rows;
            
            for(int y = 0; y<ny; y++){
                
                uchar *rowData = integratMat.ptr<uchar>(y);
                uchar *rowDataDisp = disp.ptr<uchar>(y);
                for(int x = 0; x<nx; x++){
                    if(rowDataDisp[x] != 0){
                        rowData[x] = 255;
                    }
                }
            }
        }
        cv::imshow(kname, kernel);
        //myImageShowScale(name,disp);
        //printf("i = %d",i);
    }
    //cv::imshow("outInt", integratMat);
    myImageShowScale("outInt", integratMat);
    //myImageShowScale("org",in);
   // printf("pow ==== %f\n ", pow(2, 3));
   // cv::imshow("d",viz);
   // cv::imshow("org", in);
    int key =   cv::waitKey(1);
        if(key == 'q')
            break;
    }

}

//main jiabgbo
//int main(int ac, const char * av[]) {
//    cv::Mat frame;
//    cv::VideoCapture cap(1);
//    
//    
//    cv::Mat srcImg = cv::imread("/Users/vk/Downloads/IMG_3334.jpg");
//    cv::Mat pFrame;// = cv::imread("/Users/vk/Pictures/SkinColorImg/3/filename178.jpg");
//    cv::Mat cFrame;// = cv::imread("/Users/vk/Pictures/SkinColorImg/3/filename179.jpg");
//    //filterFrame = cv::Mat(480,640,CV_8UC3,cv::Scalar(0,0,0));
//    cv::Mat cleanImage;
//  //  cv::Mat outFrame;
//   // cv::Mat lastFrame;
//   // cv::Mat diffFrame;
//    bool saveBool = false;
//    char filename[200];
//    int n = 0;
//    
//    int nr = cFrame.rows;
//    int nc = cFrame.cols;
////    cv::Mat labelImg;
//    
//    outframe = cv::Mat(480,640,CV_8UC1,cv::Scalar(0));
//    filterFrame = cv::Mat(480,640,CV_8UC1,cv::Scalar(0));
//    
//    while (cap.read(frame)) {
//        cv::resize(frame, frame, cv::Size(640,480));
//        cFrame = frame.clone();
//        //  算法过程
//        cv::imshow("orgFrame", frame);
//        
//        if(pFrame.rows != 0)
//        {
//        //cv::resize(frame, frame, cv::Size(640,480));
//        
//        //filter_and_threshold(frame,outframe);
//    //imageProcessDiffFrame(pFrame,cFrame,outframe);
//    
//    //cv::Mat diffImg;
//    diffFrameAndFilterImgAndFrontPoint(cFrame, pFrame, outframe, filterFrame ,forePts, 120);
//    //cv::imshow("myDiff", diffImg);
//    
//    cv::Vec3b pixelValid;
////    
////    for(int i = 0;i<forePts.size();i++){
////        filterFrame.at<cv::Vec3b>(forePts[i].y*4,forePts[i].x*4) = cFrame.at<cv::Vec3b>(forePts[i].y*4, forePts[i].x*4);
////    }
//    
////    for(int y = 0;y<nr;y++){
////        for(int x=0;x<nc;x++){
////            pixelValid = cFrame.at<cv::Vec3b>(y, x);
////            if( outframe.at<uchar>(y, x) > 100 )
////            {
////                filterFrame.at<cv::Vec3b>(y, x) = pixelValid;
////                forePts.push_back(cvPoint(x/SCALE, y/SCALE));
////            }
////        }
////    }
//       // double t = (double)cvGetTickCount();
//        cv::Mat lsResult = imageProcessingLS(cFrame);
//       // t = (double)cvGetTickCount() - t;
//       // printf( "run time = %gms\n", t/(cvGetTickFrequency()*1000) );
//        
//    
//    cv::Mat binImage;
//
//    cv::threshold(lsResult, binImage, 50, 1, CV_THRESH_OTSU) ;
//     //cv::imshow("binImage", binImage) ;
//    //icvprCcaByTwoPass(binImage, labelImg);
//#if 1
//    int labelNum;
//    //if(labelImg.rows != 0)
//        
//        
//        cv::Mat labelImg;
//        
//        icvprCcaBySeedFill(binImage, labelImg ,labelNum);  //导致release错误
//
////    std::cout<<"labelNum = " <<labelNum<<std::endl;
////    int labelImgChannel = labelImg.channels();
////    std::cout<<labelImgChannel<<std::endl;
//    
//  //  cv::imshow("labelImg", labelImg) ;
//        cv::Mat grayImg;
//   // labelImg *= 10 ;
//    labelImg.convertTo(grayImg, CV_8UC1);
//    cv::Mat blobFilterResultMat;
//    filterBlob(grayImg,filterFrame,blobFilterResultMat);
//    
//    
//    cv::imshow("labelImg", blobFilterResultMat) ;
//    blobFilterResultMat.convertTo(blobFilterResultMat, CV_32SC1);
//    
//    cv::Mat colorLabelImg ;
//    //icvprLabelColor(labelImg, colorLabelImg) ;
//    icvprLabelColor(blobFilterResultMat, colorLabelImg) ;
//    cv::imshow("colorImg", colorLabelImg) ;
//    
//    
////    cv::imshow("outImage", outframe);
//    cv::imshow("filterFrame", filterFrame);
////    cv::imshow("cFrame", cFrame);
//    cv::imshow("lsResult", lsResult);
//
//#endif
//    //   cv::setMouseCallback("lsResult", on_mouse);
//    
//      //  cv::imshow("pFrame", pFrame);
//      //  cv::imshow("cFrame", cFrame);
//        
//    
//        
//        //testImageProcess(frame);
//        //imageProcess(frame);
//        
//        //frame = srcImg;
////        filter_and_threshold(frame,outframe);
////        myImageShow("Gaussian", outframe, cv::Size(640,480));
//        
//        
////        if(lastFrame.rows != 0)
////        {
////            cv::imshow("lastFrame", lastFrame);
////            cv::absdiff(outframe, lastFrame, diffFrame);
////            cv::imshow("diffFrame", diffFrame);
////        }
//       // filter_and_threshold_withc(frame,outframeORG);
//       // myImageShow("cameraImg", frame, cv::Size(640,480));
//        
////        lastFrame = outframe.clone();
////        if(saveBool)
////        {
////            sprintf(filename,"filename%.3d.jpg",n++ );
////            cv::imwrite(filename, frame);
////            std::cout<<"Save"<<n<<std::endl;
////        }
//    //int key = cv::waitKey(0);
//    
//    //cvWaitKey(0);
//    //while (1) {
//        //printf( "run time = %gs\n", t/(cvGetTickFrequency()*1000000) );
//        }
//        pFrame = cFrame.clone();
//        int key = cv::waitKey(2);
//        if(key == 'q' )
//        {
//            break;
//        }
//        else if(key == 's')
//        {
//            saveBool = !saveBool;
//        }
////        else if (key == 'r')
////        {
////            filterFrame = cv::Mat(480,640,CV_8UC3,cv::Scalar(0,0,0));
////            for(int y = 0;y<nr;y++){
////                for(int x=0;x<nc;x++){
////                    pixelValid = cFrame.at<cv::Vec3b>(y, x);
////                    if( outframe.at<uchar>(y, x) > 100 )
////                    {
////                        filterFrame.at<cv::Vec3b>(y, x) = pixelValid;
////                    }
////                }
////            }
////            cv::imshow("filterFrame", filterFrame);
////        }
//    }
//    
//    
////    while (1) {
////        int key = cv::waitKey(2);
////        if(key == 'q' )
////        {
////            break;
////        }
////    }
//
//    return 0;
//}


/*
 * Simple hand detection algorithm based on OpenCV
 *
 * (C) Copyright 2012-2013 <b.galvani@gmail.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 */
//
//#include <stdio.h>
//
//#include <opencv2/imgproc/imgproc_c.h>
//#include <opencv2/highgui/highgui_c.h>
//
//#define VIDEO_FILE	"video.avi"
//#define VIDEO_FORMAT	CV_FOURCC('M', 'J', 'P', 'G')
//#define NUM_FINGERS	5
//#define NUM_DEFECTS	8
//
//#define RED     CV_RGB(255, 0, 0)
//#define GREEN   CV_RGB(0, 255, 0)
//#define BLUE    CV_RGB(0, 0, 255)
//#define YELLOW  CV_RGB(255, 255, 0)
//#define PURPLE  CV_RGB(255, 0, 255)
//#define GREY    CV_RGB(200, 200, 200)
//
//struct ctx {
//    CvCapture	*capture;	/* Capture handle */
//    CvVideoWriter	*writer;	/* File recording handle */
//    
//    IplImage	*image;		/* Input image */
//    IplImage	*thr_image;	/* After filtering and thresholding */
//    IplImage	*temp_image1;	/* Temporary image (1 channel) */
//    IplImage	*temp_image3;	/* Temporary image (3 channels) */
//    
//    CvSeq		*contour;	/* Hand contour */
//    CvSeq		*hull;		/* Hand convex hull */
//    
//    CvPoint		hand_center;
//    CvPoint		*fingers;	/* Detected fingers positions */
//    CvPoint		*defects;	/* Convexity defects depth points */
//    
//    CvMemStorage	*hull_st;
//    CvMemStorage	*contour_st;
//    CvMemStorage	*temp_st;
//    CvMemStorage	*defects_st;
//    
//    IplConvKernel	*kernel;	/* Kernel for morph operations */
//    
//    int		num_fingers;
//    int		hand_radius;
//    int		num_defects;
//};
//
//void init_capture(struct ctx *ctx)
//{
//    ctx->capture = cvCaptureFromCAM(0);
//    if (!ctx->capture) {
//        fprintf(stderr, "Error initializing capture\n");
//        exit(1);
//    }
//    ctx->image = cvQueryFrame(ctx->capture);
//}
//
//void init_recording(struct ctx *ctx)
//{
//    int fps, width, height;
//    
//    fps = cvGetCaptureProperty(ctx->capture, CV_CAP_PROP_FPS);
//    width = cvGetCaptureProperty(ctx->capture, CV_CAP_PROP_FRAME_WIDTH);
//    height = cvGetCaptureProperty(ctx->capture, CV_CAP_PROP_FRAME_HEIGHT);
//    
//    if (fps < 0)
//        fps = 10;
//    
//    ctx->writer = cvCreateVideoWriter(VIDEO_FILE, VIDEO_FORMAT, fps,
//                                      cvSize(width, height), 1);
//    
//    if (!ctx->writer) {
//        fprintf(stderr, "Error initializing video writer\n");
//        exit(1);
//    }
//}
//
//void init_windows(void)
//{
//    cvNamedWindow("output", CV_WINDOW_AUTOSIZE);
//    cvNamedWindow("thresholded", CV_WINDOW_AUTOSIZE);
//    cvMoveWindow("output", 50, 50);
//    cvMoveWindow("thresholded", 700, 50);
//}
//
//void init_ctx(struct ctx *ctx)
//{
//    ctx->thr_image = cvCreateImage(cvGetSize(ctx->image), 8, 1);
//    ctx->temp_image1 = cvCreateImage(cvGetSize(ctx->image), 8, 1);
//    ctx->temp_image3 = cvCreateImage(cvGetSize(ctx->image), 8, 3);
//    ctx->kernel = cvCreateStructuringElementEx(9, 9, 4, 4, CV_SHAPE_RECT,
//                                               NULL);
//    ctx->contour_st = cvCreateMemStorage(0);
//    ctx->hull_st = cvCreateMemStorage(0);
//    ctx->temp_st = cvCreateMemStorage(0);
//    ctx->fingers = (CvPoint*)calloc(NUM_FINGERS + 1, sizeof(CvPoint));
//    ctx->defects = (CvPoint*)calloc(NUM_DEFECTS, sizeof(CvPoint));
//    
//    
//}
//
//void filter_and_threshold(struct ctx *ctx)
//{
//    
//    /* Soften image */
//    cvSmooth(ctx->image, ctx->temp_image3, CV_GAUSSIAN, 11, 11, 0, 0);
//    /* Remove some impulsive noise */
//    cvSmooth(ctx->temp_image3, ctx->temp_image3, CV_MEDIAN, 11, 11, 0, 0);
//    
//    cvCvtColor(ctx->temp_image3, ctx->temp_image3, CV_BGR2HSV);
//    
//    /*
//     * Apply threshold on HSV values to detect skin color
//     */
//    cvInRangeS(ctx->temp_image3,
//               cvScalar(0, 55, 90, 255),
//               cvScalar(28, 175, 230, 255),
//               ctx->thr_image);
//    
//    /* Apply morphological opening */
//    cvMorphologyEx(ctx->thr_image, ctx->thr_image, NULL, ctx->kernel,
//                   CV_MOP_OPEN, 1);
//    cvSmooth(ctx->thr_image, ctx->thr_image, CV_GAUSSIAN, 3, 3, 0, 0);
//    cvShowImage("thr_image", ctx->thr_image);
//}
//
//void find_contour(struct ctx *ctx)
//{
//    double area, max_area = 0.0;
//    CvSeq *contours, *tmp, *contour = NULL;
//    
//    /* cvFindContours modifies input image, so make a copy */
//    cvCopy(ctx->thr_image, ctx->temp_image1, NULL);
//    cvFindContours(ctx->temp_image1, ctx->temp_st, &contours,
//                   sizeof(CvContour), CV_RETR_EXTERNAL,
//                   CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
//    
//    /* Select contour having greatest area */
//    for (tmp = contours; tmp; tmp = tmp->h_next) {
//        area = fabs(cvContourArea(tmp, CV_WHOLE_SEQ, 0));
//        if (area > max_area) {
//            max_area = area;
//            contour = tmp;
//        }
//    }
//    
//    /* Approximate contour with poly-line */
//    if (contour) {
//        contour = cvApproxPoly(contour, sizeof(CvContour),
//                               ctx->contour_st, CV_POLY_APPROX_DP, 2,
//                               1);
//        ctx->contour = contour;
//    }
//}
//
//void find_convex_hull(struct ctx *ctx)
//{
//    CvSeq *defects;
//    CvConvexityDefect *defect_array;
//    int i;
//    int x = 0, y = 0;
//    int dist = 0;
//    
//    ctx->hull = NULL;
//    
//    if (!ctx->contour)
//        return;
//    
//    ctx->hull = cvConvexHull2(ctx->contour, ctx->hull_st, CV_CLOCKWISE, 0);
//    
//    if (ctx->hull) {
//        
//        /* Get convexity defects of contour w.r.t. the convex hull */
//        defects = cvConvexityDefects(ctx->contour, ctx->hull,
//                                     ctx->defects_st);
//        
//        if (defects && defects->total) {
//            defect_array = (CvConvexityDefect*)calloc(defects->total,
//                                  sizeof(CvConvexityDefect));
//            cvCvtSeqToArray(defects, defect_array, CV_WHOLE_SEQ);
//            
//            /* Average depth points to get hand center */
//            for (i = 0; i < defects->total && i < NUM_DEFECTS; i++) {
//                x += defect_array[i].depth_point->x;
//                y += defect_array[i].depth_point->y;
//                
//                ctx->defects[i] = cvPoint(defect_array[i].depth_point->x,
//                                          defect_array[i].depth_point->y);
//            }
//            
//            x /= defects->total;
//            y /= defects->total;
//            
//            ctx->num_defects = defects->total;
//            ctx->hand_center = cvPoint(x, y);
//            
//            /* Compute hand radius as mean of distances of
//             defects' depth point to hand center */
//            for (i = 0; i < defects->total; i++) {
//                int d = (x - defect_array[i].depth_point->x) *
//                (x - defect_array[i].depth_point->x) +
//                (y - defect_array[i].depth_point->y) *
//                (y - defect_array[i].depth_point->y);
//                
//                dist += sqrt(d);
//            }
//            
//            ctx->hand_radius = dist / defects->total;
//            free(defect_array);
//        }
//    }
//}
//
//void find_fingers(struct ctx *ctx)
//{
//    int n;
//    int i;
//    CvPoint *points;
//    CvPoint max_point;
//    int dist1 = 0, dist2 = 0;
//    
//    ctx->num_fingers = 0;
//    
//    if (!ctx->contour || !ctx->hull)
//        return;
//    
//    n = ctx->contour->total;
//    points = (CvPoint*)calloc(n, sizeof(CvPoint));
//    
//    cvCvtSeqToArray(ctx->contour, points, CV_WHOLE_SEQ);
//    
//    /*
//     * Fingers are detected as points where the distance to the center
//     * is a local maximum
//     */
//    for (i = 0; i < n; i++) {
//        int dist;
//        int cx = ctx->hand_center.x;
//        int cy = ctx->hand_center.y;
//        
//        dist = (cx - points[i].x) * (cx - points[i].x) +
//        (cy - points[i].y) * (cy - points[i].y);
//        
//        if (dist < dist1 && dist1 > dist2 && max_point.x != 0
//            && max_point.y < cvGetSize(ctx->image).height - 10) {
//            
//            ctx->fingers[ctx->num_fingers++] = max_point;
//            if (ctx->num_fingers >= NUM_FINGERS + 1)
//                break;
//        }
//        
//        dist2 = dist1;
//        dist1 = dist;
//        max_point = points[i];
//    }
//    
//    free(points);
//}
//
//void display(struct ctx *ctx)
//{
//    int i;
//    
//    if (ctx->num_fingers == NUM_FINGERS) {
//        
//#if defined(SHOW_HAND_CONTOUR)
//        cvDrawContours(ctx->image, ctx->contour, BLUE, GREEN, 0, 1,
//                       CV_AA, cvPoint(0, 0));
//#endif
//        cvCircle(ctx->image, ctx->hand_center, 5, PURPLE, 1, CV_AA, 0);
//        cvCircle(ctx->image, ctx->hand_center, ctx->hand_radius,
//                 RED, 1, CV_AA, 0);
//        
//        for (i = 0; i < ctx->num_fingers; i++) {
//            
//            cvCircle(ctx->image, ctx->fingers[i], 10,
//                     GREEN, 3, CV_AA, 0);
//            
//            cvLine(ctx->image, ctx->hand_center, ctx->fingers[i],
//                   YELLOW, 1, CV_AA, 0);
//        }
//        
//        for (i = 0; i < ctx->num_defects; i++) {
//            cvCircle(ctx->image, ctx->defects[i], 2,
//                     GREY, 2, CV_AA, 0);
//        }
//    }
//    
//    cvShowImage("output", ctx->image);
//    cvShowImage("thresholded", ctx->thr_image);
//}
//
//void filter_and_threshold(cv::Mat imFrame,cv::Mat & outFrame)
//{
//    
//    /* Soften image */
//    cv::Mat tmpMat;
//    cv::GaussianBlur(imFrame, tmpMat, cv::Size(11,11), 0,0);
//    /* Remove some impulsive noise */
//    cv::medianBlur(tmpMat, tmpMat,11);
//    //    cvSmooth(ctx->temp_image3, ctx->temp_image3, CV_MEDIAN, 11, 11, 0, 0);
//    cv::cvtColor(tmpMat, tmpMat, CV_BGR2HSV);
//    //
//    //    /*
//    //     * Apply threshold on HSV values to detect skin color
//    //     */
//    
//    cv::inRange(tmpMat, cv::Scalar(0,55,90,255), cv::Scalar(28,175,230,255), outFrame);
//    
//    //    cvInRangeS(ctx->temp_image3,
//    //               cvScalar(0, 55, 90, 255),
//    //               cvScalar(28, 175, 230, 255),
//    //               ctx->thr_image);
//    //
//    //    /* Apply morphological opening */
//    //    cvMorphologyEx(ctx->thr_image, ctx->thr_image, NULL, ctx->kernel,
//    //                   CV_MOP_OPEN, 1);
//    //    cvSmooth(ctx->thr_image, ctx->thr_image, CV_GAUSSIAN, 3, 3, 0, 0);
//    
//    //  IplConvKernel *kernel = cvCreateStructuringElementEx(9, 9, 4, 4, CV_SHAPE_RECT,NULL);
//    cv::Mat kernelMat = getStructuringElement(CV_SHAPE_RECT, cv::Size(9,9),cv::Point(4,4));
//    
//    
//    cv::morphologyEx(outFrame, outFrame, CV_MOP_OPEN,kernelMat);
//    cv::GaussianBlur(outFrame, outFrame, cv::Size(3,3), 0,0);
//  //   cvSmooth(ctx->thr_image, ctx->thr_image, CV_GAUSSIAN, 3, 3, 0, 0);
//    
//    cv::imshow("Gaussian", outFrame);
////    myImageShow("Gaussian", outFrame, cv::Size(640,480));
//    
//}
//
//
//
//void jiangboInitImg(struct ctx *ctx)
//{
//    cv::Mat srcImg = cv::imread("/Users/vk/Pictures/hand.JPG");
//    cv::Mat mat2Ipl;
//    cv::Mat outImg;
//    cv::resize(srcImg, mat2Ipl , cv::Size(640,480));
//    
//    filter_and_threshold(mat2Ipl,outImg);
//    
//    IplImage myImage = IplImage(mat2Ipl);
//    ctx->image = cvCloneImage(&myImage);  //深拷贝
//}
//
//
//
//int main(int argc, char **argv)
//{
//    struct ctx ctx = { };
//    int key;
//    
//    init_capture(&ctx);
//    //---
//    jiangboInitImg(&ctx);
//    //---
//    init_recording(&ctx);
//    init_windows();
//    init_ctx(&ctx);
//    
//    do {
//       // ctx.image = cvQueryFrame(ctx.capture);
////    cv::Mat srcImg = cv::imread("/Users/vk/Pictures/hand.JPG");
////    cv::Mat mat2Ipl;
////    
////    cv::resize(srcImg, mat2Ipl , cv::Size(640,480));
////    
////    IplImage myImage = IplImage(mat2Ipl); //cvLoadImage("/Users/vk/Pictures/hand.JPG");
////    IplImage *outImg;
////   // cvResize(myImage, outImg,cvSize(640, 480));
////    
////  
////    
////    ctx.image = &myImage;
//        filter_and_threshold(&ctx);
//        find_contour(&ctx);
//        find_convex_hull(&ctx);
//        find_fingers(&ctx);
//        
//        display(&ctx);
//       // cvWriteFrame(ctx.writer, ctx.image);
//    
//        key = cvWaitKey(1);
//    } while (key != 'q');
//    
//    return 0;
//}



//#include <cv.h>
//#include <highgui.h>
//#include <stdio.h>

//IplImage* src=0;
//void on_mouse( int event, int x, int y, int flags, void* ustc)
//{
//    CvFont font;
//    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);
//    
//    if( event == CV_EVENT_LBUTTONDOWN )
//    {
//        CvPoint pt = cvPoint(x,y);
//        
//        
//        
//        char temp[16];
//        sprintf(temp,"(%d,%d)",pt.x,pt.y);
//        cvPutText(src,temp, pt, &font, cvScalar(255, 255, 255, 0));
//        cvCircle( src, pt, 2,cvScalar(255,0,0,0) ,CV_FILLED, CV_AA, 0 );
//        cvShowImage( "src", src );
//    }
//}
//
//int main()
//{
//    src=cvLoadImage("/Users/vk/Pictures/SkinColorImg/3/filename178.jpg",1);
//    
//    cvNamedWindow("src",1);
//    cvSetMouseCallback( "src", on_mouse, 0 );
//    
//    cvShowImage("src",src);
//    cvWaitKey(0);
//    cvDestroyAllWindows();
//    cvReleaseImage(&src);
//    
//    return 0;
//}
