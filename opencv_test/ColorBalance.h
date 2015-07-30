//
//  ColorBalance.h
//  opencv_test
//
//  Created by vk on 15/5/11.
//  Copyright (c) 2015年 leisheng526. All rights reserved.
//

#ifndef opencv_test_ColorBalance_h
#define opencv_test_ColorBalance_h

struct _ColorBalance
{
    bool preserve_luminosity;
    double cyan_red[3];
    double magenta_green[3];
    double yellow_blue[3];
    
    uchar r_lookup[256];
    uchar g_lookup[256];
    uchar b_lookup[256];
};

typedef _ColorBalance ColorBalance;

#endif
