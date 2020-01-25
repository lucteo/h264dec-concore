/*
 * Copyright 2013 TU Berlin
 *
 * This file is part of Starbench.
 *
 * This Starbench is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Starbench is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 * @author Chi Ching Chi <chi.c.chi@tu-berlin.de>
 */

#ifndef AVCODEC_H264PRED_H
#define AVCODEC_H264PRED_H

//#include "libavutil/common.h"
//#include "dsputil.h"

/**
 * Prediction types
 */
//@{
#define VERT_PRED             0
#define HOR_PRED              1
#define DC_PRED               2
#define DIAG_DOWN_LEFT_PRED   3
#define DIAG_DOWN_RIGHT_PRED  4
#define VERT_RIGHT_PRED       5
#define HOR_DOWN_PRED         6
#define VERT_LEFT_PRED        7
#define HOR_UP_PRED           8

#define LEFT_DC_PRED          9
#define TOP_DC_PRED           10
#define DC_128_PRED           11

#define DIAG_DOWN_LEFT_PRED_RV40_NODOWN   12
#define HOR_UP_PRED_RV40_NODOWN           13
#define VERT_LEFT_PRED_RV40_NODOWN        14

#define DC_PRED8x8            0
#define HOR_PRED8x8           1
#define VERT_PRED8x8          2
#define PLANE_PRED8x8         3

#define LEFT_DC_PRED8x8       4
#define TOP_DC_PRED8x8        5
#define DC_128_PRED8x8        6

#define ALZHEIMER_DC_L0T_PRED8x8 7
#define ALZHEIMER_DC_0LT_PRED8x8 8
#define ALZHEIMER_DC_L00_PRED8x8 9
#define ALZHEIMER_DC_0L0_PRED8x8 10
//@}

/**
 * Context for storing H.264 prediction functions
 */
typedef struct H264PredContext{
    void (*pred4x4  [9+3+3])(uint8_t *src, uint8_t *topright, int stride);//FIXME move to dsp?
    void (*pred8x8l [9+3])(uint8_t *src, int topleft, int topright, int stride);
    void (*pred8x8  [4+3+4])(uint8_t *src, int stride);
    void (*pred16x16[4+3])(uint8_t *src, int stride);

    void (*pred4x4_add  [2])(uint8_t *pix/*align  4*/, const DCTELEM *block/*align 16*/, int stride);
    void (*pred8x8l_add [2])(uint8_t *pix/*align  8*/, const DCTELEM *block/*align 16*/, int stride);
    void (*pred8x8_add  [3])(uint8_t *pix/*align  8*/, const int *block_offset, const DCTELEM *block/*align 16*/, int stride);
    void (*pred16x16_add[3])(uint8_t *pix/*align 16*/, const int *block_offset, const DCTELEM *block/*align 16*/, int stride);
}H264PredContext;

void ff_h264_pred_init(H264PredContext *h);
void ff_h264_pred_init_arm(H264PredContext *h);


#endif /* AVCODEC_H264PRED_H */
