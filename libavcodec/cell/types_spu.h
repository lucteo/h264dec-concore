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

#ifndef TYPES_SPU_H
#define TYPES_SPU_H

/***********************************************************************
 * Scalar types
 **********************************************************************/
    typedef signed char  int8_t;
    typedef signed short int16_t;
    typedef signed int   int32_t;
    typedef unsigned char  uint8_t;
    typedef unsigned short uint16_t;
    typedef unsigned int   uint32_t;
    typedef unsigned long long uint64_t;

//     typedef short DCTELEM;		// transform coeficients of dct

/***********************************************************************
 * Vector types
 **********************************************************************/
    typedef	vector	signed int	vsint32_t;
    typedef	vector	unsigned int	vuint32_t;
    typedef	vector	signed short	vsint16_t;
    typedef	vector	unsigned short	vuint16_t;
    typedef	vector	signed char	vsint8_t;
    typedef	vector	unsigned char	vuint8_t;

/***********************************************************************
 * Functions
 **********************************************************************/
    typedef void (*qpel_mc_func)(uint8_t *dst, uint8_t *src, int dst_stride, int h);
    typedef void (*h264_chroma_mc_func)(uint8_t *dst, uint8_t *src, int dst_stride, int h, int x, int y);
    typedef void (*h264_idct_func)(uint8_t *dst, short *block, int stride);
    typedef void (*h264_weight_func)(uint8_t *block, int stride, int log2_denom, int weight, int offset);
    typedef void (*h264_biweight_func)(uint8_t *dst, uint8_t *src, int dst_stride, int src_stride, int log2_denom, int weightd,
                  int weights, int offset);
    typedef void(* intra_pred4x4)(uint8_t *src, uint8_t *topright, int stride);
    typedef void(* intra_pred16x16)(uint8_t *src, int stride);
    typedef void(* intra_pred8x8)(uint8_t *src, int stride);
    typedef void(* intra_pred8x8l)(uint8_t *src, int topleft, int topright, int stride);


#define AVV(x...) {x}


#endif // AVCODEC_TYPES_SPU_H




