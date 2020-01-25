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

#ifndef H264_MC_SPU_H
#define H264_MC_SPU_H

//#include "types_spu.h"

// motion compensation constants:
#define MB_TYPE_16x16      0x0008
#define MB_TYPE_16x8       0x0010
#define MB_TYPE_8x16       0x0020
#define MB_TYPE_8x8        0x0040
#define MB_TYPE_P0L0       0x1000
#define IS_16X16(a)        ((a)&MB_TYPE_16x16)
#define IS_16X8(a)         ((a)&MB_TYPE_16x8)
#define IS_8X16(a)         ((a)&MB_TYPE_8x16)
#define IS_8X8(a)          ((a)&MB_TYPE_8x8)
#define IS_SUB_8X8(a)      ((a)&MB_TYPE_16x16) //note reused
#define IS_SUB_8X4(a)      ((a)&MB_TYPE_16x8)  //note reused
#define IS_SUB_4X8(a)      ((a)&MB_TYPE_8x16)  //note reused
#define IS_SUB_4X4(a)      ((a)&MB_TYPE_8x8)   //note reused
#define IS_DIR(a, part, list) ((a) & (MB_TYPE_P0L0<<((part)+2*(list))))

#define FFMAX(a,b) ((a) > (b) ? (a) : (b))
#define FFMIN(a,b) ((a) > (b) ? (b) : (a))

//Motion compensation buffer strides
#define STRIDE_Y 48
#define STRIDE_C 32

typedef struct ref_data{
	uint8_t *data[3];
}ref_data;

typedef struct H264mc_part{
	int n;
	int chroma_height;
	int x_offset;
	int y_offset;
	int itp;
	int weight;
	int list0;
	int list1;
	int use_weight;
	ref_data ref[2];

}H264mc_part;

typedef struct H264mc{
	H264mc_part mc_part[16];
	int npart;
}H264mc;


#endif
