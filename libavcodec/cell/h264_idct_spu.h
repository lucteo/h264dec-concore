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


#ifndef H264_IDCT_SPU_H
#define H264_IDCT_SPU_H

void h264_idct4_add_spu(uint8_t *dst, short *block, int stride);
void h264_idct8_add_spu(uint8_t *dst, short *block, int stride);

/***********************************************************************
 * VEC_1D_IDCT
 ***********************************************************************
 * 1-dimensional 4x4 H264 integer DCT inverse transform.
 * Actually source and destination are 8x4. The low elements of the
 * source are discarded and the low elements of the destination mustn't
 * be used.
 * __vz0-__vz3 registers need to be declared in the caller function
 ***********************************************************************/
#define VEC_1D_DCT(vb0,vb1,vb2,vb3,va0,va1,va2,va3)				\
  /* 1st stage */								\
  __vz0 = spu_add(vb0,vb2);		/* temp[0] = Y[0] + Y[2] 	*/	\
  __vz1 = spu_sub(vb0,vb2);		/* temp[1] = Y[0] - Y[2] 	*/	\
  __vz2 = spu_rlmaska(vb1,-1);							\
  __vz2 = spu_sub(__vz2,vb3);		/* temp[2] = Y[1].1/2 - Y[3] 	*/	\
  __vz3 = spu_rlmaska(vb3,-1);							\
  __vz3 = spu_add(vb1,__vz3);		/* temp[3] = Y[1] + Y[3].1/2 	*/	\
										\
  /* 2nd stage: output */							\
  va0 = spu_add(__vz0,__vz3);		/* x[0] = temp[0] + temp[3] 	*/	\
  va1 = spu_add(__vz1,__vz2);		/* x[1] = temp[1] + temp[2] 	*/	\
  va2 = spu_sub(__vz1,__vz2);		/* x[2] = temp[1] - temp[2] 	*/  	\
  va3 = spu_sub(__vz0,__vz3)		/* x[3] = temp[0] - temp[3] 	*/

/***********************************************************************
 * VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8
 ***********************************************************************
 * load a vuint8_t vector from a unaligned memory position p
 * Converts the vector to vsint16_t
 * Adds the loaded and converted vector to a defined vector va
 * converts back the result to vuint8_t and store it to memory
 **********************************************************************/

#define VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(p,shift,va,align_dst)	\
    vdst_orig = *(vuint8_t *) (p);					\
    vdst = spu_or(spu_slqwbyte(vdst_orig, shift),(vuint8_t) vzero);	\
    vdst_ss = (vsint16_t) spu_shuffle((vuint8_t)vzero,vdst,mergehu8);	\
    va = spu_add(va,vdst_ss);						\
    sat = spu_cmpgt(va,(vsint16_t)vzero);				\
    va = spu_and(va,(vsint16_t)sat);					\
    sat = spu_cmpgt(va,vmax);						\
    va = spu_sel(va,vmax,sat);						\
    va_u8 = (vuint8_t) spu_shuffle(va,(vsint16_t) vzero,packu16);	\
    vfdst = spu_shuffle(vdst_orig, va_u8, align_dst);			\
    *(vuint8_t *) (dst) = vfdst

/***********************************************************************
 * VEC_TRANSPOSE_8
 ***********************************************************************
 * Transposes a 8x8 matrix of s16 vectors
 **********************************************************************/
#define VEC_TRANSPOSE_8(a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7) \
    b0 = spu_shuffle( a0, a4, m1 ); \
    b1 = spu_shuffle( a1, a5, m1 ); \
    b2 = spu_shuffle( a2, a6, m1 ); \
    b3 = spu_shuffle( a3, a7, m1 ); \
    b4 = spu_shuffle( a4, a0, m2 ); \
    b5 = spu_shuffle( a5, a1, m2 ); \
    b6 = spu_shuffle( a6, a2, m2 ); \
    b7 = spu_shuffle( a7, a3, m2 ); \
    a0 = spu_shuffle( b0, b2, m3 ); \
    a1 = spu_shuffle( b1, b3, m3 ); \
    a2 = spu_shuffle( b2, b0, m4 ); \
    a3 = spu_shuffle( b3, b1, m4 ); \
    a4 = spu_shuffle( b4, b6, m3 ); \
    a5 = spu_shuffle( b5, b7, m3 ); \
    a6 = spu_shuffle( b6, b4, m4 ); \
    a7 = spu_shuffle( b7, b5, m4 ); \
    b0 = spu_shuffle( a0, a1, m5 ); \
    b1 = spu_shuffle( a1, a0, m6 ); \
    b2 = spu_shuffle( a2, a3, m5 ); \
    b3 = spu_shuffle( a3, a2, m6 ); \
    b4 = spu_shuffle( a4, a5, m5 ); \
    b5 = spu_shuffle( a5, a4, m6 ); \
    b6 = spu_shuffle( a6, a7, m5 ); \
    b7 = spu_shuffle( a7, a6, m6 )

/***********************************************************************
 * VEC_1D_IDCT8
 ***********************************************************************
 * 1-dimensional 8x8 H264 integer DCT inverse transform.
 ***********************************************************************/
#define VEC_1D_DCT8(vb0,vb1,vb2,vb3,vb4,vb5,vb6,vb7)						\
  vza0 = spu_add(vb0,vb4);		/* a[0] = Y[0] + Y[4] 	*/				\
  vza2 = spu_sub(vb0,vb4);		/* a[2] = Y[0] - Y[4]	*/				\
  vza4 = spu_rlmaska(vb2,-1);									\
  vza4 = spu_sub(vza4,vb6);		/* a[4] = Y[2]>>1 - Y[6]	*/			\
  vza6 = spu_rlmaska(vb6,-1	);								\
  vza6 = spu_add(vb2,vza6);		/* a[6] = Y[2]    + Y[6]>>1	*/			\
  												\
  vzb0 = spu_add(vza0,vza6);		/* b[0] = a[0] + a[6]	*/				\
  vzb2 = spu_add(vza2,vza4);		/* b[2] = a[2] + a[4]	*/				\
  vzb4 = spu_sub(vza2,vza4);		/* b[4] = a[2] - a[4]	*/				\
  vzb6 = spu_sub(vza0,vza6);		/* b[6] = a[0] - a[6]	*/				\
  												\
  vza1 = spu_rlmaska(vb7,-1);									\
  vzal = spu_add(vza1,vb7);									\
  vzah = spu_sub(vb5,vb3);									\
  vza1 = spu_sub(vzah,vzal);	/* a1 = (-Y[3] + Y[5]) - (Y[7] + (Y[7]>>1))	*/		\
  												\
  vza3 = spu_rlmaska(vb3,-1);									\
  vzal = spu_add(vza3,vb3);									\
  vzah = spu_add(vb1,vb7);									\
  vza3 = spu_sub(vzah,vzal);  	/* a3 =  (Y[1] + Y[7]) - (Y[3] + (Y[3]>>1))	*/		\
  												\
  vza5 = spu_rlmaska(vb5,-1);									\
  vzal = spu_add(vza5,vb5);									\
  vzah = spu_sub(vb7,vb1);									\
  vza5 = spu_add(vzah,vzal);	/* a5 = (-Y[1] + Y[7]) + (Y[5] + Y[5]>>1))	*/		\
												\
  vza7 = spu_rlmaska(vb1,-1);									\
  vzal = spu_add(vza7,vb1);									\
  vzah = spu_add(vb3,vb5);									\
  vza7 = spu_add(vzah,vzal);	/* a7 =  (Y[3] + Y[5]) + (Y[1] + (Y[1]>>1))	*/		\
  												\
  vzb1 = spu_rlmaska(vza7,-2);									\
  vzb1 = spu_add(vzb1,vza1);		/* b1 = (a7>>2) + a1	*/				\
  vzb3 = spu_rlmaska(vza5,-2);									\
  vzb3 = spu_add(vzb3,vza3);		/* b3 =  a3 + (a5>>2)	*/				\
  vzb5 = spu_rlmaska(vza3,-2);									\
  vzb5 = spu_sub(vzb5,vza5);  		/* b5 = (a3>>2) - a5	*/				\
  vzb7 = spu_rlmaska(vza1,-2);									\
  vzb7 = spu_sub(vza7,vzb7);		/* b7 =  a7 - (a1>>2)	*/				\
  												\
  vb0 = spu_add(vzb0,vzb7); 		/* src[i][0] = b0 + b7	*/				\
  vb7 = spu_sub(vzb0,vzb7);		/* src[i][7] = b0 - b7	*/				\
  vb1 = spu_add(vzb2,vzb5);		/* src[i][1] = b2 + b5	*/				\
  vb6 = spu_sub(vzb2,vzb5);		/* src[i][6] = b2 - b5	*/				\
  vb2 = spu_add(vzb4,vzb3);		/* src[i][2] = b4 + b3	*/				\
  vb5 = spu_sub(vzb4,vzb3);		/* src[i][5] = b4 - b3	*/				\
  vb3 = spu_add(vzb6,vzb1);		/* src[i][3] = b6 + b1	*/				\
  vb4 = spu_sub(vzb6,vzb1);		/* src[i][4] = b6 - b1	*/


#endif /*H264_IDCT_SPU_H*/
