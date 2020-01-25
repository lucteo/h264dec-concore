/*
 * Copyright (c) 2009 TUDelft 
 * 
 * Cell Parallel SPU - Macroblock Decoding.
 */

/**
 * @file libavcodec/cell/spu/h264_main_spu.c
 * Cell Parallel SPU - Macroblock Decoding
 * @author C C Chi <c.c.chi@student.tudelft.nl>
 * 
 * SIMD kernels 
 * H.264/AVC motion compensation
 * @author Mauricio Alvarez <alvarez@ac.upc.edu>
 * @author Albert Paradis <apar7632@hotmail.com>
 */ 

#include <spu_intrinsics.h>
#include "types_spu.h"
#include "h264_tables.h"
#include "h264_idct_spu.h"
#include "h264_intra_spu.h"

/***********************************************************************
 * ff_h264_idct_add_spu
 ***********************************************************************
 *  h264 idct 4x4 transform with SPU SIMD intrinsics
 *  using the factorized algorithm 
 *  Mauricio Alvarez: alvarez@ac.upc.edu
 *  - DCTELEM* block: transformed coefficients are stored consecutvely in memory, 
 *  - for the 4x4 transform the structure is like that:
 *       || coef_00 | coef_01 || coef_02 | coef_03 ||..||coef_0F||
 *  - Usually the DCTELEM block is declared with an alignment modificator in such a way 
 *    that the  array is 128 bit (16 byte, 8 short) aligned.
 *  - The dst pointer can be unaligned with unaligment as a multiple of 4.
 ***********************************************************************/

// idct_dc
void ff_idct_dc_add(uint8_t *dst, short *block, int stride){
    int i, j;
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;
    int dc = (block[0] + 32) >> 6;
    for( j = 0; j < 4; j++ ){
        for( i = 0; i < 4; i++ )
            dst[i] = cm[ dst[i] + dc ];
        dst += stride;
    }
}

void ff_idct8_dc_add(uint8_t *dst, short *block, int stride){
    int i, j;
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;
    int dc = (block[0] + 32) >> 6;
    for( j = 0; j < 8; j++ ){
        for( i = 0; i < 8; i++ )
            dst[i] = cm[ dst[i] + dc ];
        dst += stride;
    }
}

// add without idct

void add_pixels8_c(uint8_t *pixels, short *block, int line_size)
{
    int i;
    for(i=0;i<8;i++) {
        pixels[0] += block[0];
        pixels[1] += block[1];
        pixels[2] += block[2];
        pixels[3] += block[3];
        pixels[4] += block[4];
        pixels[5] += block[5];
        pixels[6] += block[6];
        pixels[7] += block[7];
        pixels += line_size;
        block += 8;
    }
}

void add_pixels4_c(uint8_t *pixels, short *block, int line_size)
{
    int i;
    for(i=0;i<4;i++) {
        pixels[0] += block[0];
        pixels[1] += block[1];
        pixels[2] += block[2];
        pixels[3] += block[3];
        pixels += line_size;
        block += 4;
    }
}

void h264_luma_dc_dequant_idct_c(short *block, int qmul){
	#define stride 16
	int i;
	int temp[16]; //FIXME check if this is a good idea
	static const int x_offset[4]={0, 1*stride, 4* stride,  5*stride};
	static const int y_offset[4]={0, 2*stride, 8* stride, 10*stride};

	for(i=0; i<4; i++){
		const int offset= y_offset[i];
		const int z0= block[offset+stride*0] + block[offset+stride*4];
		const int z1= block[offset+stride*0] - block[offset+stride*4];
		const int z2= block[offset+stride*1] - block[offset+stride*5];
		const int z3= block[offset+stride*1] + block[offset+stride*5];

		temp[4*i+0]= z0+z3;
		temp[4*i+1]= z1+z2;
		temp[4*i+2]= z1-z2;
		temp[4*i+3]= z0-z3;
	}

	for(i=0; i<4; i++){
		const int offset= x_offset[i];
		const int z0= temp[4*0+i] + temp[4*2+i];
		const int z1= temp[4*0+i] - temp[4*2+i];
		const int z2= temp[4*1+i] - temp[4*3+i];
		const int z3= temp[4*1+i] + temp[4*3+i];

		block[stride*0 +offset]= ((((z0 + z3)*qmul + 128 ) >> 8)); //FIXME think about merging this into decode_residual
		block[stride*2 +offset]= ((((z1 + z2)*qmul + 128 ) >> 8));
		block[stride*8 +offset]= ((((z1 - z2)*qmul + 128 ) >> 8));
		block[stride*10+offset]= ((((z0 - z3)*qmul + 128 ) >> 8));
	}
}
#undef stride

void chroma_dc_dequant_idct_c(short *block, int qmul){
	const int stride= 16*2;
	const int xStride= 16;
	int a,b,c,d,e;

	a= block[stride*0 + xStride*0];
	b= block[stride*0 + xStride*1];
	c= block[stride*1 + xStride*0];
	d= block[stride*1 + xStride*1];

	e= a-b;
	a= a+b;
	b= c-d;
	c= c+d;

	block[stride*0 + xStride*0]= ((a+c)*qmul) >> 7;
	block[stride*0 + xStride*1]= ((e+b)*qmul) >> 7;
	block[stride*1 + xStride*0]= ((a-c)*qmul) >> 7;
	block[stride*1 + xStride*1]= ((e-b)*qmul) >> 7;
}

void h264_idct4_add_spu(uint8_t *dst, short *block, int stride)
{
  vsint16_t __vz0, __vz1, __vz2, __vz3; // used as temporal storage in for VEC_1D_DCT
  vsint16_t va0, va1, va2, va3;
  vsint16_t vtmp0, vtmp1, vtmp2, vtmp3;
  vuint16_t sat;
  vuint8_t va_u8;
  vsint16_t vdst_ss;
  vuint8_t dstperm;
  vuint8_t vdst, vdst_orig, vfdst;
  const int16_t imax = 255;
  const vsint32_t vzero = spu_splats(0);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  const int shift_dst = (unsigned int) dst  & 15;
  const vuint8_t packu16   = AVV(0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F);
  const vuint8_t mergehu8  = AVV(0x00,0x10,0x01,0x11,0x02,0x12,0x03,0x13,0x04,0x14,0x05,0x15,0x06,0x16,0x07,0x17);
  //for optimized matrix transpose:
  const vuint8_t tr0 =AVV(0x00,0x01,0x08,0x09,0x10,0x11,0x18,0x19,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00);
  const vuint8_t tr1 =AVV(0x02,0x03,0x0A,0x0B,0x12,0x13,0x1A,0x1B,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00);
  const vuint8_t tr2 =AVV(0x04,0x05,0x0C,0x0D,0x14,0x15,0x1C,0x1D,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00);
  const vuint8_t tr3 =AVV(0x06,0x07,0x0E,0x0F,0x16,0x17,0x1E,0x1F,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00);
  const vuint8_t conc =AVV(0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17);

  block[0] += 32;  // add 32 as a DC-level for rounding

  //load matrix
  vtmp0 = *(vsint16_t *)(block);
  vtmp1 = spu_rlqwbyte(vtmp0,8);
  vtmp2 = *(vsint16_t *)(block+8);
  vtmp3 = spu_rlqwbyte(vtmp2,8);

  VEC_1D_DCT(vtmp0,vtmp1,vtmp2,vtmp3,va0,va1,va2,va3);

  //concatenate first two rows of matrix
  va0=spu_shuffle(va0,va1,conc);
  //concatenate last two rows of matrix
  va2=spu_shuffle(va2,va3,conc);

  //do transpose starting from two vectors, storing as four vectors of which the second part is unused
  vtmp0 = spu_shuffle( va0, va2, tr0);
  vtmp1 = spu_shuffle( va0, va2, tr1);
  vtmp2 = spu_shuffle( va0, va2, tr2);
  vtmp3 = spu_shuffle( va0, va2, tr3);

  VEC_1D_DCT(vtmp0,vtmp1,vtmp2,vtmp3,va0,va1,va2,va3);

  // division by 64
  va0 = spu_rlmaska(va0,-6);
  va1 = spu_rlmaska(va1,-6);
  va2 = spu_rlmaska(va2,-6);
  va3 = spu_rlmaska(va3,-6);

  switch (shift_dst){
    case 0: {
      dstperm = (vuint8_t)AVV(0x10, 0x11, 0x12, 0x13, 0x04, 0x05, 0x06, 0x07,
                              0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
    } break;
    case 4: {
      dstperm = (vuint8_t)AVV(0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
                              0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
    } break;
    case 8: {
      dstperm = (vuint8_t)AVV(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
  	                      0x10, 0x11, 0x12, 0x13, 0x0C, 0x0D, 0x0E, 0x0F);
    } break;
    case 12: {
      dstperm = (vuint8_t)AVV(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                              0x08, 0x09, 0x0A, 0x0B, 0x10, 0x11, 0x12, 0x13);
    } break;
    default: {
      dstperm = (vuint8_t)AVV(0x10, 0x11, 0x12, 0x13, 0x04, 0x05, 0x06, 0x07,
                              0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
    } break;
  }

  VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(dst,shift_dst,va0,dstperm);
  dst += stride;
  VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(dst,shift_dst,va1,dstperm);
  dst += stride;
  VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(dst,shift_dst,va2,dstperm);
  dst += stride;
  VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(dst,shift_dst,va3,dstperm);
}

void h264_idct8_add_spu(uint8_t *dst, short *block, int stride)
{
	vsint16_t va0, va1, va2, va3, va4, va5, va6, va7;
	vsint16_t vza0, vza1, vza2, vza3, vza4, vza5, vza6, vza7, vzal,vzah;
	vsint16_t vzb0, vzb1, vzb2, vzb3, vzb4, vzb5, vzb6, vzb7;
	vsint16_t vtmp0, vtmp1, vtmp2, vtmp3, vtmp4, vtmp5, vtmp6, vtmp7;
	vuint16_t sat;
	vuint8_t va_u8;
	const int block_stride=8;
	vsint16_t vdst_ss;
	const int16_t imax = 255;
	const vsint32_t vzero = spu_splats(0);
	const vsint16_t vmax = (vsint16_t)spu_splats(imax);
	vuint8_t vdst, vdst_orig, vfdst;
	vuint8_t dstperm;
	const int shift_dst = (unsigned int) dst  & 15;
	const vuint8_t packu16   = AVV(0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F);
	const vuint8_t mergehu8  = AVV(0x00,0x10,0x01,0x11,0x02,0x12,0x03,0x13,0x04,0x14,0x05,0x15,0x06,0x16,0x07,0x17);
	const vuint8_t m1        = AVV(0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17);
	const vuint8_t m2        = AVV(0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F);
	const vuint8_t m3        = AVV(0x00,0x01,0x02,0x03,0x10,0x11,0x12,0x13,0x08,0x09,0x0A,0x0B,0x18,0x19,0x1A,0x1B);
	const vuint8_t m4        = AVV(0x14,0x15,0x16,0x17,0x04,0x05,0x06,0x07,0x1C,0x1D,0x1E,0x1F,0x0C,0x0D,0x0E,0x0F);
	const vuint8_t m5        = AVV(0x00,0x01,0x10,0x11,0x04,0x05,0x14,0x15,0x08,0x09,0x18,0x19,0x0C,0x0D,0x1C,0x1D);
	const vuint8_t m6        = AVV(0x12,0x13,0x02,0x03,0x16,0x17,0x06,0x07,0x1A,0x1B,0x0A,0x0B,0x1E,0x1F,0x0E,0x0F);

	block[0] += 32;  // add 32 as a DC-level for rounding

	vtmp0 = *(vsint16_t *)(block);
	vtmp1 = *(vsint16_t *)(block + block_stride);
	vtmp2 = *(vsint16_t *)(block + 2*block_stride);
	vtmp3 = *(vsint16_t *)(block + 3*block_stride);
	vtmp4 = *(vsint16_t *)(block + 4*block_stride);
	vtmp5 = *(vsint16_t *)(block + 5*block_stride);
	vtmp6 = *(vsint16_t *)(block + 6*block_stride);
	vtmp7 = *(vsint16_t *)(block + 7*block_stride);

	VEC_1D_DCT8(vtmp0,vtmp1,vtmp2,vtmp3,vtmp4,vtmp5,vtmp6,vtmp7);
	VEC_TRANSPOSE_8(vtmp0,vtmp1,vtmp2,vtmp3,vtmp4,vtmp5,vtmp6,vtmp7,va0,va1,va2,va3,va4,va5,va6,va7);
	VEC_1D_DCT8(va0, va1, va2, va3, va4, va5, va6, va7);

	va0 = spu_rlmaska(va0,-6);
	va1 = spu_rlmaska(va1,-6);
	va2 = spu_rlmaska(va2,-6);
	va3 = spu_rlmaska(va3,-6);
	va4 = spu_rlmaska(va4,-6);
	va5 = spu_rlmaska(va5,-6);
	va6 = spu_rlmaska(va6,-6);
	va7 = spu_rlmaska(va7,-6);

	if (shift_dst==8)
		dstperm = (vuint8_t)AVV(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
				   0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17);
	else																		    dstperm = (vuint8_t)AVV(0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
			0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);

	VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(dst,shift_dst,va0,dstperm);
	dst += stride;
	VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(dst,shift_dst,va1,dstperm);
	dst += stride;
	VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(dst,shift_dst,va2,dstperm);
	dst += stride;
	VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(dst,shift_dst,va3,dstperm);
	dst += stride;
	VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(dst,shift_dst,va4,dstperm);
	dst += stride;
	VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(dst,shift_dst,va5,dstperm);
	dst += stride;
	VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(dst,shift_dst,va6,dstperm);
	dst += stride;
	VEC_LOAD_UNALIGNED_U8_ADD_S16_STORE_U8(dst,shift_dst,va7,dstperm);

}

/*

void h264_idct4_add_spu(uint8_t *dst, short *block, int stride){
    int i;
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    block[0] += 32;

    for(i=0; i<4; i++){
        const int z0=  block[0 + 4*i]     +  block[2 + 4*i];
        const int z1=  block[0 + 4*i]     -  block[2 + 4*i];
        const int z2= (block[1 + 4*i]>>1) -  block[3 + 4*i];
        const int z3=  block[1 + 4*i]     + (block[3 + 4*i]>>1);

        block[0 + 4*i]= z0 + z3;
        block[1 + 4*i]= z1 + z2;
        block[2 + 4*i]= z1 - z2;
        block[3 + 4*i]= z0 - z3;
    }

    for(i=0; i<4; i++){
        const int z0=  block[i + 4*0]     +  block[i + 4*2];
        const int z1=  block[i + 4*0]     -  block[i + 4*2];
        const int z2= (block[i + 4*1]>>1) -  block[i + 4*3];
        const int z3=  block[i + 4*1]     + (block[i + 4*3]>>1);

        dst[i + 0*stride]= cm[ dst[i + 0*stride] + ((z0 + z3) >> 6) ];
        dst[i + 1*stride]= cm[ dst[i + 1*stride] + ((z1 + z2) >> 6) ];
        dst[i + 2*stride]= cm[ dst[i + 2*stride] + ((z1 - z2) >> 6) ];
        dst[i + 3*stride]= cm[ dst[i + 3*stride] + ((z0 - z3) >> 6) ];
    }
}

void h264_idct8_add_spu(uint8_t *dst, short *block, int stride){
    int i;
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    block[0] += 32;
	
    for( i = 0; i < 8; i++ )
    {
        const int a0 =  block[0+i*8] + block[4+i*8];
        const int a2 =  block[0+i*8] - block[4+i*8];
        const int a4 = (block[2+i*8]>>1) - block[6+i*8];
        const int a6 = (block[6+i*8]>>1) + block[2+i*8];

        const int b0 = a0 + a6;
        const int b2 = a2 + a4;
        const int b4 = a2 - a4;
        const int b6 = a0 - a6;

        const int a1 = -block[3+i*8] + block[5+i*8] - block[7+i*8] - (block[7+i*8]>>1);
        const int a3 =  block[1+i*8] + block[7+i*8] - block[3+i*8] - (block[3+i*8]>>1);
        const int a5 = -block[1+i*8] + block[7+i*8] + block[5+i*8] + (block[5+i*8]>>1);
        const int a7 =  block[3+i*8] + block[5+i*8] + block[1+i*8] + (block[1+i*8]>>1);

        const int b1 = (a7>>2) + a1;
        const int b3 =  a3 + (a5>>2);
        const int b5 = (a3>>2) - a5;
        const int b7 =  a7 - (a1>>2);

        block[0+i*8] = b0 + b7;
        block[7+i*8] = b0 - b7;
        block[1+i*8] = b2 + b5;
        block[6+i*8] = b2 - b5;
        block[2+i*8] = b4 + b3;
        block[5+i*8] = b4 - b3;
        block[3+i*8] = b6 + b1;
        block[4+i*8] = b6 - b1;
    }
    for( i = 0; i < 8; i++ )
    {
        const int a0 =  block[i+0*8] + block[i+4*8];
        const int a2 =  block[i+0*8] - block[i+4*8];
        const int a4 = (block[i+2*8]>>1) - block[i+6*8];
        const int a6 = (block[i+6*8]>>1) + block[i+2*8];

        const int b0 = a0 + a6;
        const int b2 = a2 + a4;
        const int b4 = a2 - a4;
        const int b6 = a0 - a6;

        const int a1 = -block[i+3*8] + block[i+5*8] - block[i+7*8] - (block[i+7*8]>>1);
        const int a3 =  block[i+1*8] + block[i+7*8] - block[i+3*8] - (block[i+3*8]>>1);
        const int a5 = -block[i+1*8] + block[i+7*8] + block[i+5*8] + (block[i+5*8]>>1);
        const int a7 =  block[i+3*8] + block[i+5*8] + block[i+1*8] + (block[i+1*8]>>1);

        const int b1 = (a7>>2) + a1;
        const int b3 =  a3 + (a5>>2);
        const int b5 = (a3>>2) - a5;
        const int b7 =  a7 - (a1>>2);
			
		dst[i + 0*stride] = cm[ dst[i + 0*stride] + ((b0 + b7) >> 6) ];
		dst[i + 1*stride] = cm[ dst[i + 1*stride] + ((b2 + b5) >> 6) ];
		dst[i + 2*stride] = cm[ dst[i + 2*stride] + ((b4 + b3) >> 6) ];
		dst[i + 3*stride] = cm[ dst[i + 3*stride] + ((b6 + b1) >> 6) ];
		dst[i + 4*stride] = cm[ dst[i + 4*stride] + ((b6 - b1) >> 6) ];
		dst[i + 5*stride] = cm[ dst[i + 5*stride] + ((b4 - b3) >> 6) ];
		dst[i + 6*stride] = cm[ dst[i + 6*stride] + ((b2 - b5) >> 6) ];
		dst[i + 7*stride] = cm[ dst[i + 7*stride] + ((b0 - b7) >> 6) ];
	}
}*/

