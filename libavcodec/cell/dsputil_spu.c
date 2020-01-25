/*
 * Copyright (c) 2009 TUDelft 
 * 
 * Cell Parallel SPU - 2DWave Macroblock Decoding. 
 */

/**
 * @file libavcodec/cell/spu/h264_main_spu.c
 * Cell Parallel SPU - 2DWave Macroblock Decoding
 * @author C C Chi <c.c.chi@student.tudelft.nl>
 * 
 * SIMD SPU kernels 
 * H.264/AVC motion compensation
 * @author Mauricio Alvarez <alvarez@ac.upc.edu>
 * @author Albert Paradis <apar7632@hotmail.com>
 */ 


#include "dsputil_spu.h"
#include "h264_idct_spu.h"
#include "h264_deblock_spu.h"
#include "types_spu.h"
#include "libavutil/intreadwrite.h"

#include <stdio.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include <assert.h>

//Luma interpolation
#define PUT_OP_U8_SPU(d, s, dst) (void) dst; d = s
#define AVG_OP_U8_SPU(d, s, dst) d = spu_avg(dst, s)

#define OP_U8_SPU                          PUT_OP_U8_SPU
#define PREFIX_h264_qpel16_h_lowpass_spu   put_h264_qpel16_h_lowpass_spu
#define PREFIX_h264_qpel16_v_lowpass_spu   put_h264_qpel16_v_lowpass_spu
#define PREFIX_h264_qpel16_hv_lowpass_spu  put_h264_qpel16_hv_lowpass_spu
#define PREFIX_h264_qpel8_h_lowpass_spu    put_h264_qpel8_h_lowpass_spu
#define PREFIX_h264_qpel8_v_lowpass_spu    put_h264_qpel8_v_lowpass_spu
#define PREFIX_h264_qpel8_hv_lowpass_spu   put_h264_qpel8_hv_lowpass_spu
#define PREFIX_h264_qpel4_h_lowpass_spu    put_h264_qpel4_h_lowpass_spu
#define PREFIX_h264_qpel4_v_lowpass_spu    put_h264_qpel4_v_lowpass_spu
#define PREFIX_h264_qpel4_hv_lowpass_spu   put_h264_qpel4_hv_lowpass_spu
#include "h264_luma_template_spu.c"
#undef OP_U8_SPU                          
#undef PREFIX_h264_qpel16_h_lowpass_spu
#undef PREFIX_h264_qpel16_v_lowpass_spu
#undef PREFIX_h264_qpel16_hv_lowpass_spu
#undef PREFIX_h264_qpel8_h_lowpass_spu
#undef PREFIX_h264_qpel8_v_lowpass_spu
#undef PREFIX_h264_qpel8_hv_lowpass_spu
#undef PREFIX_h264_qpel4_h_lowpass_spu
#undef PREFIX_h264_qpel4_v_lowpass_spu
#undef PREFIX_h264_qpel4_hv_lowpass_spu

#define OP_U8_SPU                          AVG_OP_U8_SPU
#define PREFIX_h264_qpel16_h_lowpass_spu   avg_h264_qpel16_h_lowpass_spu
#define PREFIX_h264_qpel16_v_lowpass_spu   avg_h264_qpel16_v_lowpass_spu
#define PREFIX_h264_qpel16_hv_lowpass_spu  avg_h264_qpel16_hv_lowpass_spu
#define PREFIX_h264_qpel8_h_lowpass_spu    avg_h264_qpel8_h_lowpass_spu
#define PREFIX_h264_qpel8_v_lowpass_spu    avg_h264_qpel8_v_lowpass_spu
#define PREFIX_h264_qpel8_hv_lowpass_spu   avg_h264_qpel8_hv_lowpass_spu
#define PREFIX_h264_qpel4_h_lowpass_spu    avg_h264_qpel4_h_lowpass_spu
#define PREFIX_h264_qpel4_v_lowpass_spu    avg_h264_qpel4_v_lowpass_spu
#define PREFIX_h264_qpel4_hv_lowpass_spu   avg_h264_qpel4_hv_lowpass_spu
#include "h264_luma_template_spu.c"
#undef OP_U8_SPU                          
#undef PREFIX_h264_qpel16_h_lowpass_spu
#undef PREFIX_h264_qpel16_v_lowpass_spu
#undef PREFIX_h264_qpel16_hv_lowpass_spu
#undef PREFIX_h264_qpel8_h_lowpass_spu
#undef PREFIX_h264_qpel8_v_lowpass_spu
#undef PREFIX_h264_qpel8_hv_lowpass_spu
#undef PREFIX_h264_qpel4_h_lowpass_spu
#undef PREFIX_h264_qpel4_v_lowpass_spu
#undef PREFIX_h264_qpel4_hv_lowpass_spu

#define H264_MC(OPNAME, SIZE, CODETYPE) \
static void OPNAME ## h264_qpel ## SIZE ## _mc00_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    OPNAME ## pixels ## SIZE ## _ ## CODETYPE(dst, src, dst_stride, STRIDE_Y, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc10_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){ \
    DECLARE_ALIGNED_16(uint8_t, half[16*16]);\
    put_h264_qpel ## SIZE ## _h_lowpass_ ## CODETYPE(half, src, 16, h);\
    OPNAME ## pixels ## SIZE ## _l2_ ## CODETYPE(dst, src, half, dst_stride, STRIDE_Y, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc20_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    OPNAME ## h264_qpel ## SIZE ## _h_lowpass_ ## CODETYPE(dst, src, dst_stride, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc30_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    DECLARE_ALIGNED_16(uint8_t, half[16*16]);\
    put_h264_qpel ## SIZE ## _h_lowpass_ ## CODETYPE(half, src, 16, h);\
    OPNAME ## pixels ## SIZE ## _l2_ ## CODETYPE(dst, src+1, half, dst_stride, STRIDE_Y, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc01_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    DECLARE_ALIGNED_16(uint8_t, half[16*16]);\
    put_h264_qpel ## SIZE ## _v_lowpass_ ## CODETYPE(half, src, 16, h);\
    OPNAME ## pixels ## SIZE ## _l2_ ## CODETYPE(dst, src, half, dst_stride, STRIDE_Y, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc02_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    OPNAME ## h264_qpel ## SIZE ## _v_lowpass_ ## CODETYPE(dst, src, dst_stride, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc03_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    DECLARE_ALIGNED_16(uint8_t, half[16*16]);\
    put_h264_qpel ## SIZE ## _v_lowpass_ ## CODETYPE(half, src, 16, h);\
    OPNAME ## pixels ## SIZE ## _l2_ ## CODETYPE(dst, src+STRIDE_Y, half, dst_stride, STRIDE_Y, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc11_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    DECLARE_ALIGNED_16(uint8_t, halfH[16*16]);\
    DECLARE_ALIGNED_16(uint8_t, halfV[16*16]);\
    put_h264_qpel ## SIZE ## _h_lowpass_ ## CODETYPE(halfH, src, 16, h);\
    put_h264_qpel ## SIZE ## _v_lowpass_ ## CODETYPE(halfV, src, 16, h);\
    OPNAME ## pixels ## SIZE ## _l2_ ## CODETYPE(dst, halfH, halfV, dst_stride, 16, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc31_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    DECLARE_ALIGNED_16(uint8_t, halfH[16*16]);\
    DECLARE_ALIGNED_16(uint8_t, halfV[16*16]);\
    put_h264_qpel ## SIZE ## _h_lowpass_ ## CODETYPE(halfH, src, 16, h);\
    put_h264_qpel ## SIZE ## _v_lowpass_ ## CODETYPE(halfV, src+1, 16, h);\
    OPNAME ## pixels ## SIZE ## _l2_ ## CODETYPE(dst, halfH, halfV, dst_stride, 16, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc13_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    DECLARE_ALIGNED_16(uint8_t, halfH[16*16]);\
    DECLARE_ALIGNED_16(uint8_t, halfV[16*16]);\
    put_h264_qpel ## SIZE ## _h_lowpass_ ## CODETYPE(halfH, src + STRIDE_Y, 16, h);\
    put_h264_qpel ## SIZE ## _v_lowpass_ ## CODETYPE(halfV, src, 16, h);\
    OPNAME ## pixels ## SIZE ## _l2_ ## CODETYPE(dst, halfH, halfV, dst_stride, 16, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc33_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    DECLARE_ALIGNED_16(uint8_t, halfH[16*16]);\
    DECLARE_ALIGNED_16(uint8_t, halfV[16*16]);\
    put_h264_qpel ## SIZE ## _h_lowpass_ ## CODETYPE(halfH, src + STRIDE_Y, 16, h);\
    put_h264_qpel ## SIZE ## _v_lowpass_ ## CODETYPE(halfV, src+1, 16, h);\
    OPNAME ## pixels ## SIZE ## _l2_ ## CODETYPE(dst, halfH, halfV, dst_stride, 16, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc22_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    DECLARE_ALIGNED_16(int16_t, tmp[16*(16+8)]);\
    OPNAME ## h264_qpel ## SIZE ## _hv_lowpass_ ## CODETYPE(dst, tmp, src, dst_stride, 16, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc21_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    DECLARE_ALIGNED_16(uint8_t, halfH[16*16]);\
    DECLARE_ALIGNED_16(uint8_t, halfHV[16*16]);\
    DECLARE_ALIGNED_16(int16_t, tmp[16*(16+8)]);\
    put_h264_qpel ## SIZE ## _h_lowpass_ ## CODETYPE(halfH, src, 16, h);\
    put_h264_qpel ## SIZE ## _hv_lowpass_ ## CODETYPE(halfHV, tmp, src, 16, 16, h);\
    OPNAME ## pixels ## SIZE ## _l2_ ## CODETYPE(dst, halfH, halfHV, dst_stride, 16, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc23_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    DECLARE_ALIGNED_16(uint8_t, halfH[16*16]);\
    DECLARE_ALIGNED_16(uint8_t, halfHV[16*16]);\
    DECLARE_ALIGNED_16(int16_t, tmp[16*(16+8)]);\
    put_h264_qpel ## SIZE ## _h_lowpass_ ## CODETYPE(halfH, src + STRIDE_Y, 16, h);\
    put_h264_qpel ## SIZE ## _hv_lowpass_ ## CODETYPE(halfHV, tmp, src, 16, 16, h);\
    OPNAME ## pixels ## SIZE ## _l2_ ## CODETYPE(dst, halfH, halfHV, dst_stride, 16, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc12_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    DECLARE_ALIGNED_16(uint8_t, halfV[16*16]);\
    DECLARE_ALIGNED_16(uint8_t, halfHV[16*16]);\
    DECLARE_ALIGNED_16(int16_t, tmp[16*(16+8)]);\
    put_h264_qpel ## SIZE ## _v_lowpass_ ## CODETYPE(halfV, src, 16, h);\
    put_h264_qpel ## SIZE ## _hv_lowpass_ ## CODETYPE(halfHV, tmp, src, 16, 16, h);\
    OPNAME ## pixels ## SIZE ## _l2_ ## CODETYPE(dst, halfV, halfHV, dst_stride, 16, h);\
}\
\
static void OPNAME ## h264_qpel ## SIZE ## _mc32_ ## CODETYPE(uint8_t *dst, uint8_t *src, int dst_stride, int h){\
    DECLARE_ALIGNED_16(uint8_t, halfV[16*16]);\
    DECLARE_ALIGNED_16(uint8_t, halfHV[16*16]);\
    DECLARE_ALIGNED_16(int16_t, tmp[16*(16+8)]);\
    put_h264_qpel ## SIZE ## _v_lowpass_ ## CODETYPE(halfV, src+1, 16, h);\
    put_h264_qpel ## SIZE ## _hv_lowpass_ ## CODETYPE(halfHV, tmp, src, 16, 16, h);\
    OPNAME ## pixels ## SIZE ## _l2_ ## CODETYPE(dst, halfV, halfHV, dst_stride, 16, h);\
}\


/**************************/
/* put pixels functions   */
/*************************/

static void put_pixels16_l2_spu( uint8_t * dst, const uint8_t * src1,
                                    const uint8_t * src2, int dst_stride,
                                    int src_stride1, int h)
{
  int i;

  const int perm_src1 = (unsigned int) src1 & 15;

  for (i=0; i<h; i++){
      //unaligned load of src1
      const vuint8_t srctmpa1 = *(vuint8_t *)(src1);
      const vuint8_t srctmpa2 = *(vuint8_t *)(src1+16);
      const vuint8_t srca= spu_or(spu_slqwbyte(srctmpa1, perm_src1), spu_rlmaskqwbyte(srctmpa2, perm_src1-16));

      //aligned load of src2
      const vuint8_t srcb = *(vuint8_t *)(src2);

      //average and rounding
      const vuint8_t avgc = spu_avg(srca,srcb);

      // 16x16 dest luma blocks are always aligned
      *(vuint8_t *)dst=avgc;

      src1 +=src_stride1;
      src2 +=16;
      dst  +=dst_stride;
  }
}

static void avg_pixels16_l2_spu( uint8_t * dst, const uint8_t * src1,
                                    const uint8_t * src2, int dst_stride,
                                    int src_stride1, int h)
{
  int i;

  const int perm_src1 = (unsigned int) src1 & 15;

  for (i=0; i<h; i++){
      //unaligned load of src1
      const vuint8_t srctmpa1 = *(vuint8_t *)(src1);
      const vuint8_t srctmpa2 = *(vuint8_t *)(src1+16);
      const vuint8_t srca= spu_or(spu_slqwbyte(srctmpa1, perm_src1), spu_rlmaskqwbyte(srctmpa2, perm_src1-16));

      //aligned load of src2
      const vuint8_t srcb = *(vuint8_t *)(src2);

      //average and rounding
      const vuint8_t avgc = spu_avg(spu_avg(srca,srcb), *(vuint8_t *)dst);

      // 16x16 dest luma blocks are always aligned
      *(vuint8_t *)dst=avgc;

      src1 +=src_stride1;
      src2 +=16;
      dst  +=dst_stride;
  }
}

// next one assumes that ((line_size % 16) == 0)
void put_pixels16_spu(uint8_t *dst, const uint8_t *src, int dst_stride, int src_stride, int h)
{
    register vector unsigned char pixelsv1, pixelsv2;
    register vector unsigned char pixelsv1B, pixelsv2B;
    register vector unsigned char pixelsv1C, pixelsv2C;
    register vector unsigned char pixelsv1D, pixelsv2D;

    const int perm = (unsigned int) src & 15;
    int i;
	register int line_size   = src_stride;
    register int line_size_2 = line_size << 1;
    register int line_size_3 = line_size + line_size_2;
    register int line_size_4 = line_size << 2;

    register int dst_stride_2 = dst_stride << 1;
    register int dst_stride_3 = dst_stride_2 + dst_stride;
    register int dst_stride_4 = dst_stride << 2;

    for(i=0; i<h; i+=4) {
      pixelsv1 = *(vuint8_t *)(src);
      pixelsv2 = *(vuint8_t *)(src+16);
      pixelsv1B = *(vuint8_t *)(src + line_size);
      pixelsv2B = *(vuint8_t *)(src+16 + line_size);
      pixelsv1C = *(vuint8_t *)(src + line_size_2);
      pixelsv2C = *(vuint8_t *)(src+16 + line_size_2);
      pixelsv1D = *(vuint8_t *)(src + line_size_3);
      pixelsv2D = *(vuint8_t *)(src+16 + line_size_3);

      *(vuint8_t *) dst                 = spu_or(spu_slqwbyte(pixelsv1, perm), spu_rlmaskqwbyte(pixelsv2, perm-16));
      *(vuint8_t *)(dst +   dst_stride) = spu_or(spu_slqwbyte(pixelsv1B, perm), spu_rlmaskqwbyte(pixelsv2B, perm-16));
      *(vuint8_t *)(dst + dst_stride_2) = spu_or(spu_slqwbyte(pixelsv1C, perm), spu_rlmaskqwbyte(pixelsv2C, perm-16));
      *(vuint8_t *)(dst + dst_stride_3) = spu_or(spu_slqwbyte(pixelsv1D, perm), spu_rlmaskqwbyte(pixelsv2D, perm-16));

      src+= line_size_4;
      dst+= dst_stride_4;
    }
}

// next one assumes that ((line_size % 16) == 0)
void avg_pixels16_spu(uint8_t *dst, const uint8_t *src, int dst_stride, int src_stride, int h)
{
    register vector unsigned char pixelsv1, pixelsv2;
    register vector unsigned char pixelsv1B, pixelsv2B;
    register vector unsigned char pixelsv1C, pixelsv2C;
    register vector unsigned char pixelsv1D, pixelsv2D;

    const int perm = (unsigned int) src & 15;
    int i;
	register int line_size   = src_stride;
    register int line_size_2 = line_size << 1;
    register int line_size_3 = line_size + line_size_2;
    register int line_size_4 = line_size << 2;

    register int dst_stride_2 = dst_stride << 1;
    register int dst_stride_3 = dst_stride_2 + dst_stride;
    register int dst_stride_4 = dst_stride << 2;


    for(i=0; i<h; i+=4) {
      pixelsv1 = *(vuint8_t *)(src);
      pixelsv2 = *(vuint8_t *)(src+16);
      pixelsv1B = *(vuint8_t *)(src + line_size);
      pixelsv2B = *(vuint8_t *)(src+16 + line_size);
      pixelsv1C = *(vuint8_t *)(src + line_size_2);
      pixelsv2C = *(vuint8_t *)(src+16 + line_size_2);
      pixelsv1D = *(vuint8_t *)(src + line_size_3);
      pixelsv2D = *(vuint8_t *)(src+16 + line_size_3);

      *(vuint8_t *)dst = spu_avg(spu_or(spu_slqwbyte(pixelsv1, perm), spu_rlmaskqwbyte(pixelsv2, perm-16)), *(vuint8_t *)dst);
      *(vuint8_t *)(dst + dst_stride) = spu_avg(spu_or(spu_slqwbyte(pixelsv1B, perm), spu_rlmaskqwbyte(pixelsv2B, perm-16)), *(vuint8_t *)(dst+dst_stride));
      *(vuint8_t *)(dst + dst_stride_2) = spu_avg(spu_or(spu_slqwbyte(pixelsv1C, perm), spu_rlmaskqwbyte(pixelsv2C, perm-16)), *(vuint8_t *)(dst+dst_stride_2));
      *(vuint8_t *)(dst + dst_stride_3) = spu_avg(spu_or(spu_slqwbyte(pixelsv1D, perm), spu_rlmaskqwbyte(pixelsv2D, perm-16)), *(vuint8_t *)(dst+dst_stride_3));

      src+= line_size_4;
      dst+= dst_stride_4;
    }
}

void put_pixels8_l2_spu(uint8_t * dst, const uint8_t * src1, const uint8_t * src2,
				   int dst_stride, int src_stride1, int h)
{
  int i;

  const int perm_src1 = (unsigned int) src1 & 15;
  const int shift_dst = (unsigned int) dst & 15;

  // 8x dest luma blocks are aligned or desaligned by 8
  vuint8_t dstmask;
  const vuint8_t dst8mask1= {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst8mask2= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17};

  if(shift_dst==0){
    dstmask = dst8mask1;
  }
  else{
    dstmask = dst8mask2;
  }

  for (i=0; i<h; i++){
      //unaligned load of src1
      const vuint8_t srctmpa1 = *(vuint8_t *)(src1);
      const vuint8_t srctmpa2 = *(vuint8_t *)(src1+16);
      const vuint8_t srca= spu_or(spu_slqwbyte(srctmpa1, perm_src1), spu_rlmaskqwbyte(srctmpa2, perm_src1-16));

      //aligned load of src2
      const vuint8_t srcb = *(vuint8_t *)(src2);

      //average and rounding
      const vuint8_t avgc = spu_avg(srca,srcb);

      const vuint8_t dst1 = *(vuint8_t *)dst;

      const vuint8_t davgc = spu_shuffle(dst1, avgc, dstmask);

      *(vuint8_t *)dst=davgc;

      src1 +=src_stride1;
      src2 +=16;
      dst  +=dst_stride;
  }
}

void avg_pixels8_l2_spu(uint8_t * dst, const uint8_t * src1, const uint8_t * src2,
				   int dst_stride, int src_stride1, int h)
{
  int i;

  const int perm_src1 = (unsigned int) src1 & 15;
  const int shift_dst = (unsigned int) dst & 15;

  // 8x dest luma blocks are aligned or desaligned by 8
  vuint8_t dstmask;
  const vuint8_t dst8mask1= {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst8mask2= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17};

  if(shift_dst==0){
    dstmask = dst8mask1;
  }
  else{
    dstmask = dst8mask2;
  }

  for (i=0; i<h; i++){
      //unaligned load of src1
      const vuint8_t srctmpa1 = *(vuint8_t *)(src1);
      const vuint8_t srctmpa2 = *(vuint8_t *)(src1+16);
      const vuint8_t srca= spu_or(spu_slqwbyte(srctmpa1, perm_src1), spu_rlmaskqwbyte(srctmpa2, perm_src1-16));

      //aligned load of src2
      const vuint8_t srcb = *(vuint8_t *)(src2);

      //average and rounding
      const vuint8_t avgc = spu_avg(srca,srcb);

      const vuint8_t dst1 = *(vuint8_t *)dst;

      const vuint8_t davgc1 = spu_shuffle(dst1, avgc, dstmask);

      const vuint8_t davgc = spu_avg(dst1,davgc1);

      *(vuint8_t *)dst=davgc;

      src1 +=src_stride1;
      src2 +=16;
      dst  +=dst_stride;
  }
}

// next one assumes that ((line_size % 16) == 0)
void put_pixels8_spu(uint8_t *dst, const uint8_t *src, int dst_stride, int src_stride, int h)
{
    register vector unsigned char pixelsv1A, pixelsv2A;
    register vector unsigned char pixelsv1B, pixelsv2B;
    register vector unsigned char pixelsv1C, pixelsv2C;
    register vector unsigned char pixelsv1D, pixelsv2D;

    const int perm = (unsigned int) src & 15;
    const int shift_dst = (unsigned int) dst & 15;

    // 8x dest luma blocks are aligned or desaligned by 8
    vuint8_t dstmask;
    const vuint8_t dst8mask1= {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
    const vuint8_t dst8mask2= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17};

    if(shift_dst==0){
      dstmask = dst8mask1;
    }
    else{
      dstmask = dst8mask2;
    }

    int i;
	register int line_size   = src_stride;
    register int line_size_2 = line_size << 1;
    register int line_size_3 = line_size + line_size_2;
    register int line_size_4 = line_size << 2;

    register int dst_stride_2 = dst_stride << 1;
    register int dst_stride_3 = dst_stride_2 + dst_stride;
    register int dst_stride_4 = dst_stride << 2;

    for(i=0; i<h; i+=4) {
      pixelsv1A = *(vuint8_t *)(src);
      pixelsv2A = *(vuint8_t *)(src+16);
      pixelsv1B = *(vuint8_t *)(src + line_size);
      pixelsv2B = *(vuint8_t *)(src+16 + line_size);
      pixelsv1C = *(vuint8_t *)(src + line_size_2);
      pixelsv2C = *(vuint8_t *)(src+16 + line_size_2);
      pixelsv1D = *(vuint8_t *)(src + line_size_3);
      pixelsv2D = *(vuint8_t *)(src+16 + line_size_3);

      const vuint8_t block1 = *(vuint8_t *)dst;
      const vuint8_t put1 = spu_shuffle(block1, spu_or(spu_slqwbyte(pixelsv1A, perm), spu_rlmaskqwbyte(pixelsv2A, perm-16)), dstmask);
      const vuint8_t block2 = *(vuint8_t *)(dst+dst_stride);
      const vuint8_t put2 = spu_shuffle(block2, spu_or(spu_slqwbyte(pixelsv1B, perm), spu_rlmaskqwbyte(pixelsv2B, perm-16)), dstmask);
      const vuint8_t block3 = *(vuint8_t *)(dst+2*dst_stride);
      const vuint8_t put3 = spu_shuffle(block3, spu_or(spu_slqwbyte(pixelsv1C, perm), spu_rlmaskqwbyte(pixelsv2C, perm-16)), dstmask);
      const vuint8_t block4 = *(vuint8_t *)(dst+3*dst_stride);
      const vuint8_t put4 = spu_shuffle(block4, spu_or(spu_slqwbyte(pixelsv1D, perm), spu_rlmaskqwbyte(pixelsv2D, perm-16)), dstmask);

      *(vuint8_t *) dst = put1;
      *(vuint8_t *)(dst + dst_stride) = put2;
      *(vuint8_t *)(dst + dst_stride_2) = put3;
      *(vuint8_t *)(dst + dst_stride_3) = put4;

      src += line_size_4;
      dst += dst_stride_4;
    }
}

// next one assumes that ((line_size % 16) == 0)
void avg_pixels8_spu(uint8_t *dst, const uint8_t *src, int dst_stride, int src_stride, int h)
{
    register vector unsigned char pixelsv1A, pixelsv2A;
    register vector unsigned char pixelsv1B, pixelsv2B;
    register vector unsigned char pixelsv1C, pixelsv2C;
    register vector unsigned char pixelsv1D, pixelsv2D;

    const int perm = (unsigned int) src & 15;
    const int shift_dst = (unsigned int) dst & 15;

    // 8x dest luma blocks are aligned or desaligned by 8
    vuint8_t dstmask;
    const vuint8_t dst8mask1= {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
    const vuint8_t dst8mask2= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17};

    if(shift_dst==0){
      dstmask = dst8mask1;
    }
    else{
      dstmask = dst8mask2;
    }

    int i;
	register int line_size   = src_stride;
    register int line_size_2 = line_size << 1;
    register int line_size_3 = line_size + line_size_2;
    register int line_size_4 = line_size << 2;

	register int dst_stride_2 = dst_stride << 1;
    register int dst_stride_3 = dst_stride_2 + dst_stride;
    register int dst_stride_4 = dst_stride << 2;

    for(i=0; i<h; i+=4) {
      pixelsv1A = *(vuint8_t *)(src);
      pixelsv2A = *(vuint8_t *)(src+16);
      pixelsv1B = *(vuint8_t *)(src + line_size);
      pixelsv2B = *(vuint8_t *)(src+16 + line_size);
      pixelsv1C = *(vuint8_t *)(src + line_size_2);
      pixelsv2C = *(vuint8_t *)(src+16 + line_size_2);
      pixelsv1D = *(vuint8_t *)(src + line_size_3);
      pixelsv2D = *(vuint8_t *)(src+16 + line_size_3);

      const vuint8_t block1 = *(vuint8_t *) dst;
      const vuint8_t put1a = spu_shuffle(block1, spu_or(spu_slqwbyte(pixelsv1A, perm), spu_rlmaskqwbyte(pixelsv2A, perm-16)), dstmask);
      const vuint8_t put1 = spu_avg(block1,put1a);

      const vuint8_t block2 = *(vuint8_t *)(dst + dst_stride);
      const vuint8_t put2a = spu_shuffle(block2, spu_or(spu_slqwbyte(pixelsv1B, perm), spu_rlmaskqwbyte(pixelsv2B, perm-16)), dstmask);
      const vuint8_t put2 = spu_avg(block2,put2a);

      const vuint8_t block3 = *(vuint8_t *)(dst + dst_stride_2);
      const vuint8_t put3a = spu_shuffle(block3, spu_or(spu_slqwbyte(pixelsv1C, perm), spu_rlmaskqwbyte(pixelsv2C, perm-16)), dstmask);
      const vuint8_t put3 = spu_avg(block3,put3a);

      const vuint8_t block4 = *(vuint8_t *)(dst + dst_stride_3);
      const vuint8_t put4a = spu_shuffle(block4, spu_or(spu_slqwbyte(pixelsv1D, perm), spu_rlmaskqwbyte(pixelsv2D, perm-16)), dstmask);
      const vuint8_t put4 = spu_avg(block4,put4a);

      *(vuint8_t *) dst = put1;
      *(vuint8_t *)(dst + dst_stride) = put2;
      *(vuint8_t *)(dst + dst_stride_2) = put3;
      *(vuint8_t *)(dst + dst_stride_3) = put4;

      src+= line_size_4;
      dst+= dst_stride_4;
    }
}

void put_pixels4_l2_spu(uint8_t * dst, const uint8_t * src1, const uint8_t * src2,
				   int dst_stride, int src_stride1, int h)
{
  int i;

  const int perm_src1 = (unsigned int) src1 & 15;
  const int shift_dst = (unsigned int) dst & 15;

  // 4x dest luma blocks are desaligned by 0, 4, 8, or 12
  vuint8_t dstmask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vuint8_t dstmask0=  {0x10,0x11,0x12,0x13,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask4=  {0x00,0x01,0x02,0x03,0x10,0x11,0x12,0x13,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask8=  {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask12= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x10,0x11,0x12,0x13};

  switch(shift_dst){
    case 0:  dstmask = dstmask0;
             break;
    case 4:  dstmask = dstmask4;
             break;
    case 8:  dstmask = dstmask8;
             break;
    case 12: dstmask = dstmask12;
             break;
  }

  for (i=0; i<h; i++){
      //unaligned load of src1
      const vuint8_t srctmpa1 = *(vuint8_t *)(src1);
      const vuint8_t srctmpa2 = *(vuint8_t *)(src1+16);
      const vuint8_t srca= spu_or(spu_slqwbyte(srctmpa1, perm_src1), spu_rlmaskqwbyte(srctmpa2, perm_src1-16));

      //aligned load of src2
      const vuint8_t srcb = *(vuint8_t *)(src2);

      //average and rounding
      const vuint8_t avgc = spu_avg(srca,srcb);

      const vuint8_t dst1 = *(vuint8_t *)dst;

      const vuint8_t davgc = spu_shuffle(dst1, avgc, dstmask);

      *(vuint8_t *)dst=davgc;

      src1 +=src_stride1;
      src2 +=16;
      dst  +=dst_stride;
  }
}

void avg_pixels4_l2_spu(uint8_t * dst, const uint8_t * src1, const uint8_t * src2,
				   int dst_stride, int src_stride1, int h)
{
  int i;

  const int perm_src1 = (unsigned int) src1 & 15;
  const int shift_dst = (unsigned int) dst & 15;

  // 4x dest luma blocks are desaligned by 0, 4, 8, or 12
  vuint8_t dstmask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vuint8_t dstmask0=  {0x10,0x11,0x12,0x13,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask4=  {0x00,0x01,0x02,0x03,0x10,0x11,0x12,0x13,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask8=  {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask12= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x10,0x11,0x12,0x13};

  switch(shift_dst){
    case 0:  dstmask = dstmask0;
             break;
    case 4:  dstmask = dstmask4;
             break;
    case 8:  dstmask = dstmask8;
             break;
    case 12: dstmask = dstmask12;
             break;
  }

  for (i=0; i<h; i++){
      //unaligned load of src1
      const vuint8_t srctmpa1 = *(vuint8_t *)(src1);
      const vuint8_t srctmpa2 = *(vuint8_t *)(src1+16);
      const vuint8_t srca= spu_or(spu_slqwbyte(srctmpa1, perm_src1), spu_rlmaskqwbyte(srctmpa2, perm_src1-16));

      //aligned load of src2
      const vuint8_t srcb = *(vuint8_t *)(src2);

      //average and rounding
      const vuint8_t avgc = spu_avg(srca,srcb);

      const vuint8_t dst1 = *(vuint8_t *)dst;

      const vuint8_t davgc1 = spu_shuffle(dst1, avgc, dstmask);

      const vuint8_t davgc = spu_avg(dst1,davgc1);

      *(vuint8_t *)dst=davgc;

      src1 +=src_stride1;
      src2 +=16;
      dst  +=dst_stride;
  }
}

// next one assumes that ((line_size % 16) == 0)
void put_pixels4_spu(uint8_t *dst, const uint8_t *src, int dst_stride, int src_stride, int h)
{
    register vector unsigned char pixelsv1A, pixelsv2A;
    register vector unsigned char pixelsv1B, pixelsv2B;
    register vector unsigned char pixelsv1C, pixelsv2C;
    register vector unsigned char pixelsv1D, pixelsv2D;

    const int perm = (unsigned int) src & 15;
    const int shift_dst = (unsigned int) dst & 15;

    // 4x dest luma blocks are desaligned by 0, 4, 8, or 12
    vuint8_t dstmask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const vuint8_t dstmask0=  {0x10,0x11,0x12,0x13,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
    const vuint8_t dstmask4=  {0x00,0x01,0x02,0x03,0x10,0x11,0x12,0x13,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
    const vuint8_t dstmask8=  {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x0C,0x0D,0x0E,0x0F};
    const vuint8_t dstmask12= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x10,0x11,0x12,0x13};

    switch(shift_dst){
      case 0:  dstmask = dstmask0;
               break;
      case 4:  dstmask = dstmask4;
               break;
      case 8:  dstmask = dstmask8;
               break;
      case 12: dstmask = dstmask12;
               break;
    }

    int i;
	register int line_size   = src_stride;
    register int line_size_2 = line_size << 1;
    register int line_size_3 = line_size + line_size_2;
    register int line_size_4 = line_size << 2;

	register int dst_stride_2 = dst_stride << 1;
    register int dst_stride_3 = dst_stride_2 + dst_stride;
    register int dst_stride_4 = dst_stride << 2;

    for(i=0; i<h; i+=4) {
	  pixelsv1A = *(vuint8_t *)(src);
      pixelsv2A = *(vuint8_t *)(src+16);
      pixelsv1B = *(vuint8_t *)(src + line_size);
      pixelsv2B = *(vuint8_t *)(src+16 + line_size);
      pixelsv1C = *(vuint8_t *)(src + line_size_2);
      pixelsv2C = *(vuint8_t *)(src+16 + line_size_2);
      pixelsv1D = *(vuint8_t *)(src + line_size_3);
      pixelsv2D = *(vuint8_t *)(src+16 + line_size_3);

      const vuint8_t block1 = *(vuint8_t *)dst;
      const vuint8_t put1 = spu_shuffle(block1, spu_or(spu_slqwbyte(pixelsv1A, perm), spu_rlmaskqwbyte(pixelsv2A, perm-16)), dstmask);
      const vuint8_t block2 = *(vuint8_t *)(dst+dst_stride);
      const vuint8_t put2 = spu_shuffle(block2, spu_or(spu_slqwbyte(pixelsv1B, perm), spu_rlmaskqwbyte(pixelsv2B, perm-16)), dstmask);
      const vuint8_t block3 = *(vuint8_t *)(dst+dst_stride_2);
      const vuint8_t put3 = spu_shuffle(block3, spu_or(spu_slqwbyte(pixelsv1C, perm), spu_rlmaskqwbyte(pixelsv2C, perm-16)), dstmask);
      const vuint8_t block4 = *(vuint8_t *)(dst+dst_stride_3);
      const vuint8_t put4 = spu_shuffle(block4, spu_or(spu_slqwbyte(pixelsv1D, perm), spu_rlmaskqwbyte(pixelsv2D, perm-16)), dstmask);

      *(vuint8_t *) dst = put1;
      *(vuint8_t *)(dst + dst_stride) = put2;
      *(vuint8_t *)(dst + dst_stride_2) = put3;
      *(vuint8_t *)(dst + dst_stride_3) = put4;

      src += line_size_4;
      dst += dst_stride_4;
    }
}

// next one assumes that ((line_size % 16) == 0)
void avg_pixels4_spu(uint8_t *dst, const uint8_t *src, int dst_stride, int src_stride, int h)
{
    register vector unsigned char pixelsv1A, pixelsv2A;
    register vector unsigned char pixelsv1B, pixelsv2B;
    register vector unsigned char pixelsv1C, pixelsv2C;
    register vector unsigned char pixelsv1D, pixelsv2D;

    const int perm = (unsigned int) src & 15;
    const int shift_dst = (unsigned int) dst & 15;

    // 4x dest luma blocks are desaligned by 0, 4, 8, or 12
    vuint8_t dstmask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const vuint8_t dstmask0=  {0x10,0x11,0x12,0x13,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
    const vuint8_t dstmask4=  {0x00,0x01,0x02,0x03,0x10,0x11,0x12,0x13,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
    const vuint8_t dstmask8=  {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x0C,0x0D,0x0E,0x0F};
    const vuint8_t dstmask12= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x10,0x11,0x12,0x13};

    switch(shift_dst){
      case 0:  dstmask = dstmask0;
               break;
      case 4:  dstmask = dstmask4;
               break;
      case 8:  dstmask = dstmask8;
               break;
      case 12: dstmask = dstmask12;
               break;
    }

    int i;
	register int line_size   = src_stride;
    register int line_size_2 = line_size << 1;
    register int line_size_3 = line_size + line_size_2;
    register int line_size_4 = line_size << 2;

	register int dst_stride_2 = dst_stride << 1;
    register int dst_stride_3 = dst_stride_2 + dst_stride;
    register int dst_stride_4 = dst_stride << 2;

    for(i=0; i<h; i+=4) {
	  pixelsv1A = *(vuint8_t *)(src);
      pixelsv2A = *(vuint8_t *)(src+16);
      pixelsv1B = *(vuint8_t *)(src + line_size);
      pixelsv2B = *(vuint8_t *)(src+16 + line_size);
      pixelsv1C = *(vuint8_t *)(src + line_size_2);
      pixelsv2C = *(vuint8_t *)(src+16 + line_size_2);
      pixelsv1D = *(vuint8_t *)(src + line_size_3);
      pixelsv2D = *(vuint8_t *)(src+16 + line_size_3);

      const vuint8_t block1 = *(vuint8_t *) dst;
      const vuint8_t put1a = spu_shuffle(block1, spu_or(spu_slqwbyte(pixelsv1A, perm), spu_rlmaskqwbyte(pixelsv2A, perm-16)), dstmask);
      const vuint8_t put1 = spu_avg(block1,put1a);

      const vuint8_t block2 = *(vuint8_t *)(dst + dst_stride);
      const vuint8_t put2a = spu_shuffle(block2, spu_or(spu_slqwbyte(pixelsv1B, perm), spu_rlmaskqwbyte(pixelsv2B, perm-16)), dstmask);
      const vuint8_t put2 = spu_avg(block2,put2a);

      const vuint8_t block3 = *(vuint8_t *)(dst + dst_stride_2);
      const vuint8_t put3a = spu_shuffle(block3, spu_or(spu_slqwbyte(pixelsv1C, perm), spu_rlmaskqwbyte(pixelsv2C, perm-16)), dstmask);
      const vuint8_t put3 = spu_avg(block3,put3a);

      const vuint8_t block4 = *(vuint8_t *)(dst + dst_stride_3);
      const vuint8_t put4a = spu_shuffle(block4, spu_or(spu_slqwbyte(pixelsv1D, perm), spu_rlmaskqwbyte(pixelsv2D, perm-16)), dstmask);
      const vuint8_t put4 = spu_avg(block4,put4a);

      *(vuint8_t *) dst = put1;
      *(vuint8_t *)(dst + dst_stride) = put2;
      *(vuint8_t *)(dst + dst_stride_2) = put3;
      *(vuint8_t *)(dst + dst_stride_3) = put4;

      src+= line_size_4;
      dst+= dst_stride_4;
    }
}

/* Here we create all the interpolation modes H.264 motion compensation stage for luma */
  H264_MC(put_, 16, spu)
  H264_MC(put_, 8, spu)
  H264_MC(put_, 4, spu)

  H264_MC(avg_, 16, spu)
  H264_MC(avg_, 8, spu)
  H264_MC(avg_, 4, spu)


//Chroma interpolation:

#define OP_U8_SPU                          PUT_OP_U8_SPU
#define PREFIX_h264_chroma_mc8_spu         put_h264_chroma_mc8_spu
#define PREFIX_h264_chroma_mc4_spu         put_h264_chroma_mc4_spu
#define PREFIX_h264_chroma_mc2_spu         put_h264_chroma_mc2_spu
#include "h264_chroma_template_spu.c"
#undef OP_U8_SPU
#undef PREFIX_h264_chroma_mc8_spu
#undef PREFIX_h264_chroma_mc4_spu
#undef PREFIX_h264_chroma_mc2_spu

#define OP_U8_SPU                          AVG_OP_U8_SPU
#define PREFIX_h264_chroma_mc8_spu         avg_h264_chroma_mc8_spu
#define PREFIX_h264_chroma_mc4_spu         avg_h264_chroma_mc4_spu
#define PREFIX_h264_chroma_mc2_spu         avg_h264_chroma_mc2_spu
#include "h264_chroma_template_spu.c"
#undef OP_U8_SPU
#undef PREFIX_h264_chroma_mc8_spu
#undef PREFIX_h264_chroma_mc4_spu
#undef PREFIX_h264_chroma_mc2_spu

// Weight and Biweight functions

#define op_scale1(x)  dst[x] = av_clip_uint8( (dst[x]*weight + offset) >> log2_denom )
#define op_scale2(x)  dst[x] = av_clip_uint8( (src[x]*weights + dst[x]*weightd + offset) >> (log2_denom+1))
#define H264_WEIGHT(W,H) \
static void weight_h264_pixels ## W ## x ## H ## _c(uint8_t *dst, int stride, int log2_denom, int weight, int offset){ \
    int y; \
    offset <<= log2_denom; \
    if(log2_denom) offset += 1<<(log2_denom-1); \
    for(y=0; y<H; y++, dst += stride){ \
        op_scale1(0); \
        op_scale1(1); \
        if(W==2) continue; \
        op_scale1(2); \
        op_scale1(3); \
        if(W==4) continue; \
        op_scale1(4); \
        op_scale1(5); \
        op_scale1(6); \
        op_scale1(7); \
        if(W==8) continue; \
        op_scale1(8); \
        op_scale1(9); \
        op_scale1(10); \
        op_scale1(11); \
        op_scale1(12); \
        op_scale1(13); \
        op_scale1(14); \
        op_scale1(15); \
    } \
} \
static void biweight_h264_pixels ## W ## x ## H ## _c(uint8_t *dst, uint8_t *src, int dst_stride, int src_stride, int log2_denom, int weightd, int weights, int offset){ \
    int y; \
    offset = ((offset + 1) | 1) << log2_denom; \
    for(y=0; y<H; y++, dst += dst_stride, src += src_stride){ \
        op_scale2(0); \
        op_scale2(1); \
        if(W==2) continue; \
        op_scale2(2); \
        op_scale2(3); \
        if(W==4) continue; \
        op_scale2(4); \
        op_scale2(5); \
        op_scale2(6); \
        op_scale2(7); \
        if(W==8) continue; \
        op_scale2(8); \
        op_scale2(9); \
        op_scale2(10); \
        op_scale2(11); \
        op_scale2(12); \
        op_scale2(13); \
        op_scale2(14); \
        op_scale2(15); \
    } \
}

H264_WEIGHT(16,16)
H264_WEIGHT(16,8)
H264_WEIGHT(8,16)
H264_WEIGHT(8,8)
H264_WEIGHT(8,4)
H264_WEIGHT(4,8)
H264_WEIGHT(4,4)
H264_WEIGHT(4,2)
H264_WEIGHT(2,4)
H264_WEIGHT(2,2)

#undef op_scale1
#undef op_scale2
#undef H264_WEIGHT

/////////////////////////////////////////////////////////////////////////////////////////

static inline void h264_loop_filter_luma_c(uint8_t *pix, int xstride, int ystride, int alpha, int beta, int8_t *tc0)
{
    int i, d;
    for( i = 0; i < 4; i++ ) {
        if( tc0[i] < 0 ) {
            pix += 4*ystride;
            continue;
        }
        for( d = 0; d < 4; d++ ) {
            const int p0 = pix[-1*xstride];
            const int p1 = pix[-2*xstride];
            const int p2 = pix[-3*xstride];
            const int q0 = pix[0];
            const int q1 = pix[1*xstride];
            const int q2 = pix[2*xstride];

            if( FFABS( p0 - q0 ) < alpha &&
                FFABS( p1 - p0 ) < beta &&
                FFABS( q1 - q0 ) < beta ) {

                int tc = tc0[i];
                int i_delta;

                if( FFABS( p2 - p0 ) < beta ) {
                    pix[-2*xstride] = p1 + av_clip( (( p2 + ( ( p0 + q0 + 1 ) >> 1 ) ) >> 1) - p1, -tc0[i], tc0[i] );
                    tc++;
                }
                if( FFABS( q2 - q0 ) < beta ) {
                    pix[   xstride] = q1 + av_clip( (( q2 + ( ( p0 + q0 + 1 ) >> 1 ) ) >> 1) - q1, -tc0[i], tc0[i] );
                    tc++;
                }

                i_delta = av_clip( (((q0 - p0 ) << 2) + (p1 - q1) + 4) >> 3, -tc, tc );
                pix[-xstride] = av_clip_uint8( p0 + i_delta );    /* p0' */
                pix[0]        = av_clip_uint8( q0 - i_delta );    /* q0' */
            }
            pix += ystride;
        }
    }
}
static void h264_v_loop_filter_luma_c(uint8_t *pix, int stride, int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_luma_c(pix, stride, 1, alpha, beta, tc0);
}
static void h264_h_loop_filter_luma_c(uint8_t *pix, int stride, int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_luma_c(pix, 1, stride, alpha, beta, tc0);
}

static inline void h264_loop_filter_luma_intra_c(uint8_t *pix, int xstride, int ystride, int alpha, int beta)
{
    int d;
    for( d = 0; d < 16; d++ ) {
        const int p2 = pix[-3*xstride];
        const int p1 = pix[-2*xstride];
        const int p0 = pix[-1*xstride];

        const int q0 = pix[ 0*xstride];
        const int q1 = pix[ 1*xstride];
        const int q2 = pix[ 2*xstride];

        if( FFABS( p0 - q0 ) < alpha &&
            FFABS( p1 - p0 ) < beta &&
            FFABS( q1 - q0 ) < beta ) {

            if(FFABS( p0 - q0 ) < (( alpha >> 2 ) + 2 )){
                if( FFABS( p2 - p0 ) < beta)
                {
                    const int p3 = pix[-4*xstride];
                    /* p0', p1', p2' */
                    pix[-1*xstride] = ( p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4 ) >> 3;
                    pix[-2*xstride] = ( p2 + p1 + p0 + q0 + 2 ) >> 2;
                    pix[-3*xstride] = ( 2*p3 + 3*p2 + p1 + p0 + q0 + 4 ) >> 3;
                } else {
                    /* p0' */
                    pix[-1*xstride] = ( 2*p1 + p0 + q1 + 2 ) >> 2;
                }
                if( FFABS( q2 - q0 ) < beta)
                {
                    const int q3 = pix[3*xstride];
                    /* q0', q1', q2' */
                    pix[0*xstride] = ( p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4 ) >> 3;
                    pix[1*xstride] = ( p0 + q0 + q1 + q2 + 2 ) >> 2;
                    pix[2*xstride] = ( 2*q3 + 3*q2 + q1 + q0 + p0 + 4 ) >> 3;
                } else {
                    /* q0' */
                    pix[0*xstride] = ( 2*q1 + q0 + p1 + 2 ) >> 2;
                }
            }else{
                /* p0', q0' */
                pix[-1*xstride] = ( 2*p1 + p0 + q1 + 2 ) >> 2;
                pix[ 0*xstride] = ( 2*q1 + q0 + p1 + 2 ) >> 2;
            }
        }
        pix += ystride;
    }
}
static void h264_v_loop_filter_luma_intra_c(uint8_t *pix, int stride, int alpha, int beta)
{
    h264_loop_filter_luma_intra_c(pix, stride, 1, alpha, beta);
}
static void h264_h_loop_filter_luma_intra_c(uint8_t *pix, int stride, int alpha, int beta)
{
    h264_loop_filter_luma_intra_c(pix, 1, stride, alpha, beta);
}

static inline void h264_loop_filter_chroma_c(uint8_t *pix, int xstride, int ystride, int alpha, int beta, int8_t *tc0)
{
    int i, d;
    for( i = 0; i < 4; i++ ) {
        const int tc = tc0[i];
        if( tc <= 0 ) {
            pix += 2*ystride;
            continue;
        }
        for( d = 0; d < 2; d++ ) {
            const int p0 = pix[-1*xstride];
            const int p1 = pix[-2*xstride];
            const int q0 = pix[0];
            const int q1 = pix[1*xstride];

            if( FFABS( p0 - q0 ) < alpha &&
                FFABS( p1 - p0 ) < beta &&
                FFABS( q1 - q0 ) < beta ) {

                int delta = av_clip( (((q0 - p0 ) << 2) + (p1 - q1) + 4) >> 3, -tc, tc );

                pix[-xstride] = av_clip_uint8( p0 + delta );    /* p0' */
                pix[0]        = av_clip_uint8( q0 - delta );    /* q0' */
            }
            pix += ystride;
        }
    }
}
static void h264_v_loop_filter_chroma_c(uint8_t *pix, int stride, int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_chroma_c(pix, stride, 1, alpha, beta, tc0);
}
static void h264_h_loop_filter_chroma_c(uint8_t *pix, int stride, int alpha, int beta, int8_t *tc0)
{
    h264_loop_filter_chroma_c(pix, 1, stride, alpha, beta, tc0);
}

static inline void h264_loop_filter_chroma_intra_c(uint8_t *pix, int xstride, int ystride, int alpha, int beta)
{
    int d;
    for( d = 0; d < 8; d++ ) {
        const int p0 = pix[-1*xstride];
        const int p1 = pix[-2*xstride];
        const int q0 = pix[0];
        const int q1 = pix[1*xstride];

        if( FFABS( p0 - q0 ) < alpha &&
            FFABS( p1 - p0 ) < beta &&
            FFABS( q1 - q0 ) < beta ) {

            pix[-xstride] = ( 2*p1 + p0 + q1 + 2 ) >> 2;   /* p0' */
            pix[0]        = ( 2*q1 + q0 + p1 + 2 ) >> 2;   /* q0' */
        }
        pix += ystride;
    }
}
static void h264_v_loop_filter_chroma_intra_c(uint8_t *pix, int stride, int alpha, int beta)
{
    h264_loop_filter_chroma_intra_c(pix, stride, 1, alpha, beta);
}
static void h264_h_loop_filter_chroma_intra_c(uint8_t *pix, int stride, int alpha, int beta)
{
    h264_loop_filter_chroma_intra_c(pix, 1, stride, alpha, beta);
}


void dsputil_h264_init_cell(DSPContext_spu* c) {

	c->h264_v_loop_filter_luma= h264_v_loop_filter_luma_c;
    c->h264_h_loop_filter_luma= h264_h_loop_filter_luma_c;
    c->h264_v_loop_filter_luma_intra= h264_v_loop_filter_luma_intra_c;
    c->h264_h_loop_filter_luma_intra= h264_h_loop_filter_luma_intra_c;
    c->h264_v_loop_filter_chroma= h264_v_loop_filter_chroma_c;
    c->h264_h_loop_filter_chroma= h264_h_loop_filter_chroma_c;
    c->h264_v_loop_filter_chroma_intra= h264_v_loop_filter_chroma_intra_c;
    c->h264_h_loop_filter_chroma_intra= h264_h_loop_filter_chroma_intra_c;

    c->h264_idct_add[0] = h264_idct8_add_spu;
    c->h264_idct_add[1] = h264_idct4_add_spu;


    c->put_h264_chroma_pixels_tab[0] = put_h264_chroma_mc8_spu;
    c->put_h264_chroma_pixels_tab[1] = put_h264_chroma_mc4_spu;
    c->put_h264_chroma_pixels_tab[2] = put_h264_chroma_mc2_spu;
    c->avg_h264_chroma_pixels_tab[0] = avg_h264_chroma_mc8_spu;
    c->avg_h264_chroma_pixels_tab[1] = avg_h264_chroma_mc4_spu;
    c->avg_h264_chroma_pixels_tab[2] = avg_h264_chroma_mc2_spu;

    c->weight_h264_pixels_tab[0]= weight_h264_pixels16x16_c;
    c->weight_h264_pixels_tab[1]= weight_h264_pixels16x8_c;
    c->weight_h264_pixels_tab[2]= weight_h264_pixels8x16_c;
    c->weight_h264_pixels_tab[3]= weight_h264_pixels8x8_c;
    c->weight_h264_pixels_tab[4]= weight_h264_pixels8x4_c;
    c->weight_h264_pixels_tab[5]= weight_h264_pixels4x8_c;
    c->weight_h264_pixels_tab[6]= weight_h264_pixels4x4_c;
    c->weight_h264_pixels_tab[7]= weight_h264_pixels4x2_c;
    c->weight_h264_pixels_tab[8]= weight_h264_pixels2x4_c;
    c->weight_h264_pixels_tab[9]= weight_h264_pixels2x2_c;
    c->biweight_h264_pixels_tab[0]= biweight_h264_pixels16x16_c;
    c->biweight_h264_pixels_tab[1]= biweight_h264_pixels16x8_c;
    c->biweight_h264_pixels_tab[2]= biweight_h264_pixels8x16_c;
    c->biweight_h264_pixels_tab[3]= biweight_h264_pixels8x8_c;
    c->biweight_h264_pixels_tab[4]= biweight_h264_pixels8x4_c;
    c->biweight_h264_pixels_tab[5]= biweight_h264_pixels4x8_c;
    c->biweight_h264_pixels_tab[6]= biweight_h264_pixels4x4_c;
    c->biweight_h264_pixels_tab[7]= biweight_h264_pixels4x2_c;
    c->biweight_h264_pixels_tab[8]= biweight_h264_pixels2x4_c;
    c->biweight_h264_pixels_tab[9]= biweight_h264_pixels2x2_c;


#define dspfunc(PFX, IDX, NUM) \
    c->PFX ## _pixels_tab[IDX][ 0] = PFX ## NUM ## _mc00_spu; \
    c->PFX ## _pixels_tab[IDX][ 1] = PFX ## NUM ## _mc10_spu; \
    c->PFX ## _pixels_tab[IDX][ 2] = PFX ## NUM ## _mc20_spu; \
    c->PFX ## _pixels_tab[IDX][ 3] = PFX ## NUM ## _mc30_spu; \
    c->PFX ## _pixels_tab[IDX][ 4] = PFX ## NUM ## _mc01_spu; \
    c->PFX ## _pixels_tab[IDX][ 5] = PFX ## NUM ## _mc11_spu; \
    c->PFX ## _pixels_tab[IDX][ 6] = PFX ## NUM ## _mc21_spu; \
    c->PFX ## _pixels_tab[IDX][ 7] = PFX ## NUM ## _mc31_spu; \
    c->PFX ## _pixels_tab[IDX][ 8] = PFX ## NUM ## _mc02_spu; \
    c->PFX ## _pixels_tab[IDX][ 9] = PFX ## NUM ## _mc12_spu; \
    c->PFX ## _pixels_tab[IDX][10] = PFX ## NUM ## _mc22_spu; \
    c->PFX ## _pixels_tab[IDX][11] = PFX ## NUM ## _mc32_spu; \
    c->PFX ## _pixels_tab[IDX][12] = PFX ## NUM ## _mc03_spu; \
    c->PFX ## _pixels_tab[IDX][13] = PFX ## NUM ## _mc13_spu; \
    c->PFX ## _pixels_tab[IDX][14] = PFX ## NUM ## _mc23_spu; \
    c->PFX ## _pixels_tab[IDX][15] = PFX ## NUM ## _mc33_spu

    dspfunc(put_h264_qpel, 0, 16);
    dspfunc(put_h264_qpel, 1, 8);
    dspfunc(put_h264_qpel, 2, 4);

    dspfunc(avg_h264_qpel, 0, 16);
    dspfunc(avg_h264_qpel, 1, 8);
    dspfunc(avg_h264_qpel, 2, 4);

#undef dspfunc


}
