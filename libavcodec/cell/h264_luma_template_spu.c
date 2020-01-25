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

static void PREFIX_h264_qpel16_v_lowpass_spu(uint8_t * dst, uint8_t * src, int dstStride, int h) {

  register int i;

  const int16_t i20ss= 20;
  const int16_t i5ss= 5;
  const int16_t i16ss= 16;
  const int16_t imax = 255;

  const vsint32_t vzero = spu_splats(0);
  const vsint16_t v20ss = spu_splats(i20ss);
  const vsint16_t v5ss = spu_splats(i5ss);
  const vsint16_t v16ss = spu_splats(i16ss);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  vuint16_t sat;

  const int shift_src =(unsigned int) src & 15;
  const vuint8_t mergeh = {0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07};
  const vuint8_t mergel = {0x80,0x08,0x80,0x09,0x80,0x0A,0x80,0x0B,0x80,0x0C,0x80,0x0D,0x80,0x0E,0x80,0x0F};
  const vuint8_t packsu = {0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F};
  const vuint8_t mez = {0x02,0x03,0x12,0x13,0x06,0x07,0x16,0x17,0x0A,0x0B,0x1A,0x1B,0x0E,0x0F,0x1E,0x1F};

  uint8_t *srcbis = src - (STRIDE_Y * 2);

  const vuint8_t srcM2a = *(vuint8_t *)(srcbis);
  const vuint8_t srcM2b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcM2= spu_or(spu_slqwbyte(srcM2a, shift_src), spu_rlmaskqwbyte(srcM2b, shift_src-16));

  srcbis += STRIDE_Y;
  const vuint8_t srcM1a = *(vuint8_t *)(srcbis);
  const vuint8_t srcM1b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcM1= spu_or(spu_slqwbyte(srcM1a, shift_src), spu_rlmaskqwbyte(srcM1b, shift_src-16));

  srcbis += STRIDE_Y;
  const vuint8_t srcP0a = *(vuint8_t *)(srcbis);
  const vuint8_t srcP0b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcP0= spu_or(spu_slqwbyte(srcP0a, shift_src), spu_rlmaskqwbyte(srcP0b, shift_src-16));

  srcbis += STRIDE_Y;
  const vuint8_t srcP1a = *(vuint8_t *)(srcbis);
  const vuint8_t srcP1b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcP1= spu_or(spu_slqwbyte(srcP1a, shift_src), spu_rlmaskqwbyte(srcP1b, shift_src-16));

  srcbis += STRIDE_Y;
  const vuint8_t srcP2a = *(vuint8_t *)(srcbis);
  const vuint8_t srcP2b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcP2= spu_or(spu_slqwbyte(srcP2a, shift_src), spu_rlmaskqwbyte(srcP2b, shift_src-16));

  srcbis += STRIDE_Y;

  vsint16_t srcM2ssA = (vsint16_t)spu_shuffle(srcM2, srcM2, mergeh);
  vsint16_t srcM2ssB = (vsint16_t)spu_shuffle(srcM2, srcM2, mergel);
  vsint16_t srcM1ssA = (vsint16_t)spu_shuffle(srcM1, srcM1, mergeh);
  vsint16_t srcM1ssB = (vsint16_t)spu_shuffle(srcM1, srcM1, mergel);
  vsint16_t srcP0ssA = (vsint16_t)spu_shuffle(srcP0, srcP0, mergeh);
  vsint16_t srcP0ssB = (vsint16_t)spu_shuffle(srcP0, srcP0, mergel);
  vsint16_t srcP1ssA = (vsint16_t)spu_shuffle(srcP1, srcP1, mergeh);
  vsint16_t srcP1ssB = (vsint16_t)spu_shuffle(srcP1, srcP1, mergel);
  vsint16_t srcP2ssA = (vsint16_t)spu_shuffle(srcP2, srcP2, mergeh);
  vsint16_t srcP2ssB = (vsint16_t)spu_shuffle(srcP2, srcP2, mergel);

  for (i = 0 ; i < h ; i++) {
    const vuint8_t srcP3a = *(vuint8_t *)(srcbis);
    const vuint8_t srcP3b = *(vuint8_t *)(srcbis+16);
    const vuint8_t srcP3= spu_or(spu_slqwbyte(srcP3a, shift_src), spu_rlmaskqwbyte(srcP3b, shift_src-16));

    const vsint16_t srcP3ssA = (vsint16_t)spu_shuffle(srcP3, srcP3, mergeh);
    const vsint16_t srcP3ssB = (vsint16_t)spu_shuffle(srcP3, srcP3, mergel);
    srcbis += STRIDE_Y;

    const vsint16_t sum1A = spu_add(srcP0ssA, srcP1ssA);
    const vsint16_t sum1B = spu_add(srcP0ssB, srcP1ssB);
    const vsint16_t sum2A = spu_add(srcM1ssA, srcP2ssA);
    const vsint16_t sum2B = spu_add(srcM1ssB, srcP2ssB);
    const vsint16_t sum3A = spu_add(srcM2ssA, srcP3ssA);
    const vsint16_t sum3B = spu_add(srcM2ssB, srcP3ssB);

    srcM2ssA = srcM1ssA;
    srcM2ssB = srcM1ssB;
    srcM1ssA = srcP0ssA;
    srcM1ssB = srcP0ssB;
    srcP0ssA = srcP1ssA;
    srcP0ssB = srcP1ssB;
    srcP1ssA = srcP2ssA;
    srcP1ssB = srcP2ssB;
    srcP2ssA = srcP3ssA;
    srcP2ssB = srcP3ssB;

    const vsint32_t pp1A1 = spu_mule(sum1A, v20ss);
    const vsint32_t pp1A2 = spu_mulo(sum1A, v20ss);
    const vsint16_t pp1A3 = (vsint16_t)spu_shuffle((vsint16_t)pp1A1, (vsint16_t)pp1A2, mez);
    const vsint16_t pp1A = spu_add(pp1A3, v16ss);

    const vsint32_t pp1B1 = spu_mule(sum1B, v20ss);
    const vsint32_t pp1B2 = spu_mulo(sum1B, v20ss);
    const vsint16_t pp1B3 = (vsint16_t)spu_shuffle((vsint16_t)pp1B1, (vsint16_t)pp1B2, mez);
    const vsint16_t pp1B = spu_add(pp1B3, v16ss);

    const vsint32_t pp2A1 = spu_mule(sum2A, v5ss);
    const vsint32_t pp2A2 = spu_mulo(sum2A, v5ss);
    const vsint16_t pp2A = (vsint16_t)spu_shuffle((vsint16_t)pp2A1, (vsint16_t)pp2A2, mez);

    const vsint32_t pp2B1 = spu_mule(sum2B, v5ss);
    const vsint32_t pp2B2 = spu_mulo(sum2B, v5ss);
    const vsint16_t pp2B = (vsint16_t)spu_shuffle((vsint16_t)pp2B1, (vsint16_t)pp2B2, mez);

    const vsint16_t pp3A = spu_add(sum3A, pp1A);
    const vsint16_t pp3B = spu_add(sum3B, pp1B);

    const vsint16_t psumA = spu_sub(pp3A, pp2A);
    const vsint16_t psumB = spu_sub(pp3B, pp2B);

    vsint16_t sumA = spu_rlmask(psumA, -5);
    vsint16_t sumB = spu_rlmask(psumB, -5);

    //Saturation to 0 and 255
    sat = spu_cmpgt(sumA,(vsint16_t)vzero);
    sumA = spu_and(sumA,(vsint16_t)sat);
    sat = spu_cmpgt(sumA,vmax);
    sumA = spu_sel(sumA,vmax,sat);
    sat = spu_cmpgt(sumB,(vsint16_t)vzero);
    sumB = spu_and(sumB,(vsint16_t)sat);
    sat = spu_cmpgt(sumB,vmax);
    sumB = spu_sel(sumB,vmax,sat);

    const vuint8_t sum = (vuint8_t)spu_shuffle(sumA, sumB, packsu);

    /* 16x16 dest luma blocks are alway aligned */
    const vuint8_t vdst = *(vuint8_t *)dst;

    vuint8_t fsum;
    OP_U8_SPU(fsum, sum, vdst);

    *(vuint8_t *)dst=fsum;

    dst += dstStride; /* stride is  multiple of 16 ,so dstperm and dstmask can remain out of the loop */
  }
}

static void PREFIX_h264_qpel16_h_lowpass_spu(uint8_t * dst, uint8_t * src, int dstStride, int h) {

  register int i;

  const int16_t i20ss = 20;
  const int16_t i5ss = 5;
  const int16_t i16ss = 16;
  const int16_t imax = 255;

  const vsint32_t vzero = spu_splats(0);
  const vsint16_t v20ss = spu_splats(i20ss);
  const vsint16_t v5ss = spu_splats(i5ss);
  const vsint16_t v16ss = spu_splats(i16ss);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  vuint16_t sat;

  const vuint8_t mergeh = {0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07};
  const vuint8_t mergel = {0x80,0x08,0x80,0x09,0x80,0x0A,0x80,0x0B,0x80,0x0C,0x80,0x0D,0x80,0x0E,0x80,0x0F};
  const vuint8_t packsu = {0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F};
  const vuint8_t mez = {0x02,0x03,0x12,0x13,0x06,0x07,0x16,0x17,0x0A,0x0B,0x1A,0x1B,0x0E,0x0F,0x1E,0x1F};

  const int permM2 = (unsigned int) (src-2) & 15;
  const int permM1 = (unsigned int) (src-1) & 15;
  const int permP0 = (unsigned int) (src) & 15;
  const int permP1 = (unsigned int) (src+1) & 15;
  const int permP2 = (unsigned int) (src+2) & 15;
  const int permP3 = (unsigned int) (src+3) & 15;

  register int align = ((((unsigned long)src) - 2) % 16);

  for (i = 0 ; i < h ; i ++) {
    vuint8_t srcM2, srcM1, srcP0, srcP1, srcP2, srcP3;
    vuint8_t srcR1 = *(vuint8_t *)(src-2);
    vuint8_t srcR2 = *(vuint8_t *)(src+14);

    switch (align) {
    default: {
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR1, permP2), spu_rlmaskqwbyte(srcR2, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR1, permP3), spu_rlmaskqwbyte(srcR2, permP3-16));
    } break;
    case 11: {
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR1, permP2), spu_rlmaskqwbyte(srcR2, permP2-16));
      srcP3 = srcR2;
    } break;
    case 12: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = srcR2;
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 13: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = srcR2;
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 14: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = srcR2;
      srcP1 = spu_or(spu_slqwbyte(srcR2, permP1), spu_rlmaskqwbyte(srcR3, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 15: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = srcR2;
      srcP0 = spu_or(spu_slqwbyte(srcR2, permP0), spu_rlmaskqwbyte(srcR3, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR2, permP1), spu_rlmaskqwbyte(srcR3, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    }

    const vsint16_t srcP0A = (vsint16_t)spu_shuffle(srcP0, srcP0, mergeh);
    const vsint16_t srcP0B = (vsint16_t)spu_shuffle(srcP0, srcP0, mergel);
    const vsint16_t srcP1A = (vsint16_t)spu_shuffle(srcP1, srcP1, mergeh);
    const vsint16_t srcP1B = (vsint16_t)spu_shuffle(srcP1, srcP1, mergel);

    const vsint16_t srcP2A = (vsint16_t)spu_shuffle(srcP2, srcP2, mergeh);
    const vsint16_t srcP2B = (vsint16_t)spu_shuffle(srcP2, srcP2, mergel);
    const vsint16_t srcP3A = (vsint16_t)spu_shuffle(srcP3, srcP3, mergeh);
    const vsint16_t srcP3B = (vsint16_t)spu_shuffle(srcP3, srcP3, mergel);

    const vsint16_t srcM2A = (vsint16_t)spu_shuffle(srcM2, srcM2, mergeh);
    const vsint16_t srcM2B = (vsint16_t)spu_shuffle(srcM2, srcM2, mergel);
    const vsint16_t srcM1A = (vsint16_t)spu_shuffle(srcM1, srcM1, mergeh);
    const vsint16_t srcM1B = (vsint16_t)spu_shuffle(srcM1, srcM1, mergel);

    const vsint16_t sum1A = spu_add(srcP0A, srcP1A);
    const vsint16_t sum1B = spu_add(srcP0B, srcP1B);
    const vsint16_t sum2A = spu_add(srcM1A, srcP2A);
    const vsint16_t sum2B = spu_add(srcM1B, srcP2B);
    const vsint16_t sum3A = spu_add(srcM2A, srcP3A);
    const vsint16_t sum3B = spu_add(srcM2B, srcP3B);

    const vsint32_t pp1A1 = spu_mule(sum1A, v20ss);
    const vsint32_t pp1A2 = spu_mulo(sum1A, v20ss);
    const vsint16_t pp1A3 = (vsint16_t)spu_shuffle((vsint16_t)pp1A1, (vsint16_t)pp1A2, mez);
    const vsint16_t pp1A = spu_add(pp1A3, v16ss);

    const vsint32_t pp1B1 = spu_mule(sum1B, v20ss);
    const vsint32_t pp1B2 = spu_mulo(sum1B, v20ss);
    const vsint16_t pp1B3 = (vsint16_t)spu_shuffle((vsint16_t)pp1B1, (vsint16_t)pp1B2, mez);
    const vsint16_t pp1B = spu_add(pp1B3, v16ss);

    const vsint32_t pp2A1 = spu_mule(sum2A, v5ss);
    const vsint32_t pp2A2 = spu_mulo(sum2A, v5ss);
    const vsint16_t pp2A = (vsint16_t)spu_shuffle((vsint16_t)pp2A1, (vsint16_t)pp2A2, mez);

    const vsint32_t pp2B1 = spu_mule(sum2B, v5ss);
    const vsint32_t pp2B2 = spu_mulo(sum2B, v5ss);
    const vsint16_t pp2B = (vsint16_t)spu_shuffle((vsint16_t)pp2B1, (vsint16_t)pp2B2, mez);

    const vsint16_t pp3A = spu_add(sum3A, pp1A);
    const vsint16_t pp3B = spu_add(sum3B, pp1B);

    const vsint16_t psumA = spu_sub(pp3A, (vsint16_t)pp2A);
    const vsint16_t psumB = spu_sub(pp3B, (vsint16_t)pp2B);

    vsint16_t sumA = spu_rlmask(psumA, -5);
    vsint16_t sumB = spu_rlmask(psumB, -5);

    //Saturation to 0 and 255
    sat = spu_cmpgt(sumA,(vsint16_t)vzero);
    sumA = spu_and(sumA,(vsint16_t)sat);
    sat = spu_cmpgt(sumA,vmax);
    sumA = spu_sel(sumA,vmax,sat);
    sat = spu_cmpgt(sumB,(vsint16_t)vzero);
    sumB = spu_and(sumB,(vsint16_t)sat);
    sat = spu_cmpgt(sumB,vmax);
    sumB = spu_sel(sumB,vmax,sat);

    const vuint8_t sum = (vuint8_t)spu_shuffle(sumA, sumB, packsu);

    /* 16x16 dest luma blocks are alway aligned */
    const vuint8_t vdst = *(vuint8_t *)dst;

    vuint8_t fsum;
    OP_U8_SPU(fsum, sum, vdst);

    *(vuint8_t *)dst=fsum;

    src += STRIDE_Y;
    dst += dstStride; /* stride is multiple of 16 so dstperm and dstmask can remain out of the loop */
   }
}

/* this code assume stride % 16 == 0 *and* tmp is properly aligned */
static void PREFIX_h264_qpel16_hv_lowpass_spu(uint8_t * dst, int16_t * tmp, uint8_t * src, int dstStride, int tmpStride, int h) {
  register int i;

  const int16_t i20ss = 20;
  const int16_t i5ss = 5;
  const int16_t imax = 255;

  const vsint32_t vzero = spu_splats(0);
  const vsint16_t v20ss = spu_splats(i20ss);
  const vsint16_t v5ss = spu_splats(i5ss);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  vuint16_t sat;

  const vuint8_t mergeh = {0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07};
  const vuint8_t mergel = {0x80,0x08,0x80,0x09,0x80,0x0A,0x80,0x0B,0x80,0x0C,0x80,0x0D,0x80,0x0E,0x80,0x0F};
  const vuint8_t packsu = {0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F};
  const vuint8_t mez = {0x02,0x03,0x12,0x13,0x06,0x07,0x16,0x17,0x0A,0x0B,0x1A,0x1B,0x0E,0x0F,0x1E,0x1F};

  const int permM2 = (unsigned int) (src-2) & 15;
  const int permM1 = (unsigned int) (src-1) & 15;
  const int permP0 = (unsigned int) (src) & 15;
  const int permP1 = (unsigned int) (src+1) & 15;
  const int permP2 = (unsigned int) (src+2) & 15;
  const int permP3 = (unsigned int) (src+3) & 15;

  register int align = ((((unsigned long)src) - 2) % 16);

  src -= (2 * STRIDE_Y);

  for (i = 0 ; i < (h+5) ; i ++) {
    vuint8_t srcM2, srcM1, srcP0, srcP1, srcP2, srcP3;
    vuint8_t srcR1 = *(vuint8_t *)(src-2);
    vuint8_t srcR2 = *(vuint8_t *)(src+14);

    switch (align) {
    default: {
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR1, permP2), spu_rlmaskqwbyte(srcR2, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR1, permP3), spu_rlmaskqwbyte(srcR2, permP3-16));
    } break;
    case 11: {
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR1, permP2), spu_rlmaskqwbyte(srcR2, permP2-16));
      srcP3 = srcR2;
    } break;
    case 12: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = srcR2;
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 13: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = srcR2;
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 14: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = srcR2;
      srcP1 = spu_or(spu_slqwbyte(srcR2, permP1), spu_rlmaskqwbyte(srcR3, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 15: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = srcR2;
      srcP0 = spu_or(spu_slqwbyte(srcR2, permP0), spu_rlmaskqwbyte(srcR3, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR2, permP1), spu_rlmaskqwbyte(srcR3, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    }

    const vsint16_t srcP0A = (vsint16_t)spu_shuffle(srcP0, srcP0, mergeh);
    const vsint16_t srcP0B = (vsint16_t)spu_shuffle(srcP0, srcP0, mergel);
    const vsint16_t srcP1A = (vsint16_t)spu_shuffle(srcP1, srcP1, mergeh);
    const vsint16_t srcP1B = (vsint16_t)spu_shuffle(srcP1, srcP1, mergel);

    const vsint16_t srcP2A = (vsint16_t)spu_shuffle(srcP2, srcP2, mergeh);
    const vsint16_t srcP2B = (vsint16_t)spu_shuffle(srcP2, srcP2, mergel);
    const vsint16_t srcP3A = (vsint16_t)spu_shuffle(srcP3, srcP3, mergeh);
    const vsint16_t srcP3B = (vsint16_t)spu_shuffle(srcP3, srcP3, mergel);

    const vsint16_t srcM2A = (vsint16_t)spu_shuffle(srcM2, srcM2, mergeh);
    const vsint16_t srcM2B = (vsint16_t)spu_shuffle(srcM2, srcM2, mergel);
    const vsint16_t srcM1A = (vsint16_t)spu_shuffle(srcM1, srcM1, mergeh);
    const vsint16_t srcM1B = (vsint16_t)spu_shuffle(srcM1, srcM1, mergel);

    const vsint16_t sum1A = spu_add(srcP0A, srcP1A);
    const vsint16_t sum1B = spu_add(srcP0B, srcP1B);
    const vsint16_t sum2A = spu_add(srcM1A, srcP2A);
    const vsint16_t sum2B = spu_add(srcM1B, srcP2B);
    const vsint16_t sum3A = spu_add(srcM2A, srcP3A);
    const vsint16_t sum3B = spu_add(srcM2B, srcP3B);

    const vsint32_t pp1A1 = spu_mule(sum1A, v20ss);
    const vsint32_t pp1A2 = spu_mulo(sum1A, v20ss);
    const vsint16_t pp1A3 = (vsint16_t)spu_shuffle((vsint16_t)pp1A1, (vsint16_t)pp1A2, mez);
    const vsint16_t pp1A = spu_add(pp1A3, sum3A);

    const vsint32_t pp1B1 = spu_mule(sum1B, v20ss);
    const vsint32_t pp1B2 = spu_mulo(sum1B, v20ss);
    const vsint16_t pp1B3 = (vsint16_t)spu_shuffle((vsint16_t)pp1B1, (vsint16_t)pp1B2, mez);
    const vsint16_t pp1B = spu_add(pp1B3, sum3B);

    const vsint32_t pp2A1 = spu_mule(sum2A, v5ss);
    const vsint32_t pp2A2 = spu_mulo(sum2A, v5ss);
    const vsint16_t pp2A = (vsint16_t)spu_shuffle((vsint16_t)pp2A1, (vsint16_t)pp2A2, mez);

    const vsint32_t pp2B1 = spu_mule(sum2B, v5ss);
    const vsint32_t pp2B2 = spu_mulo(sum2B, v5ss);
    const vsint16_t pp2B = (vsint16_t)spu_shuffle((vsint16_t)pp2B1, (vsint16_t)pp2B2, mez);

    const vsint16_t psumA = spu_sub(pp1A, pp2A);
    const vsint16_t psumB = spu_sub(pp1B, pp2B);

    *(vsint16_t *)tmp = psumA;
    *(vsint16_t *)(tmp+8) = psumB;

    src += STRIDE_Y;
    tmp += tmpStride; /* int16_t*, and stride is 16, so it's OK here */
  }

  const int32_t ni10si = -10;
  const int16_t i1ss = 1;
  const int32_t i512si = 512;
  const int32_t ni16si = -16;

  const vsint32_t nv10si = spu_splats(ni10si);
  const vsint16_t v1ss = spu_splats(i1ss);
  const vsint32_t v512si = spu_splats(i512si);
  const vsint32_t nv16si = spu_splats(ni16si);

  const vuint8_t mperm = {0x00,0x08,0x01,0x09,0x02,0x0A,0x03,0x0B,0x04,0x0C,0x05,0x0D,0x06,0x0E,0x07,0x0F};
  const vuint8_t packs = {0x02,0x03,0x06,0x07,0x0A,0x0B,0x0E,0x0F,0x12,0x13,0x16,0x17,0x1A,0x1B,0x1E,0x1F};

  int16_t *tmpbis = tmp - (tmpStride * (h+5));

  vsint16_t tmpM2ssA = *(vsint16_t *)(tmpbis);
  vsint16_t tmpM2ssB = *(vsint16_t *)(tmpbis+8);
  tmpbis += tmpStride;
  vsint16_t tmpM1ssA = *(vsint16_t *)(tmpbis);
  vsint16_t tmpM1ssB = *(vsint16_t *)(tmpbis+8);
  tmpbis += tmpStride;
  vsint16_t tmpP0ssA = *(vsint16_t *)(tmpbis);
  vsint16_t tmpP0ssB = *(vsint16_t *)(tmpbis+8);
  tmpbis += tmpStride;
  vsint16_t tmpP1ssA = *(vsint16_t *)(tmpbis);
  vsint16_t tmpP1ssB = *(vsint16_t *)(tmpbis+8);
  tmpbis += tmpStride;
  vsint16_t tmpP2ssA = *(vsint16_t *)(tmpbis);
  vsint16_t tmpP2ssB = *(vsint16_t *)(tmpbis+8);
  tmpbis += tmpStride;

  for (i = 0 ; i < h ; i++) {
    const vsint16_t tmpP3ssA = *(vsint16_t *)(tmpbis);
    const vsint16_t tmpP3ssB = *(vsint16_t *)(tmpbis+8);
    tmpbis += tmpStride;

    const vsint16_t sum1A = spu_add(tmpP0ssA, tmpP1ssA);
    const vsint16_t sum1B = spu_add(tmpP0ssB, tmpP1ssB);
    const vsint16_t sum2A = spu_add(tmpM1ssA, tmpP2ssA);
    const vsint16_t sum2B = spu_add(tmpM1ssB, tmpP2ssB);
    const vsint16_t sum3A = spu_add(tmpM2ssA, tmpP3ssA);
    const vsint16_t sum3B = spu_add(tmpM2ssB, tmpP3ssB);

    tmpM2ssA = tmpM1ssA;
    tmpM2ssB = tmpM1ssB;
    tmpM1ssA = tmpP0ssA;
    tmpM1ssB = tmpP0ssB;
    tmpP0ssA = tmpP1ssA;
    tmpP0ssB = tmpP1ssB;
    tmpP1ssA = tmpP2ssA;
    tmpP1ssB = tmpP2ssB;
    tmpP2ssA = tmpP3ssA;
    tmpP2ssB = tmpP3ssB;

    const vsint32_t pp1Ae = spu_mule(sum1A, v20ss);
    const vsint32_t pp1Ao = spu_mulo(sum1A, v20ss);
    const vsint32_t pp1Be = spu_mule(sum1B, v20ss);
    const vsint32_t pp1Bo = spu_mulo(sum1B, v20ss);

    const vsint32_t pp2Ae = spu_mule(sum2A, v5ss);
    const vsint32_t pp2Ao = spu_mulo(sum2A, v5ss);
    const vsint32_t pp2Be = spu_mule(sum2B, v5ss);
    const vsint32_t pp2Bo = spu_mulo(sum2B, v5ss);

    const vsint32_t pp3Ae = spu_rlmask((vsint32_t)sum3A, nv16si);
    const vsint32_t pp3Ao = spu_mulo(sum3A, v1ss);
    const vsint32_t pp3Be = spu_rlmask((vsint32_t)sum3B, nv16si);
    const vsint32_t pp3Bo = spu_mulo(sum3B, v1ss);

    const vsint32_t pp1cAe = spu_add(pp1Ae, v512si);
    const vsint32_t pp1cAo = spu_add(pp1Ao, v512si);
    const vsint32_t pp1cBe = spu_add(pp1Be, v512si);
    const vsint32_t pp1cBo = spu_add(pp1Bo, v512si);

    const vsint32_t pp32Ae = spu_sub(pp3Ae, pp2Ae);
    const vsint32_t pp32Ao = spu_sub(pp3Ao, pp2Ao);
    const vsint32_t pp32Be = spu_sub(pp3Be, pp2Be);
    const vsint32_t pp32Bo = spu_sub(pp3Bo, pp2Bo);

    const vsint32_t sumAe = spu_add(pp1cAe, pp32Ae);
    const vsint32_t sumAo = spu_add(pp1cAo, pp32Ao);
    const vsint32_t sumBe = spu_add(pp1cBe, pp32Be);
    const vsint32_t sumBo = spu_add(pp1cBo, pp32Bo);

    const vsint32_t ssumAe = spu_rlmask(sumAe, nv10si);
    const vsint32_t ssumAo = spu_rlmask(sumAo, nv10si);
    const vsint32_t ssumBe = spu_rlmask(sumBe, nv10si);
    const vsint32_t ssumBo = spu_rlmask(sumBo, nv10si);

    vsint16_t ssume = (vsint16_t)spu_shuffle(ssumAe, ssumBe, packs);
    vsint16_t ssumo = (vsint16_t)spu_shuffle(ssumAo, ssumBo, packs);

    //Saturation to 0 and 255
    sat = spu_cmpgt(ssume,(vsint16_t)vzero);
    ssume = spu_and(ssume,(vsint16_t)sat);
    sat = spu_cmpgt(ssume,vmax);
    ssume = spu_sel(ssume,vmax,sat);
    sat = spu_cmpgt(ssumo,(vsint16_t)vzero);
    ssumo = spu_and(ssumo,(vsint16_t)sat);
    sat = spu_cmpgt(ssumo,vmax);
    ssumo = spu_sel(ssumo,vmax,sat);

    const vuint8_t sumv = (vuint8_t)spu_shuffle(ssume, ssumo, packsu);

    const vuint8_t sum = spu_shuffle(sumv, sumv, mperm);

    /* 16x16 dest luma blocks are alway aligned */
    const vuint8_t vdst = *(vuint8_t *)dst;

    vuint8_t fsum;
    OP_U8_SPU(fsum, sum, vdst);

    *(vuint8_t *)dst=fsum;

    dst += dstStride; /* stride is multiple of 16 so dstperm and dstmask can remain out of the loop */

  }
}

static void PREFIX_h264_qpel8_v_lowpass_spu(uint8_t * dst, uint8_t * src, int dstStride, int h) {

  register int i;

  const int16_t i20ss= 20;
  const int16_t i5ss= 5;
  const int16_t i16ss= 16;
  const int16_t imax = 255;

  const vsint32_t vzero = spu_splats(0);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  vuint16_t sat;

  const vsint16_t v20ss = spu_splats(i20ss);
  const vsint16_t v5ss = spu_splats(i5ss);
  const vsint16_t v16ss = spu_splats(i16ss);
  const int shift_src = (unsigned int) src & 15;

  const vuint8_t mergeh = {0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07};
  const vuint8_t packsu = {0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F};
  const vuint8_t mez = {0x02,0x03,0x12,0x13,0x06,0x07,0x16,0x17,0x0A,0x0B,0x1A,0x1B,0x0E,0x0F,0x1E,0x1F};

  /* 8x8 dest luma blocks are aligned or desaligned by 8*/
  const int shift_dst = (unsigned int) dst & 15;
  vuint8_t dstmask;
  const vuint8_t dst8mask1= {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst8mask2= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17};

  if(shift_dst==0){
    dstmask = dst8mask1;
  }
  else{
    dstmask = dst8mask2;
  }

  uint8_t *srcbis = src - (STRIDE_Y * 2);

  const vuint8_t srcM2a = *(vuint8_t *)(srcbis);
  const vuint8_t srcM2b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcM2= spu_or(spu_slqwbyte(srcM2a, shift_src), spu_rlmaskqwbyte(srcM2b, shift_src-16));

  srcbis += STRIDE_Y;
  const vuint8_t srcM1a = *(vuint8_t *)(srcbis);
  const vuint8_t srcM1b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcM1= spu_or(spu_slqwbyte(srcM1a, shift_src), spu_rlmaskqwbyte(srcM1b, shift_src-16));

  srcbis += STRIDE_Y;
  const vuint8_t srcP0a = *(vuint8_t *)(srcbis);
  const vuint8_t srcP0b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcP0= spu_or(spu_slqwbyte(srcP0a, shift_src), spu_rlmaskqwbyte(srcP0b, shift_src-16));

  srcbis += STRIDE_Y;
  const vuint8_t srcP1a = *(vuint8_t *)(srcbis);
  const vuint8_t srcP1b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcP1= spu_or(spu_slqwbyte(srcP1a, shift_src), spu_rlmaskqwbyte(srcP1b, shift_src-16));

  srcbis += STRIDE_Y;
  const vuint8_t srcP2a = *(vuint8_t *)(srcbis);
  const vuint8_t srcP2b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcP2= spu_or(spu_slqwbyte(srcP2a, shift_src), spu_rlmaskqwbyte(srcP2b, shift_src-16));

  srcbis += STRIDE_Y;

  vsint16_t srcM2ssA = (vsint16_t)spu_shuffle(srcM2, srcM2, mergeh);
  vsint16_t srcM1ssA = (vsint16_t)spu_shuffle(srcM1, srcM1, mergeh);
  vsint16_t srcP0ssA = (vsint16_t)spu_shuffle(srcP0, srcP0, mergeh);
  vsint16_t srcP1ssA = (vsint16_t)spu_shuffle(srcP1, srcP1, mergeh);
  vsint16_t srcP2ssA = (vsint16_t)spu_shuffle(srcP2, srcP2, mergeh);

  for (i = 0 ; i < h ; i++) {
    const vuint8_t srcP3a = *(vuint8_t *)(srcbis);
    const vuint8_t srcP3b = *(vuint8_t *)(srcbis+16);
    const vuint8_t srcP3= spu_or(spu_slqwbyte(srcP3a, shift_src), spu_rlmaskqwbyte(srcP3b, shift_src-16));

    const vsint16_t srcP3ssA = (vsint16_t)spu_shuffle(srcP3, srcP3, mergeh);
    srcbis += STRIDE_Y;

    const vsint16_t sum1A = spu_add(srcP0ssA, srcP1ssA);
    const vsint16_t sum2A = spu_add(srcM1ssA, srcP2ssA);
    const vsint16_t sum3A = spu_add(srcM2ssA, srcP3ssA);

    srcM2ssA = srcM1ssA;
    srcM1ssA = srcP0ssA;
    srcP0ssA = srcP1ssA;
    srcP1ssA = srcP2ssA;
    srcP2ssA = srcP3ssA;

    const vsint32_t pp1A1 = spu_mule(sum1A, v20ss);
    const vsint32_t pp1A2 = spu_mulo(sum1A, v20ss);
    const vsint16_t pp1A3 = (vsint16_t)spu_shuffle((vsint16_t)pp1A1, (vsint16_t)pp1A2, mez);
    const vsint16_t pp1A = spu_add(pp1A3, v16ss);

    const vsint32_t pp2A1 = spu_mule(sum2A, v5ss);
    const vsint32_t pp2A2 = spu_mulo(sum2A, v5ss);
    const vsint16_t pp2A = (vsint16_t)spu_shuffle((vsint16_t)pp2A1, (vsint16_t)pp2A2, mez);

    const vsint16_t pp3A = spu_add(sum3A, pp1A);
    const vsint16_t psumA = spu_sub(pp3A, pp2A);
    vsint16_t sumA = spu_rlmask(psumA, -5);

    //Saturation to 0 and 255
    sat = spu_cmpgt(sumA,(vsint16_t)vzero);
    sumA = spu_and(sumA,(vsint16_t)sat);
    sat = spu_cmpgt(sumA,vmax);
    sumA = spu_sel(sumA,vmax,sat);

    const vuint8_t sum = (vuint8_t)spu_shuffle(sumA, (vsint16_t)vzero, packsu);

    const vuint8_t dst1 = *(vuint8_t *)dst;

    const vuint8_t dsum = spu_shuffle(dst1, sum, dstmask);
    vuint8_t fsum;
    OP_U8_SPU(fsum, dsum, dst1);

    *(vuint8_t *)dst=fsum;

    dst += dstStride;
  }
}

static void PREFIX_h264_qpel8_h_lowpass_spu(uint8_t * dst, uint8_t * src, int dstStride, int h) {

  register int i;

  const int16_t i20ss = 20;
  const int16_t i5ss = 5;
  const int16_t i16ss = 16;
  const int16_t imax = 255;

  const vsint32_t vzero = spu_splats(0);
  const vsint16_t v20ss = spu_splats(i20ss);
  const vsint16_t v5ss = spu_splats(i5ss);
  const vsint16_t v16ss = spu_splats(i16ss);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  vuint16_t sat;

  const vuint8_t mergeh = {0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07};
  const vuint8_t packsu = {0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F};
  const vuint8_t mez = {0x02,0x03,0x12,0x13,0x06,0x07,0x16,0x17,0x0A,0x0B,0x1A,0x1B,0x0E,0x0F,0x1E,0x1F};

  /* 8x8 dest luma blocks are aligned or desaligned by 8*/
  const int shift_dst = (unsigned int) dst & 15;
  vuint8_t dstmask;
  const vuint8_t dst8mask1= {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst8mask2= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17};

  if(shift_dst==0){
    dstmask = dst8mask1;
  }
  else{
    dstmask = dst8mask2;
  }

  const int permM2 = (unsigned int) (src-2) & 15;
  const int permM1 = (unsigned int) (src-1) & 15;
  const int permP0 = (unsigned int) (src) & 15;
  const int permP1 = (unsigned int) (src+1) & 15;
  const int permP2 = (unsigned int) (src+2) & 15;
  const int permP3 = (unsigned int) (src+3) & 15;

  register int align = ((((unsigned long)src) - 2) % 16);

  for (i = 0 ; i < h ; i ++) {
    vuint8_t srcM2, srcM1, srcP0, srcP1, srcP2, srcP3;
    vuint8_t srcR1 = *(vuint8_t *)(src-2);
    vuint8_t srcR2 = *(vuint8_t *)(src+14);

    switch (align) {
    default: {
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR1, permP2), spu_rlmaskqwbyte(srcR2, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR1, permP3), spu_rlmaskqwbyte(srcR2, permP3-16));
    } break;
    case 11: {
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR1, permP2), spu_rlmaskqwbyte(srcR2, permP2-16));
      srcP3 = srcR2;
    } break;
    case 12: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = srcR2;
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 13: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = srcR2;
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 14: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = srcR2;
      srcP1 = spu_or(spu_slqwbyte(srcR2, permP1), spu_rlmaskqwbyte(srcR3, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 15: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = srcR2;
      srcP0 = spu_or(spu_slqwbyte(srcR2, permP0), spu_rlmaskqwbyte(srcR3, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR2, permP1), spu_rlmaskqwbyte(srcR3, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    }

    const vsint16_t srcP0A = (vsint16_t)spu_shuffle(srcP0, srcP0, mergeh);
    const vsint16_t srcP1A = (vsint16_t)spu_shuffle(srcP1, srcP1, mergeh);

    const vsint16_t srcP2A = (vsint16_t)spu_shuffle(srcP2, srcP2, mergeh);
    const vsint16_t srcP3A = (vsint16_t)spu_shuffle(srcP3, srcP3, mergeh);

    const vsint16_t srcM2A = (vsint16_t)spu_shuffle(srcM2, srcM2, mergeh);
    const vsint16_t srcM1A = (vsint16_t)spu_shuffle(srcM1, srcM1, mergeh);

    const vsint16_t sum1A = spu_add(srcP0A, srcP1A);
    const vsint16_t sum2A = spu_add(srcM1A, srcP2A);
    const vsint16_t sum3A = spu_add(srcM2A, srcP3A);

    const vsint32_t pp1A1 = spu_mule(sum1A, v20ss);
    const vsint32_t pp1A2 = spu_mulo(sum1A, v20ss);
    const vsint16_t pp1A3 = (vsint16_t)spu_shuffle((vsint16_t)pp1A1, (vsint16_t)pp1A2, mez);
    const vsint16_t pp1A = spu_add(pp1A3, v16ss);

    const vsint32_t pp2A1 = spu_mule(sum2A, v5ss);
    const vsint32_t pp2A2 = spu_mulo(sum2A, v5ss);
    const vsint16_t pp2A = (vsint16_t)spu_shuffle((vsint16_t)pp2A1, (vsint16_t)pp2A2, mez);

    const vsint16_t pp3A = spu_add(sum3A, pp1A);

    const vsint16_t psumA = spu_sub(pp3A, (vsint16_t)pp2A);

    vsint16_t sumA = spu_rlmask(psumA, -5);

    //Saturation to 0 and 255
    sat = spu_cmpgt(sumA,(vsint16_t)vzero);
    sumA = spu_and(sumA,(vsint16_t)sat);
    sat = spu_cmpgt(sumA,vmax);
    sumA = spu_sel(sumA,vmax,sat);

    const vuint8_t sum = (vuint8_t)spu_shuffle(sumA, (vsint16_t)vzero, packsu);

    const vuint8_t dst1 = *(vuint8_t *)dst;

    const vuint8_t dsum = spu_shuffle(dst1, sum, dstmask);
    vuint8_t fsum;
    OP_U8_SPU(fsum, dsum, dst1);

    *(vuint8_t *)dst=fsum;

    src += STRIDE_Y;
    dst += dstStride; /* stride is multiple of 16 so dstperm and dstmask can remain out of the loop */
   }
}

/* this code assume stride % 16 == 0 *and* tmp is properly aligned */
static void PREFIX_h264_qpel8_hv_lowpass_spu(uint8_t * dst, int16_t * tmp, uint8_t * src, int dstStride, int tmpStride, int h) {
  register int i;

  const int16_t i20ss = 20;
  const int16_t i5ss = 5;
  const int16_t imax = 255;

  const vsint32_t vzero = spu_splats(0);
  const vsint16_t v20ss = spu_splats(i20ss);
  const vsint16_t v5ss = spu_splats(i5ss);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  vuint16_t sat;

  const vuint8_t mergeh = {0x10,0x00,0x11,0x01,0x12,0x02,0x13,0x03,0x14,0x04,0x15,0x05,0x16,0x06,0x17,0x07};
  const vuint8_t packsu = {0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F};
  const vuint8_t mez = {0x02,0x03,0x12,0x13,0x06,0x07,0x16,0x17,0x0A,0x0B,0x1A,0x1B,0x0E,0x0F,0x1E,0x1F};

  const int permM2 = (unsigned int) (src-2) & 15;
  const int permM1 = (unsigned int) (src-1) & 15;
  const int permP0 = (unsigned int) (src) & 15;
  const int permP1 = (unsigned int) (src+1) & 15;
  const int permP2 = (unsigned int) (src+2) & 15;
  const int permP3 = (unsigned int) (src+3) & 15;

  register int align = ((((unsigned long)src) - 2) % 16);

  src -= (2 * STRIDE_Y);

  for (i = 0 ; i < (h+5) ; i ++) {
    vuint8_t srcM2, srcM1, srcP0, srcP1, srcP2, srcP3;
    vuint8_t srcR1 = *(vuint8_t *)(src-2);
    vuint8_t srcR2 = *(vuint8_t *)(src+14);

    switch (align) {
    default: {
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR1, permP2), spu_rlmaskqwbyte(srcR2, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR1, permP3), spu_rlmaskqwbyte(srcR2, permP3-16));
    } break;
    case 11: {
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR1, permP2), spu_rlmaskqwbyte(srcR2, permP2-16));
      srcP3 = srcR2;
    } break;
    case 12: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = srcR2;
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 13: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = srcR2;
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 14: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = srcR2;
      srcP1 = spu_or(spu_slqwbyte(srcR2, permP1), spu_rlmaskqwbyte(srcR3, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 15: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = srcR2;
      srcP0 = spu_or(spu_slqwbyte(srcR2, permP0), spu_rlmaskqwbyte(srcR3, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR2, permP1), spu_rlmaskqwbyte(srcR3, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    }

    const vsint16_t srcP0A = (vsint16_t)spu_shuffle(srcP0, (vuint8_t)vzero, mergeh);
    const vsint16_t srcP1A = (vsint16_t)spu_shuffle(srcP1, (vuint8_t)vzero, mergeh);
    const vsint16_t srcP2A = (vsint16_t)spu_shuffle(srcP2, (vuint8_t)vzero, mergeh);
    const vsint16_t srcP3A = (vsint16_t)spu_shuffle(srcP3, (vuint8_t)vzero, mergeh);
    const vsint16_t srcM2A = (vsint16_t)spu_shuffle(srcM2, (vuint8_t)vzero, mergeh);
    const vsint16_t srcM1A = (vsint16_t)spu_shuffle(srcM1, (vuint8_t)vzero, mergeh);

    const vsint16_t sum1A = spu_add(srcP0A, srcP1A);
    const vsint16_t sum2A = spu_add(srcM1A, srcP2A);
    const vsint16_t sum3A = spu_add(srcM2A, srcP3A);

    const vsint32_t pp1A1 = spu_mule(sum1A, v20ss);
    const vsint32_t pp1A2 = spu_mulo(sum1A, v20ss);
    const vsint16_t pp1A3 = (vsint16_t)spu_shuffle((vsint16_t)pp1A1, (vsint16_t)pp1A2, mez);
    const vsint16_t pp1A = spu_add(pp1A3, sum3A);

    const vsint32_t pp2A1 = spu_mule(sum2A, v5ss);
    const vsint32_t pp2A2 = spu_mulo(sum2A, v5ss);
    const vsint16_t pp2A = (vsint16_t)spu_shuffle((vsint16_t)pp2A1, (vsint16_t)pp2A2, mez);

    const vsint16_t psumA = spu_sub(pp1A, pp2A);

    *(vsint16_t *)tmp = psumA;

    src += STRIDE_Y;
    tmp += tmpStride; /* int16_t*, and stride is 16, so it's OK here */
  }

  const int32_t ni10si = -10;
  const int16_t i1ss = 1;
  const int32_t i512si = 512;
  const int32_t ni16si = -16;

  const vsint32_t nv10si = spu_splats(ni10si);
  const vsint16_t v1ss = spu_splats(i1ss);
  const vsint32_t v512si = spu_splats(i512si);
  const vsint32_t nv16si = spu_splats(ni16si);

  const vuint8_t mperm = {0x00,0x08,0x01,0x09,0x02,0x0A,0x03,0x0B,0x04,0x0C,0x05,0x0D,0x06,0x0E,0x07,0x0F};
  const vuint8_t packs = {0x02,0x03,0x06,0x07,0x0A,0x0B,0x0E,0x0F,0x12,0x13,0x16,0x17,0x1A,0x1B,0x1E,0x1F};

  const int shift_dst = (unsigned int) (dst) & 15;
  /* 8x8 dest luma blocks are aligned or desaligned by 8*/
  vuint8_t dstmask;
  const vuint8_t dst8mask1= {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst8mask2= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17};

  if(shift_dst==0){
    dstmask = dst8mask1;
  }
  else{
    dstmask = dst8mask2;
  }

  int16_t *tmpbis = tmp - (tmpStride * (h+5));

  vsint16_t tmpM2ssA = *(vsint16_t *)(tmpbis);
  tmpbis += tmpStride;
  vsint16_t tmpM1ssA = *(vsint16_t *)(tmpbis);
  tmpbis += tmpStride;
  vsint16_t tmpP0ssA = *(vsint16_t *)(tmpbis);
  tmpbis += tmpStride;
  vsint16_t tmpP1ssA = *(vsint16_t *)(tmpbis);
  tmpbis += tmpStride;
  vsint16_t tmpP2ssA = *(vsint16_t *)(tmpbis);
  tmpbis += tmpStride;

  for (i = 0 ; i < h ; i++) {
    const vsint16_t tmpP3ssA = *(vsint16_t *)(tmpbis);
    tmpbis += tmpStride;

    const vsint16_t sum1A = spu_add(tmpP0ssA, tmpP1ssA);
    const vsint16_t sum2A = spu_add(tmpM1ssA, tmpP2ssA);
    const vsint16_t sum3A = spu_add(tmpM2ssA, tmpP3ssA);

    tmpM2ssA = tmpM1ssA;
    tmpM1ssA = tmpP0ssA;
    tmpP0ssA = tmpP1ssA;
    tmpP1ssA = tmpP2ssA;
    tmpP2ssA = tmpP3ssA;

    const vsint32_t pp1Ae = spu_mule(sum1A, v20ss);
    const vsint32_t pp1Ao = spu_mulo(sum1A, v20ss);
    const vsint32_t pp2Ae = spu_mule(sum2A, v5ss);
    const vsint32_t pp2Ao = spu_mulo(sum2A, v5ss);

    const vsint32_t pp3Ae = spu_rlmask((vsint32_t)sum3A, nv16si);
    const vsint32_t pp3Ao = spu_mulo(sum3A, v1ss);

    const vsint32_t pp1cAe = spu_add(pp1Ae, v512si);
    const vsint32_t pp1cAo = spu_add(pp1Ao, v512si);

    const vsint32_t pp32Ae = spu_sub(pp3Ae, pp2Ae);
    const vsint32_t pp32Ao = spu_sub(pp3Ao, pp2Ao);

    const vsint32_t sumAe = spu_add(pp1cAe, pp32Ae);
    const vsint32_t sumAo = spu_add(pp1cAo, pp32Ao);

    const vsint32_t ssumAe = spu_rlmask(sumAe, nv10si);
    const vsint32_t ssumAo = spu_rlmask(sumAo, nv10si);

    vsint16_t ssume = (vsint16_t)spu_shuffle(ssumAe, vzero, packs);
    vsint16_t ssumo = (vsint16_t)spu_shuffle(ssumAo, vzero, packs);

    //Saturation to 0 and 255
    sat = spu_cmpgt(ssume,(vsint16_t)vzero);
    ssume = spu_and(ssume,(vsint16_t)sat);
    sat = spu_cmpgt(ssume,vmax);
    ssume = spu_sel(ssume,vmax,sat);
    sat = spu_cmpgt(ssumo,(vsint16_t)vzero);
    ssumo = spu_and(ssumo,(vsint16_t)sat);
    sat = spu_cmpgt(ssumo,vmax);
    ssumo = spu_sel(ssumo,vmax,sat);

    const vuint8_t sumv = (vuint8_t)spu_shuffle(ssume, ssumo, packsu);

    const vuint8_t sum = spu_shuffle(sumv, sumv, mperm);

    const vuint8_t dst1 = *(vuint8_t *)dst;

    const vuint8_t dsum = spu_shuffle(dst1, sum, dstmask);
    vuint8_t fsum;
    OP_U8_SPU(fsum, dsum, dst1);

    *(vuint8_t *)dst=fsum;

    dst += dstStride; /* stride is multiple of 16 so dstperm and dstmask can remain out of the loop */

  }
}

static void PREFIX_h264_qpel4_v_lowpass_spu(uint8_t * dst, uint8_t * src, int dstStride, int h) {

  register int i;

  const int16_t i20ss= 20;
  const int16_t i5ss= 5;
  const int16_t i16ss= 16;
  const int16_t imax = 255;

  const vsint32_t vzero = spu_splats(0);
  const vsint16_t v20ss = spu_splats(i20ss);
  const vsint16_t v5ss = spu_splats(i5ss);
  const vsint16_t v16ss = spu_splats(i16ss);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  vuint16_t sat;

  const int shift_src = (unsigned int) src & 15;

  const vuint8_t mergeh = {0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07};
  const vuint8_t packsu = {0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F};
  const vuint8_t mez = {0x02,0x03,0x12,0x13,0x06,0x07,0x16,0x17,0x0A,0x0B,0x1A,0x1B,0x0E,0x0F,0x1E,0x1F};

  /* 4x4 dest luma blocks are aligned or desaligned by 4,8 or 12*/
  const int shift_dst = (unsigned int) dst & 15;
  vuint8_t dstmask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vuint8_t dst4mask0= {0x10,0x11,0x12,0x13,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst4mask4= {0x00,0x01,0x02,0x03,0x10,0x11,0x12,0x13,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst4mask8= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst4mask12= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x10,0x11,0x12,0x13};

  switch(shift_dst){
    case 0:  dstmask = dst4mask0;
             break;
    case 4:  dstmask = dst4mask4;
             break;
    case 8:  dstmask = dst4mask8;
             break;
    case 12: dstmask = dst4mask12;
             break;
  }

  uint8_t *srcbis = src - (STRIDE_Y * 2);

  const vuint8_t srcM2a = *(vuint8_t *)(srcbis);
  const vuint8_t srcM2b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcM2= spu_or(spu_slqwbyte(srcM2a, shift_src), spu_rlmaskqwbyte(srcM2b, shift_src-16));

  srcbis += STRIDE_Y;
  const vuint8_t srcM1a = *(vuint8_t *)(srcbis);
  const vuint8_t srcM1b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcM1= spu_or(spu_slqwbyte(srcM1a, shift_src), spu_rlmaskqwbyte(srcM1b, shift_src-16));

  srcbis += STRIDE_Y;
  const vuint8_t srcP0a = *(vuint8_t *)(srcbis);
  const vuint8_t srcP0b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcP0= spu_or(spu_slqwbyte(srcP0a, shift_src), spu_rlmaskqwbyte(srcP0b, shift_src-16));

  srcbis += STRIDE_Y;
  const vuint8_t srcP1a = *(vuint8_t *)(srcbis);
  const vuint8_t srcP1b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcP1= spu_or(spu_slqwbyte(srcP1a, shift_src), spu_rlmaskqwbyte(srcP1b, shift_src-16));

  srcbis += STRIDE_Y;
  const vuint8_t srcP2a = *(vuint8_t *)(srcbis);
  const vuint8_t srcP2b = *(vuint8_t *)(srcbis+16);
  const vuint8_t srcP2= spu_or(spu_slqwbyte(srcP2a, shift_src), spu_rlmaskqwbyte(srcP2b, shift_src-16));

  srcbis += STRIDE_Y;

  vsint16_t srcM2ssA = (vsint16_t)spu_shuffle(srcM2, srcM2, mergeh);
  vsint16_t srcM1ssA = (vsint16_t)spu_shuffle(srcM1, srcM1, mergeh);
  vsint16_t srcP0ssA = (vsint16_t)spu_shuffle(srcP0, srcP0, mergeh);
  vsint16_t srcP1ssA = (vsint16_t)spu_shuffle(srcP1, srcP1, mergeh);
  vsint16_t srcP2ssA = (vsint16_t)spu_shuffle(srcP2, srcP2, mergeh);

  for (i = 0 ; i < h ; i++) {
    const vuint8_t srcP3a = *(vuint8_t *)(srcbis);
    const vuint8_t srcP3b = *(vuint8_t *)(srcbis+16);
    const vuint8_t srcP3= spu_or(spu_slqwbyte(srcP3a, shift_src), spu_rlmaskqwbyte(srcP3b, shift_src-16));

    const vsint16_t srcP3ssA = (vsint16_t)spu_shuffle(srcP3, srcP3, mergeh);
    srcbis += STRIDE_Y;

    const vsint16_t sum1A = spu_add(srcP0ssA, srcP1ssA);
    const vsint16_t sum2A = spu_add(srcM1ssA, srcP2ssA);
    const vsint16_t sum3A = spu_add(srcM2ssA, srcP3ssA);

    srcM2ssA = srcM1ssA;
    srcM1ssA = srcP0ssA;
    srcP0ssA = srcP1ssA;
    srcP1ssA = srcP2ssA;
    srcP2ssA = srcP3ssA;

    const vsint32_t pp1A1 = spu_mule(sum1A, v20ss);
    const vsint32_t pp1A2 = spu_mulo(sum1A, v20ss);
    const vsint16_t pp1A3 = (vsint16_t)spu_shuffle((vsint16_t)pp1A1, (vsint16_t)pp1A2, mez);
    const vsint16_t pp1A = spu_add(pp1A3, v16ss);

    const vsint32_t pp2A1 = spu_mule(sum2A, v5ss);
    const vsint32_t pp2A2 = spu_mulo(sum2A, v5ss);
    const vsint16_t pp2A = (vsint16_t)spu_shuffle((vsint16_t)pp2A1, (vsint16_t)pp2A2, mez);

    const vsint16_t pp3A = spu_add(sum3A, pp1A);
    const vsint16_t psumA = spu_sub(pp3A, pp2A);
    vsint16_t sumA = spu_rlmask(psumA, -5);

    //Saturation to 0 and 255
    sat = spu_cmpgt(sumA,(vsint16_t)vzero);
    sumA = spu_and(sumA,(vsint16_t)sat);
    sat = spu_cmpgt(sumA,vmax);
    sumA = spu_sel(sumA,vmax,sat);

    const vuint8_t sum = (vuint8_t)spu_shuffle(sumA, (vsint16_t)vzero, packsu);

    const vuint8_t dst1 = *(vuint8_t *)dst;

    const vuint8_t dsum = spu_shuffle(dst1, sum, dstmask);
    vuint8_t fsum;
    OP_U8_SPU(fsum, dsum, dst1);

    *(vuint8_t *)dst=fsum;

    dst += dstStride;
  }
}

static void PREFIX_h264_qpel4_h_lowpass_spu(uint8_t * dst, uint8_t * src, int dstStride, int h) {

  register int i;

  const int16_t i20ss = 20;
  const int16_t i5ss = 5;
  const int16_t i16ss = 16;
  const int16_t imax = 255;

  const vsint32_t vzero = spu_splats(0);
  const vsint16_t v20ss = spu_splats(i20ss);
  const vsint16_t v5ss = spu_splats(i5ss);
  const vsint16_t v16ss = spu_splats(i16ss);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  vuint16_t sat;

  const vuint8_t mergeh = {0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07};
  const vuint8_t packsu = {0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F};
  const vuint8_t mez = {0x02,0x03,0x12,0x13,0x06,0x07,0x16,0x17,0x0A,0x0B,0x1A,0x1B,0x0E,0x0F,0x1E,0x1F};

  /* 4x4 dest luma blocks are aligned or desaligned by 4,8 or 12*/
  const int shift_dst = (unsigned int) dst & 15;
  vuint8_t dstmask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vuint8_t dst4mask0= {0x10,0x11,0x12,0x13,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst4mask4= {0x00,0x01,0x02,0x03,0x10,0x11,0x12,0x13,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst4mask8= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst4mask12= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x10,0x11,0x12,0x13};

  switch(shift_dst){
    case 0:  dstmask = dst4mask0;
             break;
    case 4:  dstmask = dst4mask4;
             break;
    case 8:  dstmask = dst4mask8;
             break;
    case 12: dstmask = dst4mask12;
             break;
  }

  const int permM2 = (unsigned int) (src-2) & 15;
  const int permM1 = (unsigned int) (src-1) & 15;
  const int permP0 = (unsigned int) (src) & 15;
  const int permP1 = (unsigned int) (src+1) & 15;
  const int permP2 = (unsigned int) (src+2) & 15;
  const int permP3 = (unsigned int) (src+3) & 15;

  register int align = ((((unsigned long)src) - 2) % 16);

  for (i = 0 ; i < h ; i ++) {
    vuint8_t srcM2, srcM1, srcP0, srcP1, srcP2, srcP3;
    vuint8_t srcR1 = *(vuint8_t *)(src-2);
    vuint8_t srcR2 = *(vuint8_t *)(src+14);

    switch (align) {
    default: {
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR1, permP2), spu_rlmaskqwbyte(srcR2, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR1, permP3), spu_rlmaskqwbyte(srcR2, permP3-16));
    } break;
    case 11: {
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR1, permP2), spu_rlmaskqwbyte(srcR2, permP2-16));
      srcP3 = srcR2;
    } break;
    case 12: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = srcR2;
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 13: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = srcR2;
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 14: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = srcR2;
      srcP1 = spu_or(spu_slqwbyte(srcR2, permP1), spu_rlmaskqwbyte(srcR3, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 15: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = srcR2;
      srcP0 = spu_or(spu_slqwbyte(srcR2, permP0), spu_rlmaskqwbyte(srcR3, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR2, permP1), spu_rlmaskqwbyte(srcR3, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    }

    const vsint16_t srcP0A = (vsint16_t)spu_shuffle(srcP0, srcP0, mergeh);
    const vsint16_t srcP1A = (vsint16_t)spu_shuffle(srcP1, srcP1, mergeh);

    const vsint16_t srcP2A = (vsint16_t)spu_shuffle(srcP2, srcP2, mergeh);
    const vsint16_t srcP3A = (vsint16_t)spu_shuffle(srcP3, srcP3, mergeh);

    const vsint16_t srcM2A = (vsint16_t)spu_shuffle(srcM2, srcM2, mergeh);
    const vsint16_t srcM1A = (vsint16_t)spu_shuffle(srcM1, srcM1, mergeh);

    const vsint16_t sum1A = spu_add(srcP0A, srcP1A);
    const vsint16_t sum2A = spu_add(srcM1A, srcP2A);
    const vsint16_t sum3A = spu_add(srcM2A, srcP3A);

    const vsint32_t pp1A1 = spu_mule(sum1A, v20ss);
    const vsint32_t pp1A2 = spu_mulo(sum1A, v20ss);
    const vsint16_t pp1A3 = (vsint16_t)spu_shuffle((vsint16_t)pp1A1, (vsint16_t)pp1A2, mez);
    const vsint16_t pp1A = spu_add(pp1A3, v16ss);

    const vsint32_t pp2A1 = spu_mule(sum2A, v5ss);
    const vsint32_t pp2A2 = spu_mulo(sum2A, v5ss);
    const vsint16_t pp2A = (vsint16_t)spu_shuffle((vsint16_t)pp2A1, (vsint16_t)pp2A2, mez);

    const vsint16_t pp3A = spu_add(sum3A, pp1A);

    const vsint16_t psumA = spu_sub(pp3A, (vsint16_t)pp2A);

    vsint16_t sumA = spu_rlmask(psumA, -5);

    //Saturation to 0 and 255
    sat = spu_cmpgt(sumA,(vsint16_t)vzero);
    sumA = spu_and(sumA,(vsint16_t)sat);
    sat = spu_cmpgt(sumA,vmax);
    sumA = spu_sel(sumA,vmax,sat);

    const vuint8_t sum = (vuint8_t)spu_shuffle(sumA, (vsint16_t)vzero, packsu);

    const vuint8_t dst1 = *(vuint8_t *)dst;

    const vuint8_t dsum = spu_shuffle(dst1, sum, dstmask);
    vuint8_t fsum;
    OP_U8_SPU(fsum, dsum, dst1);

    *(vuint8_t *)dst=fsum;

    src += STRIDE_Y;
    dst += dstStride; /* stride is multiple of 16 so dstperm and dstmask can remain out of the loop */
   }
}

static void PREFIX_h264_qpel4_hv_lowpass_spu(uint8_t * dst, int16_t * tmp, uint8_t * src, int dstStride, int tmpStride, int h) {
  register int i;

  const int16_t i20ss = 20;
  const int16_t i5ss = 5;
  const int16_t imax = 255;

  const vsint32_t vzero = spu_splats(0);
  const vsint16_t v20ss = spu_splats(i20ss);
  const vsint16_t v5ss = spu_splats(i5ss);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  vuint16_t sat;

  const vuint8_t mergeh = {0x10,0x00,0x11,0x01,0x12,0x02,0x13,0x03,0x14,0x04,0x15,0x05,0x16,0x06,0x17,0x07};
  const vuint8_t packsu = {0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F};
  const vuint8_t mez = {0x02,0x03,0x12,0x13,0x06,0x07,0x16,0x17,0x0A,0x0B,0x1A,0x1B,0x0E,0x0F,0x1E,0x1F};

  const int permM2 = (unsigned int) (src-2) & 15;
  const int permM1 = (unsigned int) (src-1) & 15;
  const int permP0 = (unsigned int) (src) & 15;
  const int permP1 = (unsigned int) (src+1) & 15;
  const int permP2 = (unsigned int) (src+2) & 15;
  const int permP3 = (unsigned int) (src+3) & 15;

  register int align = ((((unsigned long)src) - 2) % 16);

  src -= (2 * STRIDE_Y);

  for (i = 0 ; i < (h+5) ; i ++) {
    vuint8_t srcM2, srcM1, srcP0, srcP1, srcP2, srcP3;
    vuint8_t srcR1 = *(vuint8_t *)(src-2);
    vuint8_t srcR2 = *(vuint8_t *)(src+14);

    switch (align) {
    default: {
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR1, permP2), spu_rlmaskqwbyte(srcR2, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR1, permP3), spu_rlmaskqwbyte(srcR2, permP3-16));
    } break;
    case 11: {
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR1, permP2), spu_rlmaskqwbyte(srcR2, permP2-16));
      srcP3 = srcR2;
    } break;
    case 12: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR1, permP1), spu_rlmaskqwbyte(srcR2, permP1-16));
      srcP2 = srcR2;
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 13: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = spu_or(spu_slqwbyte(srcR1, permP0), spu_rlmaskqwbyte(srcR2, permP0-16));
      srcP1 = srcR2;
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 14: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = spu_or(spu_slqwbyte(srcR1, permM1), spu_rlmaskqwbyte(srcR2, permM1-16));
      srcP0 = srcR2;
      srcP1 = spu_or(spu_slqwbyte(srcR2, permP1), spu_rlmaskqwbyte(srcR3, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    case 15: {
      vuint8_t srcR3 = *(vuint8_t *)(src+30);
      srcM2 = spu_or(spu_slqwbyte(srcR1, permM2), spu_rlmaskqwbyte(srcR2, permM2-16));
      srcM1 = srcR2;
      srcP0 = spu_or(spu_slqwbyte(srcR2, permP0), spu_rlmaskqwbyte(srcR3, permP0-16));
      srcP1 = spu_or(spu_slqwbyte(srcR2, permP1), spu_rlmaskqwbyte(srcR3, permP1-16));
      srcP2 = spu_or(spu_slqwbyte(srcR2, permP2), spu_rlmaskqwbyte(srcR3, permP2-16));
      srcP3 = spu_or(spu_slqwbyte(srcR2, permP3), spu_rlmaskqwbyte(srcR3, permP3-16));
    } break;
    }

    const vsint16_t srcP0A = (vsint16_t)spu_shuffle(srcP0, (vuint8_t)vzero, mergeh);
    const vsint16_t srcP1A = (vsint16_t)spu_shuffle(srcP1, (vuint8_t)vzero, mergeh);
    const vsint16_t srcP2A = (vsint16_t)spu_shuffle(srcP2, (vuint8_t)vzero, mergeh);
    const vsint16_t srcP3A = (vsint16_t)spu_shuffle(srcP3, (vuint8_t)vzero, mergeh);
    const vsint16_t srcM2A = (vsint16_t)spu_shuffle(srcM2, (vuint8_t)vzero, mergeh);
    const vsint16_t srcM1A = (vsint16_t)spu_shuffle(srcM1, (vuint8_t)vzero, mergeh);

    const vsint16_t sum1A = spu_add(srcP0A, srcP1A);
    const vsint16_t sum2A = spu_add(srcM1A, srcP2A);
    const vsint16_t sum3A = spu_add(srcM2A, srcP3A);

    const vsint32_t pp1A1 = spu_mule(sum1A, v20ss);
    const vsint32_t pp1A2 = spu_mulo(sum1A, v20ss);
    const vsint16_t pp1A3 = (vsint16_t)spu_shuffle((vsint16_t)pp1A1, (vsint16_t)pp1A2, mez);
    const vsint16_t pp1A = spu_add(pp1A3, sum3A);

    const vsint32_t pp2A1 = spu_mule(sum2A, v5ss);
    const vsint32_t pp2A2 = spu_mulo(sum2A, v5ss);
    const vsint16_t pp2A = (vsint16_t)spu_shuffle((vsint16_t)pp2A1, (vsint16_t)pp2A2, mez);

    const vsint16_t psumA = spu_sub(pp1A, pp2A);

    *(vsint16_t *)tmp = psumA;

    src += STRIDE_Y;
    tmp += tmpStride; /* int16_t*, and stride is 16, so it's OK here */
  }

  const int32_t ni10si = -10;
  const int16_t i1ss = 1;
  const int32_t i512si = 512;
  const int32_t ni16si = -16;

  const vsint32_t nv10si = spu_splats(ni10si);
  const vsint16_t v1ss = spu_splats(i1ss);
  const vsint32_t v512si = spu_splats(i512si);
  const vsint32_t nv16si = spu_splats(ni16si);

  const vuint8_t mperm = {0x00,0x08,0x01,0x09,0x02,0x0A,0x03,0x0B,0x04,0x0C,0x05,0x0D,0x06,0x0E,0x07,0x0F};
  const vuint8_t packs = {0x02,0x03,0x06,0x07,0x0A,0x0B,0x0E,0x0F,0x12,0x13,0x16,0x17,0x1A,0x1B,0x1E,0x1F};

  const int shift_dst = (unsigned int) (dst) & 15;
  /* 4x4 dest luma blocks are aligned or desaligned by 4,8 or 12*/
  vuint8_t dstmask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vuint8_t dst4mask0= {0x10,0x11,0x12,0x13,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst4mask4= {0x00,0x01,0x02,0x03,0x10,0x11,0x12,0x13,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst4mask8= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dst4mask12= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x10,0x11,0x12,0x13};

  switch(shift_dst){
    case 0:  dstmask = dst4mask0;
             break;
    case 4:  dstmask = dst4mask4;
             break;
    case 8:  dstmask = dst4mask8;
             break;
    case 12: dstmask = dst4mask12;
             break;
  }

  int16_t *tmpbis = tmp - (tmpStride * (h+5));

  vsint16_t tmpM2ssA = *(vsint16_t *)(tmpbis);
  tmpbis += tmpStride;
  vsint16_t tmpM1ssA = *(vsint16_t *)(tmpbis);
  tmpbis += tmpStride;
  vsint16_t tmpP0ssA = *(vsint16_t *)(tmpbis);
  tmpbis += tmpStride;
  vsint16_t tmpP1ssA = *(vsint16_t *)(tmpbis);
  tmpbis += tmpStride;
  vsint16_t tmpP2ssA = *(vsint16_t *)(tmpbis);
  tmpbis += tmpStride;

  for (i = 0 ; i < h ; i++) {
    const vsint16_t tmpP3ssA = *(vsint16_t *)(tmpbis);
    tmpbis += tmpStride;

    const vsint16_t sum1A = spu_add(tmpP0ssA, tmpP1ssA);
    const vsint16_t sum2A = spu_add(tmpM1ssA, tmpP2ssA);
    const vsint16_t sum3A = spu_add(tmpM2ssA, tmpP3ssA);

    tmpM2ssA = tmpM1ssA;
    tmpM1ssA = tmpP0ssA;
    tmpP0ssA = tmpP1ssA;
    tmpP1ssA = tmpP2ssA;
    tmpP2ssA = tmpP3ssA;

    const vsint32_t pp1Ae = spu_mule(sum1A, v20ss);
    const vsint32_t pp1Ao = spu_mulo(sum1A, v20ss);
    const vsint32_t pp2Ae = spu_mule(sum2A, v5ss);
    const vsint32_t pp2Ao = spu_mulo(sum2A, v5ss);

    const vsint32_t pp3Ae = spu_rlmask((vsint32_t)sum3A, nv16si);
    const vsint32_t pp3Ao = spu_mulo(sum3A, v1ss);

    const vsint32_t pp1cAe = spu_add(pp1Ae, v512si);
    const vsint32_t pp1cAo = spu_add(pp1Ao, v512si);

    const vsint32_t pp32Ae = spu_sub(pp3Ae, pp2Ae);
    const vsint32_t pp32Ao = spu_sub(pp3Ao, pp2Ao);

    const vsint32_t sumAe = spu_add(pp1cAe, pp32Ae);
    const vsint32_t sumAo = spu_add(pp1cAo, pp32Ao);

    const vsint32_t ssumAe = spu_rlmask(sumAe, nv10si);
    const vsint32_t ssumAo = spu_rlmask(sumAo, nv10si);

    vsint16_t ssume = (vsint16_t)spu_shuffle(ssumAe, vzero, packs);
    vsint16_t ssumo = (vsint16_t)spu_shuffle(ssumAo, vzero, packs);

    //Saturation to 0 and 255
    sat = spu_cmpgt(ssume,(vsint16_t)vzero);
    ssume = spu_and(ssume,(vsint16_t)sat);
    sat = spu_cmpgt(ssume,vmax);
    ssume = spu_sel(ssume,vmax,sat);
    sat = spu_cmpgt(ssumo,(vsint16_t)vzero);
    ssumo = spu_and(ssumo,(vsint16_t)sat);
    sat = spu_cmpgt(ssumo,vmax);
    ssumo = spu_sel(ssumo,vmax,sat);

    const vuint8_t sumv = (vuint8_t)spu_shuffle(ssume, ssumo, packsu);

    const vuint8_t sum = spu_shuffle(sumv, sumv, mperm);

    const vuint8_t dst1 = *(vuint8_t *)dst;

    const vuint8_t dsum = spu_shuffle(dst1, sum, dstmask);
    vuint8_t fsum;
    OP_U8_SPU(fsum, dsum, dst1);

    *(vuint8_t *)dst=fsum;

    dst += dstStride; /* stride is multiple of 16 so dstperm and dstmask can remain out of the loop */

  }
}
