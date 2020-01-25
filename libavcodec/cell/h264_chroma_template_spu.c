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


static void PREFIX_h264_chroma_mc8_spu(uint8_t * dst, uint8_t * src, int dst_stride, int h, int x, int y) {

  register int i;

  const int16_t i32ss= 32;
  const int16_t imax = 255;
  const int16_t iABCD1 = ((8 - x) * (8 - y));
  const int16_t iABCD2 = ((x) * (8 - y));
  const int16_t iABCD3 = ((8 - x) * (y));
  const int16_t iABCD4 = ((x) * (y));

  const vsint16_t vA = spu_splats(iABCD1);
  const vsint16_t vB = spu_splats(iABCD2);
  const vsint16_t vC = spu_splats(iABCD3);
  const vsint16_t vD = spu_splats(iABCD4);
  const vsint32_t vzero = spu_splats(0);
  const vsint16_t v32ss = spu_splats(i32ss);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  vuint16_t sat;

  const int shift_src =(unsigned int) src & 15;
  const int shift_dst =(unsigned int) dst & 15;
  const vuint8_t mergeh = {0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07};
  const vuint8_t packsu = {0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F};
  const vuint8_t mez = {0x02,0x03,0x12,0x13,0x06,0x07,0x16,0x17,0x0A,0x0B,0x1A,0x1B,0x0E,0x0F,0x1E,0x1F};
  const vuint8_t dstmask0= {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask8= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17};
  vuint8_t dstmask;

  if(shift_dst==0)
    dstmask=dstmask0;
  else
    dstmask=dstmask8;

  vuint8_t vsrc0uc1;
  vuint8_t vsrc0uc2;
  vuint8_t vsrc0uc;
  vuint8_t vsrc1uc;
  vsrc0uc1 = *(vuint8_t *)(src);
  vsrc0uc2 = *(vuint8_t *)(src+16);
  vsrc0uc = spu_or(spu_slqwbyte(vsrc0uc1, shift_src), spu_rlmaskqwbyte(vsrc0uc2, shift_src-16));
  vsrc1uc = spu_slqwbyte(vsrc0uc, 1);

  vsint16_t vsrc0ssH = (vsint16_t)spu_shuffle(vsrc0uc, vsrc0uc, mergeh);
  vsint16_t vsrc1ssH = (vsint16_t)spu_shuffle(vsrc1uc, vsrc1uc, mergeh);

  for (i = 0 ; i < h ; i++) {

    vuint8_t vsrc2uc1;
    vuint8_t vsrc2uc2;
    vuint8_t vsrc2uc;
    vuint8_t vsrc3uc;
    vsrc2uc1 = *(vuint8_t *)(src+STRIDE_C);
    vsrc2uc2 = *(vuint8_t *)(src+STRIDE_C+16);
    vsrc2uc = spu_or(spu_slqwbyte(vsrc2uc1, shift_src), spu_rlmaskqwbyte(vsrc2uc2, shift_src-16));
    vsrc3uc = spu_slqwbyte(vsrc2uc, 1);

    vsint16_t vsrc2ssH = (vsint16_t)spu_shuffle(vsrc2uc, vsrc2uc, mergeh);
    vsint16_t vsrc3ssH = (vsint16_t)spu_shuffle(vsrc3uc, vsrc3uc, mergeh);

    vsint16_t psum;

    vsint32_t psum1 = spu_mule(vsrc0ssH, vA);
    vsint32_t psum2 = spu_mulo(vsrc0ssH, vA);
    psum = (vsint16_t)spu_shuffle((vsint16_t)psum1, (vsint16_t)psum2, mez);

    psum1 = spu_mule(vsrc1ssH, vB);
    psum2 = spu_mulo(vsrc1ssH, vB);
    vsint16_t psum3 = (vsint16_t)spu_shuffle((vsint16_t)psum1, (vsint16_t)psum2, mez);
    psum = spu_add(psum3, psum);

    psum1 = spu_mule(vsrc2ssH, vC);
    psum2 = spu_mulo(vsrc2ssH, vC);
    psum3 = (vsint16_t)spu_shuffle((vsint16_t)psum1, (vsint16_t)psum2, mez);
    psum = spu_add(psum3, psum);

    psum1 = spu_mule(vsrc3ssH, vD);
    psum2 = spu_mulo(vsrc3ssH, vD);
    psum3 = (vsint16_t)spu_shuffle((vsint16_t)psum1, (vsint16_t)psum2, mez);
    psum = spu_add(psum3, psum);

    psum = spu_add(v32ss, psum);
    psum = spu_rlmask(psum, -6);

    //Saturation from 0 to 255
    sat = spu_cmpgt(psum,(vsint16_t)vzero);
    psum = spu_and(psum,(vsint16_t)sat);
    sat = spu_cmpgt(psum,vmax);
    psum = spu_sel(psum,vmax,sat);

    const vuint8_t ppsum = (vuint8_t)spu_shuffle(psum, (vsint16_t)vzero, packsu);

    const vuint8_t dst1 = *(vuint8_t *)dst;

    const vuint8_t dsum = spu_shuffle(dst1, ppsum, dstmask);
    vuint8_t fsum;
    OP_U8_SPU(fsum, dsum, dst1);

    *(vuint8_t *)dst=fsum;

    vsrc0ssH = vsrc2ssH;
    vsrc1ssH = vsrc3ssH;

    dst += dst_stride;
    //src += src_stride;
	src += STRIDE_C;
  }
}

static void PREFIX_h264_chroma_mc4_spu(uint8_t * dst, uint8_t * src, int dst_stride, int h, int x, int y) {

  register int i;

  const int16_t i32ss= 32;
  const int16_t imax = 255;
  const int16_t iABCD1 = ((8 - x) * (8 - y));
  const int16_t iABCD2 = ((x) * (8 - y));
  const int16_t iABCD3 = ((8 - x) * (y));
  const int16_t iABCD4 = ((x) * (y));

  const vsint16_t vA = spu_splats(iABCD1);
  const vsint16_t vB = spu_splats(iABCD2);
  const vsint16_t vC = spu_splats(iABCD3);
  const vsint16_t vD = spu_splats(iABCD4);
  const vsint32_t vzero = spu_splats(0);
  const vsint16_t v32ss = spu_splats(i32ss);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  vuint16_t sat;

  const vuint8_t mergeh = {0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07};
  const vuint8_t packsu = {0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F};
  const vuint8_t mez = {0x02,0x03,0x12,0x13,0x06,0x07,0x16,0x17,0x0A,0x0B,0x1A,0x1B,0x0E,0x0F,0x1E,0x1F};

  const int shift_src = (unsigned int) src & 15;
  const int shift_dst = (unsigned int) dst & 15;
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

  vuint8_t vsrc0uc1;
  vuint8_t vsrc0uc2;
  vuint8_t vsrc0uc;
  vuint8_t vsrc1uc;
  vsrc0uc1 = *(vuint8_t *)(src);
  vsrc0uc2 = *(vuint8_t *)(src+16);
  vsrc0uc = spu_or(spu_slqwbyte(vsrc0uc1, shift_src), spu_rlmaskqwbyte(vsrc0uc2, shift_src-16));
  vsrc1uc = spu_slqwbyte(vsrc0uc, 1);

  vsint16_t vsrc0ssH = (vsint16_t)spu_shuffle(vsrc0uc, vsrc0uc, mergeh);
  vsint16_t vsrc1ssH = (vsint16_t)spu_shuffle(vsrc1uc, vsrc1uc, mergeh);

  for (i = 0 ; i < h ; i++) {

    vuint8_t vsrc2uc1;
    vuint8_t vsrc2uc2;
    vuint8_t vsrc2uc;
    vuint8_t vsrc3uc;
    vsrc2uc1 = *(vuint8_t *)(src+STRIDE_C);
    vsrc2uc2 = *(vuint8_t *)(src+STRIDE_C+16);
    vsrc2uc = spu_or(spu_slqwbyte(vsrc2uc1, shift_src), spu_rlmaskqwbyte(vsrc2uc2, shift_src-16));
    vsrc3uc = spu_slqwbyte(vsrc2uc, 1);

    vsint16_t vsrc2ssH = (vsint16_t)spu_shuffle(vsrc2uc, vsrc2uc, mergeh);
    vsint16_t vsrc3ssH = (vsint16_t)spu_shuffle(vsrc3uc, vsrc3uc, mergeh);

    vsint16_t psum;

    vsint32_t psum1 = spu_mule(vsrc0ssH, vA);
    vsint32_t psum2 = spu_mulo(vsrc0ssH, vA);
    psum = (vsint16_t)spu_shuffle((vsint16_t)psum1, (vsint16_t)psum2, mez);

    psum1 = spu_mule(vsrc1ssH, vB);
    psum2 = spu_mulo(vsrc1ssH, vB);
    vsint16_t psum3 = (vsint16_t)spu_shuffle((vsint16_t)psum1, (vsint16_t)psum2, mez);
    psum = spu_add(psum3, psum);

    psum1 = spu_mule(vsrc2ssH, vC);
    psum2 = spu_mulo(vsrc2ssH, vC);
    psum3 = (vsint16_t)spu_shuffle((vsint16_t)psum1, (vsint16_t)psum2, mez);
    psum = spu_add(psum3, psum);

    psum1 = spu_mule(vsrc3ssH, vD);
    psum2 = spu_mulo(vsrc3ssH, vD);
    psum3 = (vsint16_t)spu_shuffle((vsint16_t)psum1, (vsint16_t)psum2, mez);
    psum = spu_add(psum3, psum);

    psum = spu_add(v32ss, psum);
    psum = spu_rlmask(psum, -6);

    //Saturation from 0 to 255
    sat = spu_cmpgt(psum,(vsint16_t)vzero);
    psum = spu_and(psum,(vsint16_t)sat);
    sat = spu_cmpgt(psum,vmax);
    psum = spu_sel(psum,vmax,sat);

    const vuint8_t ppsum = (vuint8_t)spu_shuffle(psum, (vsint16_t)vzero, packsu);

    const vuint8_t dst1 = *(vuint8_t *)dst;

    const vuint8_t dsum = spu_shuffle(dst1, ppsum, dstmask);
    vuint8_t fsum;
    OP_U8_SPU(fsum, dsum, dst1);

    *(vuint8_t *)dst=fsum;

    vsrc0ssH = vsrc2ssH;
    vsrc1ssH = vsrc3ssH;

    dst += dst_stride;
    src += STRIDE_C;
  }
}

static void PREFIX_h264_chroma_mc2_spu(uint8_t * dst, uint8_t * src, int dst_stride, int h, int x, int y) {

  register int i;

  const int16_t i32ss= 32;
  const int16_t imax = 255;
  const int16_t iABCD1 = ((8 - x) * (8 - y));
  const int16_t iABCD2 = ((x) * (8 - y));
  const int16_t iABCD3 = ((8 - x) * (y));
  const int16_t iABCD4 = ((x) * (y));

  const vsint16_t vA = spu_splats(iABCD1);
  const vsint16_t vB = spu_splats(iABCD2);
  const vsint16_t vC = spu_splats(iABCD3);
  const vsint16_t vD = spu_splats(iABCD4);
  const vsint32_t vzero = spu_splats(0);
  const vsint16_t v32ss = spu_splats(i32ss);
  const vsint16_t vmax = (vsint16_t)spu_splats(imax);
  vuint16_t sat;

  const vuint8_t mergeh = {0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07};
  const vuint8_t packsu = {0x01,0x03,0x05,0x07,0x09,0x0B,0x0D,0x0F,0x11,0x13,0x15,0x17,0x19,0x1B,0x1D,0x1F};
  const vuint8_t mez = {0x02,0x03,0x12,0x13,0x06,0x07,0x16,0x17,0x0A,0x0B,0x1A,0x1B,0x0E,0x0F,0x1E,0x1F};

  const int shift_src = (unsigned int) src & 15;
  const int shift_dst = (unsigned int) dst & 15;
  vuint8_t dstmask = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const vuint8_t dstmask0=  {0x10,0x11,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask2=  {0x00,0x01,0x10,0x11,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask4=  {0x00,0x01,0x02,0x03,0x10,0x11,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask6=  {0x00,0x01,0x02,0x03,0x04,0x05,0x10,0x11,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask8=  {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x10,0x11,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask10= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x10,0x11,0x0C,0x0D,0x0E,0x0F};
  const vuint8_t dstmask12= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x10,0x11,0x0E,0x0F};
  const vuint8_t dstmask14= {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x10,0x11};

  switch(shift_dst){
    case 0:  dstmask = dstmask0;
             break;
    case 2:  dstmask = dstmask2;
             break;
    case 4:  dstmask = dstmask4;
             break;
    case 6:  dstmask = dstmask6;
             break;
    case 8:  dstmask = dstmask8;
             break;
    case 10: dstmask = dstmask10;
             break;
    case 12: dstmask = dstmask12;
             break;
    case 14: dstmask = dstmask14;
             break;
  }

  vuint8_t vsrc0uc1;
  vuint8_t vsrc0uc2;
  vuint8_t vsrc0uc;
  vuint8_t vsrc1uc;
  vsrc0uc1 = *(vuint8_t *)(src);
  vsrc0uc2 = *(vuint8_t *)(src+16);
  vsrc0uc = spu_or(spu_slqwbyte(vsrc0uc1, shift_src), spu_rlmaskqwbyte(vsrc0uc2, shift_src-16));
  vsrc1uc = spu_slqwbyte(vsrc0uc, 1);

  vsint16_t vsrc0ssH = (vsint16_t)spu_shuffle(vsrc0uc, vsrc0uc, mergeh);
  vsint16_t vsrc1ssH = (vsint16_t)spu_shuffle(vsrc1uc, vsrc1uc, mergeh);

  for (i = 0 ; i < h ; i++) {

    vuint8_t vsrc2uc1;
    vuint8_t vsrc2uc2;
    vuint8_t vsrc2uc;
    vuint8_t vsrc3uc;
    vsrc2uc1 = *(vuint8_t *)(src+STRIDE_C);
    vsrc2uc2 = *(vuint8_t *)(src+STRIDE_C+16);
    vsrc2uc = spu_or(spu_slqwbyte(vsrc2uc1, shift_src), spu_rlmaskqwbyte(vsrc2uc2, shift_src-16));
    vsrc3uc = spu_slqwbyte(vsrc2uc, 1);

    vsint16_t vsrc2ssH = (vsint16_t)spu_shuffle(vsrc2uc, vsrc2uc, mergeh);
    vsint16_t vsrc3ssH = (vsint16_t)spu_shuffle(vsrc3uc, vsrc3uc, mergeh);

    vsint16_t psum;

    vsint32_t psum1 = spu_mule(vsrc0ssH, vA);
    vsint32_t psum2 = spu_mulo(vsrc0ssH, vA);
    psum = (vsint16_t)spu_shuffle((vsint16_t)psum1, (vsint16_t)psum2, mez);

    psum1 = spu_mule(vsrc1ssH, vB);
    psum2 = spu_mulo(vsrc1ssH, vB);
    vsint16_t psum3 = (vsint16_t)spu_shuffle((vsint16_t)psum1, (vsint16_t)psum2, mez);
    psum = spu_add(psum3, psum);

    psum1 = spu_mule(vsrc2ssH, vC);
    psum2 = spu_mulo(vsrc2ssH, vC);
    psum3 = (vsint16_t)spu_shuffle((vsint16_t)psum1, (vsint16_t)psum2, mez);
    psum = spu_add(psum3, psum);

    psum1 = spu_mule(vsrc3ssH, vD);
    psum2 = spu_mulo(vsrc3ssH, vD);
    psum3 = (vsint16_t)spu_shuffle((vsint16_t)psum1, (vsint16_t)psum2, mez);
    psum = spu_add(psum3, psum);

    psum = spu_add(v32ss, psum);
    psum = spu_rlmask(psum, -6);

    //Saturation from 0 to 255
    sat = spu_cmpgt(psum,(vsint16_t)vzero);
    psum = spu_and(psum,(vsint16_t)sat);
    sat = spu_cmpgt(psum,vmax);
    psum = spu_sel(psum,vmax,sat);

    const vuint8_t ppsum = (vuint8_t)spu_shuffle(psum, (vsint16_t)vzero, packsu);

    const vuint8_t dst1 = *(vuint8_t *)dst;

    const vuint8_t dsum = spu_shuffle(dst1, ppsum, dstmask);
    vuint8_t fsum;
    OP_U8_SPU(fsum, dsum, dst1);

    *(vuint8_t *)dst=fsum;

    vsrc0ssH = vsrc2ssH;
    vsrc1ssH = vsrc3ssH;

    dst += dst_stride;
    src += STRIDE_C;
  }
}

