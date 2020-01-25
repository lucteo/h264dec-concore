/*
 * Copyright (c) 2002 Brian Foley
 * Copyright (c) 2002 Dieter Shirley
 * Copyright (c) 2003-2004 Romain Dolbeau <romain@dolbeau.org>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "config.h"
#if HAVE_ALTIVEC_H
#include <altivec.h>
#endif
#include "libavcodec/dsputil.h"
#include "dsputil_ppc.h"
#include "util_altivec.h"
#include "types_altivec.h"
#include "dsputil_altivec.h"


static void get_pixels_altivec(DCTELEM *restrict block, const uint8_t *pixels, int line_size)
{
    int i;
    vector unsigned char perm, bytes, *pixv;
    const vector unsigned char zero = (const vector unsigned char)vec_splat_u8(0);
    vector signed short shorts;

    for (i = 0; i < 8; i++) {
        // Read potentially unaligned pixels.
        // We're reading 16 pixels, and actually only want 8,
        // but we simply ignore the extras.
        perm = vec_lvsl(0, pixels);
        pixv = (vector unsigned char *) pixels;
        bytes = vec_perm(pixv[0], pixv[1], perm);

        // convert the bytes into shorts
        shorts = (vector signed short)vec_mergeh(zero, bytes);

        // save the data to the block, we assume the block is 16-byte aligned
        vec_st(shorts, i*16, (vector signed short*)block);

        pixels += line_size;
    }
}

static void diff_pixels_altivec(DCTELEM *restrict block, const uint8_t *s1,
        const uint8_t *s2, int stride)
{
    int i;
    vector unsigned char perm, bytes, *pixv;
    const vector unsigned char zero = (const vector unsigned char)vec_splat_u8(0);
    vector signed short shorts1, shorts2;

    for (i = 0; i < 4; i++) {
        // Read potentially unaligned pixels
        // We're reading 16 pixels, and actually only want 8,
        // but we simply ignore the extras.
        perm = vec_lvsl(0, s1);
        pixv = (vector unsigned char *) s1;
        bytes = vec_perm(pixv[0], pixv[1], perm);

        // convert the bytes into shorts
        shorts1 = (vector signed short)vec_mergeh(zero, bytes);

        // Do the same for the second block of pixels
        perm = vec_lvsl(0, s2);
        pixv = (vector unsigned char *) s2;
        bytes = vec_perm(pixv[0], pixv[1], perm);

        // convert the bytes into shorts
        shorts2 = (vector signed short)vec_mergeh(zero, bytes);

        // Do the subtraction
        shorts1 = vec_sub(shorts1, shorts2);

        // save the data to the block, we assume the block is 16-byte aligned
        vec_st(shorts1, 0, (vector signed short*)block);

        s1 += stride;
        s2 += stride;
        block += 8;


        // The code below is a copy of the code above... This is a manual
        // unroll.

        // Read potentially unaligned pixels
        // We're reading 16 pixels, and actually only want 8,
        // but we simply ignore the extras.
        perm = vec_lvsl(0, s1);
        pixv = (vector unsigned char *) s1;
        bytes = vec_perm(pixv[0], pixv[1], perm);

        // convert the bytes into shorts
        shorts1 = (vector signed short)vec_mergeh(zero, bytes);

        // Do the same for the second block of pixels
        perm = vec_lvsl(0, s2);
        pixv = (vector unsigned char *) s2;
        bytes = vec_perm(pixv[0], pixv[1], perm);

        // convert the bytes into shorts
        shorts2 = (vector signed short)vec_mergeh(zero, bytes);

        // Do the subtraction
        shorts1 = vec_sub(shorts1, shorts2);

        // save the data to the block, we assume the block is 16-byte aligned
        vec_st(shorts1, 0, (vector signed short*)block);

        s1 += stride;
        s2 += stride;
        block += 8;
    }
}


static void clear_block_altivec(DCTELEM *block) {
    LOAD_ZERO;
    vec_st(zero_s16v,   0, block);
    vec_st(zero_s16v,  16, block);
    vec_st(zero_s16v,  32, block);
    vec_st(zero_s16v,  48, block);
    vec_st(zero_s16v,  64, block);
    vec_st(zero_s16v,  80, block);
    vec_st(zero_s16v,  96, block);
    vec_st(zero_s16v, 112, block);
}



/* next one assumes that ((line_size % 16) == 0) */
void put_pixels16_altivec(uint8_t *block, const uint8_t *pixels, int line_size, int h)
{
POWERPC_PERF_DECLARE(altivec_put_pixels16_num, 1);
    register vector unsigned char pixelsv1, pixelsv2;
    register vector unsigned char pixelsv1B, pixelsv2B;
    register vector unsigned char pixelsv1C, pixelsv2C;
    register vector unsigned char pixelsv1D, pixelsv2D;

    register vector unsigned char perm = vec_lvsl(0, pixels);
    int i;
    register int line_size_2 = line_size << 1;
    register int line_size_3 = line_size + line_size_2;
    register int line_size_4 = line_size << 2;

POWERPC_PERF_START_COUNT(altivec_put_pixels16_num, 1);
// hand-unrolling the loop by 4 gains about 15%
// mininum execution time goes from 74 to 60 cycles
// it's faster than -funroll-loops, but using
// -funroll-loops w/ this is bad - 74 cycles again.
// all this is on a 7450, tuning for the 7450
#if 0
    for (i = 0; i < h; i++) {
        pixelsv1 = vec_ld(0, pixels);
        pixelsv2 = vec_ld(16, pixels);
        vec_st(vec_perm(pixelsv1, pixelsv2, perm),
               0, block);
        pixels+=line_size;
        block +=line_size;
    }
#else
    for (i = 0; i < h; i += 4) {
        pixelsv1  = vec_ld( 0, pixels);
        pixelsv2  = vec_ld(15, pixels);
        pixelsv1B = vec_ld(line_size, pixels);
        pixelsv2B = vec_ld(15 + line_size, pixels);
        pixelsv1C = vec_ld(line_size_2, pixels);
        pixelsv2C = vec_ld(15 + line_size_2, pixels);
        pixelsv1D = vec_ld(line_size_3, pixels);
        pixelsv2D = vec_ld(15 + line_size_3, pixels);
        vec_st(vec_perm(pixelsv1, pixelsv2, perm),
               0, (unsigned char*)block);
        vec_st(vec_perm(pixelsv1B, pixelsv2B, perm),
               line_size, (unsigned char*)block);
        vec_st(vec_perm(pixelsv1C, pixelsv2C, perm),
               line_size_2, (unsigned char*)block);
        vec_st(vec_perm(pixelsv1D, pixelsv2D, perm),
               line_size_3, (unsigned char*)block);
        pixels+=line_size_4;
        block +=line_size_4;
    }
#endif
POWERPC_PERF_STOP_COUNT(altivec_put_pixels16_num, 1);
}

/* next one assumes that ((line_size % 16) == 0) */
#define op_avg(a,b)  a = ( ((a)|(b)) - ((((a)^(b))&0xFEFEFEFEUL)>>1) )
void avg_pixels16_altivec(uint8_t *block, const uint8_t *pixels, int line_size, int h)
{
POWERPC_PERF_DECLARE(altivec_avg_pixels16_num, 1);
    register vector unsigned char pixelsv1, pixelsv2, pixelsv, blockv;
    register vector unsigned char perm = vec_lvsl(0, pixels);
    int i;

POWERPC_PERF_START_COUNT(altivec_avg_pixels16_num, 1);

    for (i = 0; i < h; i++) {
        pixelsv1 = vec_ld( 0, pixels);
        pixelsv2 = vec_ld(16,pixels);
        blockv = vec_ld(0, block);
        pixelsv = vec_perm(pixelsv1, pixelsv2, perm);
        blockv = vec_avg(blockv,pixelsv);
        vec_st(blockv, 0, (unsigned char*)block);
        pixels+=line_size;
        block +=line_size;
    }

POWERPC_PERF_STOP_COUNT(altivec_avg_pixels16_num, 1);
}

/* next one assumes that ((line_size % 8) == 0) */
static void avg_pixels8_altivec(uint8_t * block, const uint8_t * pixels, int line_size, int h)
{
POWERPC_PERF_DECLARE(altivec_avg_pixels8_num, 1);
    register vector unsigned char pixelsv1, pixelsv2, pixelsv, blockv;
    int i;

POWERPC_PERF_START_COUNT(altivec_avg_pixels8_num, 1);

   for (i = 0; i < h; i++) {
       /* block is 8 bytes-aligned, so we're either in the
          left block (16 bytes-aligned) or in the right block (not) */
       int rightside = ((unsigned long)block & 0x0000000F);

       blockv = vec_ld(0, block);
       pixelsv1 = vec_ld( 0, pixels);
       pixelsv2 = vec_ld(16, pixels);
       pixelsv = vec_perm(pixelsv1, pixelsv2, vec_lvsl(0, pixels));

       if (rightside) {
           pixelsv = vec_perm(blockv, pixelsv, vcprm(0,1,s0,s1));
       } else {
           pixelsv = vec_perm(blockv, pixelsv, vcprm(s0,s1,2,3));
       }

       blockv = vec_avg(blockv, pixelsv);

       vec_st(blockv, 0, block);

       pixels += line_size;
       block += line_size;
   }

POWERPC_PERF_STOP_COUNT(altivec_avg_pixels8_num, 1);
}

/* next one assumes that ((line_size % 8) == 0) */
static void put_pixels8_xy2_altivec(uint8_t *block, const uint8_t *pixels, int line_size, int h)
{
POWERPC_PERF_DECLARE(altivec_put_pixels8_xy2_num, 1);
    register int i;
    register vector unsigned char pixelsv1, pixelsv2, pixelsavg;
    register vector unsigned char blockv, temp1, temp2;
    register vector unsigned short pixelssum1, pixelssum2, temp3;
    register const vector unsigned char vczero = (const vector unsigned char)vec_splat_u8(0);
    register const vector unsigned short vctwo = (const vector unsigned short)vec_splat_u16(2);

    temp1 = vec_ld(0, pixels);
    temp2 = vec_ld(16, pixels);
    pixelsv1 = vec_perm(temp1, temp2, vec_lvsl(0, pixels));
    if ((((unsigned long)pixels) & 0x0000000F) ==  0x0000000F) {
        pixelsv2 = temp2;
    } else {
        pixelsv2 = vec_perm(temp1, temp2, vec_lvsl(1, pixels));
    }
    pixelsv1 = vec_mergeh(vczero, pixelsv1);
    pixelsv2 = vec_mergeh(vczero, pixelsv2);
    pixelssum1 = vec_add((vector unsigned short)pixelsv1,
                         (vector unsigned short)pixelsv2);
    pixelssum1 = vec_add(pixelssum1, vctwo);

POWERPC_PERF_START_COUNT(altivec_put_pixels8_xy2_num, 1);
    for (i = 0; i < h ; i++) {
        int rightside = ((unsigned long)block & 0x0000000F);
        blockv = vec_ld(0, block);

        temp1 = vec_ld(line_size, pixels);
        temp2 = vec_ld(line_size + 16, pixels);
        pixelsv1 = vec_perm(temp1, temp2, vec_lvsl(line_size, pixels));
        if (((((unsigned long)pixels) + line_size) & 0x0000000F) ==  0x0000000F) {
            pixelsv2 = temp2;
        } else {
            pixelsv2 = vec_perm(temp1, temp2, vec_lvsl(line_size + 1, pixels));
        }

        pixelsv1 = vec_mergeh(vczero, pixelsv1);
        pixelsv2 = vec_mergeh(vczero, pixelsv2);
        pixelssum2 = vec_add((vector unsigned short)pixelsv1,
                             (vector unsigned short)pixelsv2);
        temp3 = vec_add(pixelssum1, pixelssum2);
        temp3 = vec_sra(temp3, vctwo);
        pixelssum1 = vec_add(pixelssum2, vctwo);
        pixelsavg = vec_packsu(temp3, (vector unsigned short) vczero);

        if (rightside) {
            blockv = vec_perm(blockv, pixelsavg, vcprm(0, 1, s0, s1));
        } else {
            blockv = vec_perm(blockv, pixelsavg, vcprm(s0, s1, 2, 3));
        }

        vec_st(blockv, 0, block);

        block += line_size;
        pixels += line_size;
    }

POWERPC_PERF_STOP_COUNT(altivec_put_pixels8_xy2_num, 1);
}

/* next one assumes that ((line_size % 8) == 0) */
static void put_no_rnd_pixels8_xy2_altivec(uint8_t *block, const uint8_t *pixels, int line_size, int h)
{
POWERPC_PERF_DECLARE(altivec_put_no_rnd_pixels8_xy2_num, 1);
    register int i;
    register vector unsigned char pixelsv1, pixelsv2, pixelsavg;
    register vector unsigned char blockv, temp1, temp2;
    register vector unsigned short pixelssum1, pixelssum2, temp3;
    register const vector unsigned char vczero = (const vector unsigned char)vec_splat_u8(0);
    register const vector unsigned short vcone = (const vector unsigned short)vec_splat_u16(1);
    register const vector unsigned short vctwo = (const vector unsigned short)vec_splat_u16(2);

    temp1 = vec_ld(0, pixels);
    temp2 = vec_ld(16, pixels);
    pixelsv1 = vec_perm(temp1, temp2, vec_lvsl(0, pixels));
    if ((((unsigned long)pixels) & 0x0000000F) ==  0x0000000F) {
        pixelsv2 = temp2;
    } else {
        pixelsv2 = vec_perm(temp1, temp2, vec_lvsl(1, pixels));
    }
    pixelsv1 = vec_mergeh(vczero, pixelsv1);
    pixelsv2 = vec_mergeh(vczero, pixelsv2);
    pixelssum1 = vec_add((vector unsigned short)pixelsv1,
                         (vector unsigned short)pixelsv2);
    pixelssum1 = vec_add(pixelssum1, vcone);

POWERPC_PERF_START_COUNT(altivec_put_no_rnd_pixels8_xy2_num, 1);
    for (i = 0; i < h ; i++) {
        int rightside = ((unsigned long)block & 0x0000000F);
        blockv = vec_ld(0, block);

        temp1 = vec_ld(line_size, pixels);
        temp2 = vec_ld(line_size + 16, pixels);
        pixelsv1 = vec_perm(temp1, temp2, vec_lvsl(line_size, pixels));
        if (((((unsigned long)pixels) + line_size) & 0x0000000F) ==  0x0000000F) {
            pixelsv2 = temp2;
        } else {
            pixelsv2 = vec_perm(temp1, temp2, vec_lvsl(line_size + 1, pixels));
        }

        pixelsv1 = vec_mergeh(vczero, pixelsv1);
        pixelsv2 = vec_mergeh(vczero, pixelsv2);
        pixelssum2 = vec_add((vector unsigned short)pixelsv1,
                             (vector unsigned short)pixelsv2);
        temp3 = vec_add(pixelssum1, pixelssum2);
        temp3 = vec_sra(temp3, vctwo);
        pixelssum1 = vec_add(pixelssum2, vcone);
        pixelsavg = vec_packsu(temp3, (vector unsigned short) vczero);

        if (rightside) {
            blockv = vec_perm(blockv, pixelsavg, vcprm(0, 1, s0, s1));
        } else {
            blockv = vec_perm(blockv, pixelsavg, vcprm(s0, s1, 2, 3));
        }

        vec_st(blockv, 0, block);

        block += line_size;
        pixels += line_size;
    }

POWERPC_PERF_STOP_COUNT(altivec_put_no_rnd_pixels8_xy2_num, 1);
}

/* next one assumes that ((line_size % 16) == 0) */
static void put_pixels16_xy2_altivec(uint8_t * block, const uint8_t * pixels, int line_size, int h)
{
POWERPC_PERF_DECLARE(altivec_put_pixels16_xy2_num, 1);
    register int i;
    register vector unsigned char pixelsv1, pixelsv2, pixelsv3, pixelsv4;
    register vector unsigned char blockv, temp1, temp2;
    register vector unsigned short temp3, temp4,
        pixelssum1, pixelssum2, pixelssum3, pixelssum4;
    register const vector unsigned char vczero = (const vector unsigned char)vec_splat_u8(0);
    register const vector unsigned short vctwo = (const vector unsigned short)vec_splat_u16(2);

POWERPC_PERF_START_COUNT(altivec_put_pixels16_xy2_num, 1);

    temp1 = vec_ld(0, pixels);
    temp2 = vec_ld(16, pixels);
    pixelsv1 = vec_perm(temp1, temp2, vec_lvsl(0, pixels));
    if ((((unsigned long)pixels) & 0x0000000F) ==  0x0000000F) {
        pixelsv2 = temp2;
    } else {
        pixelsv2 = vec_perm(temp1, temp2, vec_lvsl(1, pixels));
    }
    pixelsv3 = vec_mergel(vczero, pixelsv1);
    pixelsv4 = vec_mergel(vczero, pixelsv2);
    pixelsv1 = vec_mergeh(vczero, pixelsv1);
    pixelsv2 = vec_mergeh(vczero, pixelsv2);
    pixelssum3 = vec_add((vector unsigned short)pixelsv3,
                         (vector unsigned short)pixelsv4);
    pixelssum3 = vec_add(pixelssum3, vctwo);
    pixelssum1 = vec_add((vector unsigned short)pixelsv1,
                         (vector unsigned short)pixelsv2);
    pixelssum1 = vec_add(pixelssum1, vctwo);

    for (i = 0; i < h ; i++) {
        blockv = vec_ld(0, block);

        temp1 = vec_ld(line_size, pixels);
        temp2 = vec_ld(line_size + 16, pixels);
        pixelsv1 = vec_perm(temp1, temp2, vec_lvsl(line_size, pixels));
        if (((((unsigned long)pixels) + line_size) & 0x0000000F) ==  0x0000000F) {
            pixelsv2 = temp2;
        } else {
            pixelsv2 = vec_perm(temp1, temp2, vec_lvsl(line_size + 1, pixels));
        }

        pixelsv3 = vec_mergel(vczero, pixelsv1);
        pixelsv4 = vec_mergel(vczero, pixelsv2);
        pixelsv1 = vec_mergeh(vczero, pixelsv1);
        pixelsv2 = vec_mergeh(vczero, pixelsv2);

        pixelssum4 = vec_add((vector unsigned short)pixelsv3,
                             (vector unsigned short)pixelsv4);
        pixelssum2 = vec_add((vector unsigned short)pixelsv1,
                             (vector unsigned short)pixelsv2);
        temp4 = vec_add(pixelssum3, pixelssum4);
        temp4 = vec_sra(temp4, vctwo);
        temp3 = vec_add(pixelssum1, pixelssum2);
        temp3 = vec_sra(temp3, vctwo);

        pixelssum3 = vec_add(pixelssum4, vctwo);
        pixelssum1 = vec_add(pixelssum2, vctwo);

        blockv = vec_packsu(temp3, temp4);

        vec_st(blockv, 0, block);

        block += line_size;
        pixels += line_size;
    }

POWERPC_PERF_STOP_COUNT(altivec_put_pixels16_xy2_num, 1);
}

/* next one assumes that ((line_size % 16) == 0) */
static void put_no_rnd_pixels16_xy2_altivec(uint8_t * block, const uint8_t * pixels, int line_size, int h)
{
POWERPC_PERF_DECLARE(altivec_put_no_rnd_pixels16_xy2_num, 1);
    register int i;
    register vector unsigned char pixelsv1, pixelsv2, pixelsv3, pixelsv4;
    register vector unsigned char blockv, temp1, temp2;
    register vector unsigned short temp3, temp4,
        pixelssum1, pixelssum2, pixelssum3, pixelssum4;
    register const vector unsigned char vczero = (const vector unsigned char)vec_splat_u8(0);
    register const vector unsigned short vcone = (const vector unsigned short)vec_splat_u16(1);
    register const vector unsigned short vctwo = (const vector unsigned short)vec_splat_u16(2);

POWERPC_PERF_START_COUNT(altivec_put_no_rnd_pixels16_xy2_num, 1);

    temp1 = vec_ld(0, pixels);
    temp2 = vec_ld(16, pixels);
    pixelsv1 = vec_perm(temp1, temp2, vec_lvsl(0, pixels));
    if ((((unsigned long)pixels) & 0x0000000F) ==  0x0000000F) {
        pixelsv2 = temp2;
    } else {
        pixelsv2 = vec_perm(temp1, temp2, vec_lvsl(1, pixels));
    }
    pixelsv3 = vec_mergel(vczero, pixelsv1);
    pixelsv4 = vec_mergel(vczero, pixelsv2);
    pixelsv1 = vec_mergeh(vczero, pixelsv1);
    pixelsv2 = vec_mergeh(vczero, pixelsv2);
    pixelssum3 = vec_add((vector unsigned short)pixelsv3,
                         (vector unsigned short)pixelsv4);
    pixelssum3 = vec_add(pixelssum3, vcone);
    pixelssum1 = vec_add((vector unsigned short)pixelsv1,
                         (vector unsigned short)pixelsv2);
    pixelssum1 = vec_add(pixelssum1, vcone);

    for (i = 0; i < h ; i++) {
        blockv = vec_ld(0, block);

        temp1 = vec_ld(line_size, pixels);
        temp2 = vec_ld(line_size + 16, pixels);
        pixelsv1 = vec_perm(temp1, temp2, vec_lvsl(line_size, pixels));
        if (((((unsigned long)pixels) + line_size) & 0x0000000F) ==  0x0000000F) {
            pixelsv2 = temp2;
        } else {
            pixelsv2 = vec_perm(temp1, temp2, vec_lvsl(line_size + 1, pixels));
        }

        pixelsv3 = vec_mergel(vczero, pixelsv1);
        pixelsv4 = vec_mergel(vczero, pixelsv2);
        pixelsv1 = vec_mergeh(vczero, pixelsv1);
        pixelsv2 = vec_mergeh(vczero, pixelsv2);

        pixelssum4 = vec_add((vector unsigned short)pixelsv3,
                             (vector unsigned short)pixelsv4);
        pixelssum2 = vec_add((vector unsigned short)pixelsv1,
                             (vector unsigned short)pixelsv2);
        temp4 = vec_add(pixelssum3, pixelssum4);
        temp4 = vec_sra(temp4, vctwo);
        temp3 = vec_add(pixelssum1, pixelssum2);
        temp3 = vec_sra(temp3, vctwo);

        pixelssum3 = vec_add(pixelssum4, vcone);
        pixelssum1 = vec_add(pixelssum2, vcone);

        blockv = vec_packsu(temp3, temp4);

        vec_st(blockv, 0, block);

        block += line_size;
        pixels += line_size;
    }

POWERPC_PERF_STOP_COUNT(altivec_put_no_rnd_pixels16_xy2_num, 1);
}

/* next one assumes that ((line_size % 8) == 0) */
static void avg_pixels8_xy2_altivec(uint8_t *block, const uint8_t *pixels, int line_size, int h)
{
POWERPC_PERF_DECLARE(altivec_avg_pixels8_xy2_num, 1);
    register int i;
    register vector unsigned char pixelsv1, pixelsv2, pixelsavg;
    register vector unsigned char blockv, temp1, temp2, blocktemp;
    register vector unsigned short pixelssum1, pixelssum2, temp3;

    register const vector unsigned char vczero = (const vector unsigned char)
                                        vec_splat_u8(0);
    register const vector unsigned short vctwo = (const vector unsigned short)
                                        vec_splat_u16(2);

    temp1 = vec_ld(0, pixels);
    temp2 = vec_ld(16, pixels);
    pixelsv1 = vec_perm(temp1, temp2, vec_lvsl(0, pixels));
    if ((((unsigned long)pixels) & 0x0000000F) ==  0x0000000F) {
        pixelsv2 = temp2;
    } else {
        pixelsv2 = vec_perm(temp1, temp2, vec_lvsl(1, pixels));
    }
    pixelsv1 = vec_mergeh(vczero, pixelsv1);
    pixelsv2 = vec_mergeh(vczero, pixelsv2);
    pixelssum1 = vec_add((vector unsigned short)pixelsv1,
                         (vector unsigned short)pixelsv2);
    pixelssum1 = vec_add(pixelssum1, vctwo);

POWERPC_PERF_START_COUNT(altivec_avg_pixels8_xy2_num, 1);
    for (i = 0; i < h ; i++) {
        int rightside = ((unsigned long)block & 0x0000000F);
        blockv = vec_ld(0, block);

        temp1 = vec_ld(line_size, pixels);
        temp2 = vec_ld(line_size + 16, pixels);
        pixelsv1 = vec_perm(temp1, temp2, vec_lvsl(line_size, pixels));
        if (((((unsigned long)pixels) + line_size) & 0x0000000F) ==  0x0000000F) {
            pixelsv2 = temp2;
        } else {
            pixelsv2 = vec_perm(temp1, temp2, vec_lvsl(line_size + 1, pixels));
        }

        pixelsv1 = vec_mergeh(vczero, pixelsv1);
        pixelsv2 = vec_mergeh(vczero, pixelsv2);
        pixelssum2 = vec_add((vector unsigned short)pixelsv1,
                             (vector unsigned short)pixelsv2);
        temp3 = vec_add(pixelssum1, pixelssum2);
        temp3 = vec_sra(temp3, vctwo);
        pixelssum1 = vec_add(pixelssum2, vctwo);
        pixelsavg = vec_packsu(temp3, (vector unsigned short) vczero);

        if (rightside) {
            blocktemp = vec_perm(blockv, pixelsavg, vcprm(0, 1, s0, s1));
        } else {
            blocktemp = vec_perm(blockv, pixelsavg, vcprm(s0, s1, 2, 3));
        }

        blockv = vec_avg(blocktemp, blockv);
        vec_st(blockv, 0, block);

        block += line_size;
        pixels += line_size;
    }

POWERPC_PERF_STOP_COUNT(altivec_avg_pixels8_xy2_num, 1);
}

void dsputil_init_altivec(DSPContext* c)
{
    c->diff_pixels = diff_pixels_altivec;
    c->get_pixels = get_pixels_altivec;
    c->clear_block = clear_block_altivec;

    c->put_pixels_tab[0][0] = put_pixels16_altivec;
    /* the two functions do the same thing, so use the same code */
    c->put_no_rnd_pixels_tab[0][0] = put_pixels16_altivec;
    c->avg_pixels_tab[0][0] = avg_pixels16_altivec;
    c->avg_pixels_tab[1][0] = avg_pixels8_altivec;
    c->avg_pixels_tab[1][3] = avg_pixels8_xy2_altivec;
    c->put_pixels_tab[1][3] = put_pixels8_xy2_altivec;
    c->put_no_rnd_pixels_tab[1][3] = put_no_rnd_pixels8_xy2_altivec;
    c->put_pixels_tab[0][3] = put_pixels16_xy2_altivec;
    c->put_no_rnd_pixels_tab[0][3] = put_no_rnd_pixels16_xy2_altivec;

}
