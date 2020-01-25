/*
 * DSP utils : average functions are compiled twice for 3dnow/mmx2
 * Copyright (c) 2000, 2001 Fabrice Bellard
 * Copyright (c) 2002-2004 Michael Niedermayer
 *
 * MMX optimization by Nick Kurshev <nickols_k@mail.ru>
 * mostly rewritten by Michael Niedermayer <michaelni@gmx.at>
 * and improved by Zdenek Kabelac <kabi@users.sf.net>
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

static void DEF(put_pixels8_l2)(uint8_t *dst, uint8_t *src1, uint8_t *src2, int dstStride, int src1Stride, int h)
{
    __asm__ volatile(
        "testl $1, %0                   \n\t"
            " jz 1f                     \n\t"
        "movq   (%1), %%mm0             \n\t"
        "movq   (%2), %%mm1             \n\t"
        "add    %4, %1                  \n\t"
        "add    $8, %2                  \n\t"
        PAVGB" %%mm1, %%mm0             \n\t"
        "movq   %%mm0, (%3)             \n\t"
        "add    %5, %3                  \n\t"
        "decl   %0                      \n\t"
        "1:                             \n\t"
        "movq   (%1), %%mm0             \n\t"
        "add    %4, %1                  \n\t"
        "movq   (%1), %%mm1             \n\t"
        "add    %4, %1                  \n\t"
        PAVGB" (%2), %%mm0              \n\t"
        PAVGB" 8(%2), %%mm1             \n\t"
        "movq   %%mm0, (%3)             \n\t"
        "add    %5, %3                  \n\t"
        "movq   %%mm1, (%3)             \n\t"
        "add    %5, %3                  \n\t"
        "movq   (%1), %%mm0             \n\t"
        "add    %4, %1                  \n\t"
        "movq   (%1), %%mm1             \n\t"
        "add    %4, %1                  \n\t"
        PAVGB" 16(%2), %%mm0            \n\t"
        PAVGB" 24(%2), %%mm1            \n\t"
        "movq   %%mm0, (%3)             \n\t"
        "add    %5, %3                  \n\t"
        "movq   %%mm1, (%3)             \n\t"
        "add    %5, %3                  \n\t"
        "add    $32, %2                 \n\t"
        "subl   $4, %0                  \n\t"
        "jnz    1b                      \n\t"

        :"+b"(h), "+a"(src1), "+c"(src2), "+d"(dst)
        :"S"((x86_reg)src1Stride), "D"((x86_reg)dstStride)
        :"memory");
//the following should be used, though better not with gcc ...
/*        :"+g"(h), "+r"(src1), "+r"(src2), "+r"(dst)
        :"r"(src1Stride), "r"(dstStride)
        :"memory");*/
}

static void DEF(avg_pixels8_l2)(uint8_t *dst, uint8_t *src1, uint8_t *src2, int dstStride, int src1Stride, int h)
{
    __asm__ volatile(
        "testl $1, %0                   \n\t"
            " jz 1f                     \n\t"
        "movq   (%1), %%mm0             \n\t"
        "movq   (%2), %%mm1             \n\t"
        "add    %4, %1                  \n\t"
        "add    $8, %2                  \n\t"
        PAVGB" %%mm1, %%mm0             \n\t"
        PAVGB" (%3), %%mm0              \n\t"
        "movq   %%mm0, (%3)             \n\t"
        "add    %5, %3                  \n\t"
        "decl   %0                      \n\t"
        "1:                             \n\t"
        "movq   (%1), %%mm0             \n\t"
        "add    %4, %1                  \n\t"
        "movq   (%1), %%mm1             \n\t"
        "add    %4, %1                  \n\t"
        PAVGB" (%2), %%mm0              \n\t"
        PAVGB" 8(%2), %%mm1             \n\t"
        PAVGB" (%3), %%mm0              \n\t"
        "movq   %%mm0, (%3)             \n\t"
        "add    %5, %3                  \n\t"
        PAVGB" (%3), %%mm1              \n\t"
        "movq   %%mm1, (%3)             \n\t"
        "add    %5, %3                  \n\t"
        "movq   (%1), %%mm0             \n\t"
        "add    %4, %1                  \n\t"
        "movq   (%1), %%mm1             \n\t"
        "add    %4, %1                  \n\t"
        PAVGB" 16(%2), %%mm0            \n\t"
        PAVGB" 24(%2), %%mm1            \n\t"
        PAVGB" (%3), %%mm0              \n\t"
        "movq   %%mm0, (%3)             \n\t"
        "add    %5, %3                  \n\t"
        PAVGB" (%3), %%mm1              \n\t"
        "movq   %%mm1, (%3)             \n\t"
        "add    %5, %3                  \n\t"
        "add    $32, %2                 \n\t"
        "subl   $4, %0                  \n\t"
        "jnz    1b                      \n\t"

        :"+b"(h), "+a"(src1), "+c"(src2), "+d"(dst)
        :"S"((x86_reg)src1Stride), "D"((x86_reg)dstStride)
        :"memory");
//the following should be used, though better not with gcc ...
/*        :"+g"(h), "+r"(src1), "+r"(src2), "+r"(dst)
        :"r"(src1Stride), "r"(dstStride)
        :"memory");*/
}


static void DEF(put_pixels16_l2)(uint8_t *dst, uint8_t *src1, uint8_t *src2, int dstStride, int src1Stride, int h)
{
    __asm__ volatile(
        "testl $1, %0                   \n\t"
            " jz 1f                     \n\t"
        "movq   (%1), %%mm0             \n\t"
        "movq   8(%1), %%mm1            \n\t"
        PAVGB" (%2), %%mm0              \n\t"
        PAVGB" 8(%2), %%mm1             \n\t"
        "add    %4, %1                  \n\t"
        "add    $16, %2                 \n\t"
        "movq   %%mm0, (%3)             \n\t"
        "movq   %%mm1, 8(%3)            \n\t"
        "add    %5, %3                  \n\t"
        "decl   %0                      \n\t"
        "1:                             \n\t"
        "movq   (%1), %%mm0             \n\t"
        "movq   8(%1), %%mm1            \n\t"
        "add    %4, %1                  \n\t"
        PAVGB" (%2), %%mm0              \n\t"
        PAVGB" 8(%2), %%mm1             \n\t"
        "movq   %%mm0, (%3)             \n\t"
        "movq   %%mm1, 8(%3)            \n\t"
        "add    %5, %3                  \n\t"
        "movq   (%1), %%mm0             \n\t"
        "movq   8(%1), %%mm1            \n\t"
        "add    %4, %1                  \n\t"
        PAVGB" 16(%2), %%mm0            \n\t"
        PAVGB" 24(%2), %%mm1            \n\t"
        "movq   %%mm0, (%3)             \n\t"
        "movq   %%mm1, 8(%3)            \n\t"
        "add    %5, %3                  \n\t"
        "add    $32, %2                 \n\t"
        "subl   $2, %0                  \n\t"
        "jnz    1b                      \n\t"

        :"+b"(h), "+a"(src1), "+c"(src2), "+d"(dst)

        :"S"((x86_reg)src1Stride), "D"((x86_reg)dstStride)
        :"memory");
//the following should be used, though better not with gcc ...
/*        :"+g"(h), "+r"(src1), "+r"(src2), "+r"(dst)
        :"r"(src1Stride), "r"(dstStride)
        :"memory");*/
}

static void DEF(avg_pixels16_l2)(uint8_t *dst, uint8_t *src1, uint8_t *src2, int dstStride, int src1Stride, int h)
{
    __asm__ volatile(
        "testl $1, %0                   \n\t"
            " jz 1f                     \n\t"
        "movq   (%1), %%mm0             \n\t"
        "movq   8(%1), %%mm1            \n\t"
        PAVGB" (%2), %%mm0              \n\t"
        PAVGB" 8(%2), %%mm1             \n\t"
        "add    %4, %1                  \n\t"
        "add    $16, %2                 \n\t"
        PAVGB" (%3), %%mm0              \n\t"
        PAVGB" 8(%3), %%mm1             \n\t"
        "movq   %%mm0, (%3)             \n\t"
        "movq   %%mm1, 8(%3)            \n\t"
        "add    %5, %3                  \n\t"
        "decl   %0                      \n\t"
        "1:                             \n\t"
        "movq   (%1), %%mm0             \n\t"
        "movq   8(%1), %%mm1            \n\t"
        "add    %4, %1                  \n\t"
        PAVGB" (%2), %%mm0              \n\t"
        PAVGB" 8(%2), %%mm1             \n\t"
        PAVGB" (%3), %%mm0              \n\t"
        PAVGB" 8(%3), %%mm1             \n\t"
        "movq   %%mm0, (%3)             \n\t"
        "movq   %%mm1, 8(%3)            \n\t"
        "add    %5, %3                  \n\t"
        "movq   (%1), %%mm0             \n\t"
        "movq   8(%1), %%mm1            \n\t"
        "add    %4, %1                  \n\t"
        PAVGB" 16(%2), %%mm0            \n\t"
        PAVGB" 24(%2), %%mm1            \n\t"
        PAVGB" (%3), %%mm0              \n\t"
        PAVGB" 8(%3), %%mm1             \n\t"
        "movq   %%mm0, (%3)             \n\t"
        "movq   %%mm1, 8(%3)            \n\t"
        "add    %5, %3                  \n\t"
        "add    $32, %2                 \n\t"
        "subl   $2, %0                  \n\t"
        "jnz    1b                      \n\t"

        :"+b"(h), "+a"(src1), "+c"(src2), "+d"(dst)
        :"S"((x86_reg)src1Stride), "D"((x86_reg)dstStride)
        :"memory");
//the following should be used, though better not with gcc ...
/*        :"+g"(h), "+r"(src1), "+r"(src2), "+r"(dst)
        :"r"(src1Stride), "r"(dstStride)
        :"memory");*/
}

static void DEF(avg_pixels8)(uint8_t *block, const uint8_t *pixels, int line_size, int h)
{
    __asm__ volatile(
        "lea (%3, %3), %%"REG_a"        \n\t"
        "1:                             \n\t"
        "movq (%2), %%mm0               \n\t"
        "movq (%2, %3), %%mm1           \n\t"
        PAVGB" (%1), %%mm0              \n\t"
        PAVGB" (%1, %3), %%mm1          \n\t"
        "movq %%mm0, (%2)               \n\t"
        "movq %%mm1, (%2, %3)           \n\t"
        "add %%"REG_a", %1              \n\t"
        "add %%"REG_a", %2              \n\t"
        "movq (%2), %%mm0               \n\t"
        "movq (%2, %3), %%mm1           \n\t"
        PAVGB" (%1), %%mm0              \n\t"
        PAVGB" (%1, %3), %%mm1          \n\t"
        "add %%"REG_a", %1              \n\t"
        "movq %%mm0, (%2)               \n\t"
        "movq %%mm1, (%2, %3)           \n\t"
        "add %%"REG_a", %2              \n\t"
        "subl $4, %0                    \n\t"
        "jnz 1b                         \n\t"
        :"+g"(h), "+S"(pixels), "+D"(block)
        :"r" ((x86_reg)line_size)
        :"%"REG_a, "memory");
}
