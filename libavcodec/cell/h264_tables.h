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

#ifndef H264_TABLES_H
#define H264_TABLES_H

#define MAX_NEG_CROP       1024

extern uint8_t ff_cropTbl[256+2 *MAX_NEG_CROP];
extern int block_offset[16+4+4];

static const uint8_t scan8[16 + 2*4]={
	4+1*8, 5+1*8, 4+2*8, 5+2*8,
	6+1*8, 7+1*8, 6+2*8, 7+2*8,
	4+3*8, 5+3*8, 4+4*8, 5+4*8,
	6+3*8, 7+3*8, 6+4*8, 7+4*8,
	1+1*8, 2+1*8,
	1+2*8, 2+2*8,
	1+4*8, 2+4*8,
	1+5*8, 2+5*8,
};

static const uint8_t ff_zigzag_direct[64] = {
    0,   1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

static const uint8_t zigzag_scan[16]={
 0+0*4, 1+0*4, 0+1*4, 0+2*4,
 1+1*4, 2+0*4, 3+0*4, 2+1*4,
 1+2*4, 0+3*4, 1+3*4, 2+2*4,
 3+1*4, 3+2*4, 2+3*4, 3+3*4,
};

static const uint8_t luma_dc_zigzag_scan[16]={
 0*16 + 0*64, 1*16 + 0*64, 2*16 + 0*64, 0*16 + 2*64,
 3*16 + 0*64, 0*16 + 1*64, 1*16 + 1*64, 2*16 + 1*64,
 1*16 + 2*64, 2*16 + 2*64, 3*16 + 2*64, 0*16 + 3*64,
 3*16 + 1*64, 1*16 + 3*64, 2*16 + 3*64, 3*16 + 3*64,
};

static const uint8_t chroma_dc_scan[4]={
 (0+0*2)*16, (1+0*2)*16,
 (0+1*2)*16, (1+1*2)*16,  //FIXME
};

static const uint8_t rem6[52]={
0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
};

static const uint8_t div6[52]={
0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8,
};

static const uint8_t dequant4_coeff_init[6][3]={
  {10,13,16},
  {11,14,18},
  {13,16,20},
  {14,18,23},
  {16,20,25},
  {18,23,29},
};

static const uint8_t dequant8_coeff_init_scan[16] = {
  0,3,4,3, 3,1,5,1, 4,5,2,5, 3,1,5,1
};
static const uint8_t dequant8_coeff_init[6][6]={
  {20,18,32,19,25,24},
  {22,19,35,21,28,26},
  {26,23,42,24,33,31},
  {28,25,45,26,35,33},
  {32,28,51,30,40,38},
  {36,32,58,34,46,43},
};


void init_block_offset(int linesize, int uvlinesize);
void ff_cropTbl_init();

#endif
