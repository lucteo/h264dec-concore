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


#ifndef H264_DMA_H
#define H264_DMA_H

#include "libavutil/mem.h"

typedef struct dma_list_elem {
	union {
		unsigned int all32;
		struct {
		unsigned int stall    : 1;
		unsigned int reserved : 15;
		unsigned int nbytes   : 16;
		} bits;
	} size;
	uint64_t ea_low : 32;
}dma_list_elem_t;

extern DECLARE_ALIGNED_16(dma_list_elem_t, put_list_buf[2*(52+26+26)]);
extern dma_list_elem_t* put_list;

extern DECLARE_ALIGNED_16(dma_list_elem_t, get_list_buf[16*(4+5 + 2*3)]);
extern dma_list_elem_t* get_list;

enum{
	MBD_slice=1,
	MBD_buf1,
	MBD_buf2,
	MBD_buf3,
	MBD_put,
	MBD_pic,
	MBD_mc_buf1,
	MBD_mc_buf2
};

enum{
	ED_spe=1,
	ED_slice,
	ED_raw,
	ED_get,
	ED_get2,
	ED_get_mv,
	ED_put,
	ED_putmb0,
	ED_putmb1,
};

// Functions to get/put a block from/to main memory
void get_dma_list(void *dst, void* ea, unsigned int w, unsigned int h, unsigned int stride, unsigned int tag, int barrier);
void put_dma_list(void *src, void* ea, unsigned int size, unsigned int h, unsigned int stride, unsigned int tag);

//Functions to do a dma transfer for 32-bit
void spu_dma_get(void *ls, unsigned ea, int size, int tag);
void spu_dma_put(void *ls, unsigned ea, int size, int tag);
void spu_dma_barrier_put(void *ls, unsigned ea, int size, int tag);

// Function that wait to finish a DMA transfer with especific id
void wait_dma_id(int id);

#endif
