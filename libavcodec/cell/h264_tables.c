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

#include <stdint.h>
#include "h264_tables.h"

uint8_t ff_cropTbl[256+2 *MAX_NEG_CROP] = {0, };

int block_offset[16+4+4];

void ff_cropTbl_init(){
    int i;
    for(i=0;i<256;i++) ff_cropTbl[i + MAX_NEG_CROP] = i;
    for(i=0;i<MAX_NEG_CROP;i++) {
        ff_cropTbl[i] = 0;
        ff_cropTbl[i + MAX_NEG_CROP + 256] = 255;
    }
}

void init_block_offset(int linesize, int uvlinesize){
	int i;
	for(i=0; i<16; i++){
        block_offset[i]= 4*((scan8[i] - scan8[0])&7) + 4*linesize*((scan8[i] - scan8[0])>>3);
    }
    for(i=0; i<4; i++){
        block_offset[16+i]=
        block_offset[20+i]= 4*((scan8[i] - scan8[0])&7) + 4*uvlinesize*((scan8[i] - scan8[0])>>3);
    }
}