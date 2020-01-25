/*
 * utils for libavcodec
 * Copyright (c) 2001 Fabrice Bellard
 * Copyright (c) 2002-2004 Michael Niedermayer <michaelni@gmx.at>
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

/**
 * @file
 * utils.
 */

/* needed for mkstemp() */
#define _XOPEN_SOURCE 600

#include "avcodec.h"
#include "dsputil.h"

#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>
#include <float.h>
//#undef NDEBUG
#include <assert.h>

#include <fcntl.h>

void *av_fast_realloc(void *ptr, unsigned int *size, unsigned int min_size)
{
    if(min_size < *size)
        return ptr;

    *size= FFMAX(17*min_size/16 + 32, min_size);

    ptr= av_realloc(ptr, *size);
    if(!ptr) //we could set this to the unmodified min_size but this is safer if the user lost the ptr and uses NULL now
        *size= 0;

    return ptr;
}

void av_fast_malloc(void *ptr, unsigned int *size, unsigned int min_size)
{
    void **p = ptr;
    if (min_size < *size)
        return;
    *size= FFMAX(17*min_size/16 + 32, min_size);
    av_free(*p);
    *p = av_malloc(*size);
    if (!*p) *size = 0;
}


