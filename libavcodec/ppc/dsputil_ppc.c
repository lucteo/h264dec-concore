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

#include "libavcodec/dsputil.h"
#include "dsputil_ppc.h"
#include "dsputil_altivec.h"

static void prefetch_ppc(void *mem, int stride, int h)
{
    register const uint8_t *p = mem;
    do {
        __asm__ volatile ("dcbt 0,%0" : : "r" (p));
        p+= stride;
    } while(--h);
}

void dsputil_init_ppc(DSPContext* c)
{
    c->prefetch = prefetch_ppc;

#if HAVE_ALTIVEC
	dsputil_h264_init_ppc(c);	
	dsputil_init_altivec(c);

	c->idct_put = idct_put_altivec;
	c->idct_add = idct_add_altivec;

#endif /* HAVE_ALTIVEC */
}
