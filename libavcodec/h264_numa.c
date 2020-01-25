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

#include <pthread.h>
#include "h264.h"
#include "malloc.h"

/*
* Pthread version with affinity lock for ED and MBD threads. Deprecated
*/
int av_transcode_pthread_affinity(int ifile, int ofile, int frame_width, int frame_height, h264_options *opts) {
	H264Context *h;
	pthread_t read_thr, parsenal_thr, entropy_thr, mbdec_thr, write_thr;

	h = ff_h264_decode_init(ifile, ofile, frame_width, frame_height, opts);
	timer_start = av_gettime();

	pthread_create(&read_thr, NULL, read_thread, h);
	pthread_create(&parsenal_thr, NULL, parsenal_thread, h);
	pthread_create(&entropy_thr, NULL, entropy_IPB_thread, h);
	pthread_create(&mbdec_thr, NULL, mbdec_thread, h);
	pthread_create(&write_thr, NULL, write_thread, h);


	pthread_join(read_thr, NULL);
	pthread_join(parsenal_thr, NULL);
	pthread_join(entropy_thr, NULL);
	pthread_join(mbdec_thr, NULL);
	pthread_join(write_thr, NULL);

	/* finished ! */
	ff_h264_decode_end(h);

	return 0;
}
