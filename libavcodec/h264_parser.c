/*
 * H.26L/H.264/AVC/JVT/14496-10/... parser
 * Copyright (c) 2003 Michael Niedermayer <michaelni@gmx.at>
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
 * H.264 / AVC / MPEG4 part10 parser.
 * @author Michael Niedermayer <michaelni@gmx.at>
 */

#include <unistd.h>

#include "golomb.h"
#include "libavutil/error.h"
#include "h264_types.h"

#undef NDEBUG
#include <assert.h>

#define END_NOT_FOUND (-100)

static int ff_h264_find_frame_end(ParserContext *s, const uint8_t *buf, int buf_size)
{
    int i;
    uint32_t state;

    state= s->state;
    if(state>13)
        state= 7;

    for(i=0; i<buf_size; i++){
        if(state==7){
        /* we check i<buf_size instead of i+3/7 because its simpler
         * and there should be FF_INPUT_BUFFER_PADDING_SIZE bytes at the end
         */
            while(i<buf_size && !((~*(const uint64_t*)(buf+i) & (*(const uint64_t*)(buf+i) - 0x0101010101010101ULL)) & 0x8080808080808080ULL))
                i+=8;

            for(; i<buf_size; i++){
                if(!buf[i]){
                    state=2;
                    break;
                }
            }
        }else if(state<=2){
            if(buf[i]==1)   state^= 5; //2->7, 1->4, 0->5
            else if(buf[i]) state = 7;
            else            state>>=1; //2->1, 1->0, 0->0
        }else if(state<=5){
            int v= buf[i] & 0x1F;
            if(v==6 || v==7 || v==8 || v==9){
                if(s->frame_start_found){
                    i++;
                    goto found;
                }
            }else if(v==1 || v==2 || v==5){
                if(s->frame_start_found){
                    state+=8;
                    continue;
                }else
                    s->frame_start_found = 1;
            }
            state= 7;
        }else{
            if(buf[i] & 0x80)
                goto found;
            state= 7;
        }
    }
    s->state= state;
    return END_NOT_FOUND;

found:
    s->state=7;
    s->frame_start_found= 0;
    return i-(state&5);
}

static int ff_combine_frame(ParserContext *s, GetBitContext *gb, int next, uint8_t **buf, int *buf_size)
{
    int i;
    /* Copy overread bytes from last frame into buffer. */
    for(i =0; s->overread_cnt>0; s->overread_cnt--, i++){
        gb->raw[s->index++]= s->overread[i];
    }

    /* EOF - END_NOT_FOUND means no next frame start is found in current partial read. If buf_size of the partial read is 0 we are at EOF */
    if(!*buf_size && next == END_NOT_FOUND){
        next= 0;
    }
    s->last_index= s->index;

    /* copy into buffer end return */
    if(next == END_NOT_FOUND){
        gb->raw = av_fast_realloc(gb->raw, &gb->alloc_size, (*buf_size) + s->index + FF_INPUT_BUFFER_PADDING_SIZE);
        memcpy(&gb->raw[s->index], *buf, *buf_size);
        s->index += *buf_size;
        return -1;
    }

    ///end found
    *buf_size=  s->index + next;
    /* append to buffer */

    gb->raw = av_fast_realloc(gb->raw, &gb->alloc_size, next + s->index + FF_INPUT_BUFFER_PADDING_SIZE);
    memcpy(&gb->raw[s->index], *buf, next + FF_INPUT_BUFFER_PADDING_SIZE );
    s->index = 0;

    /* store overread bytes */
    for(i=0; next < 0; next++, i++){
        s->state = (s->state<<8) | gb->raw[s->last_index + next];
        s->overread[i] = gb->raw[s->last_index + next];
        s->overread_cnt++;
    }

    return 0;
}

static int h264_parse(ParserContext *s, GetBitContext *gb,
                      uint8_t *buf, int buf_size)
{
    int next;

    next= ff_h264_find_frame_end(s, buf, buf_size);

    if (ff_combine_frame(s, gb, next, &buf, &buf_size) < 0) {
        gb->buf_size = 0;
        return buf_size;
    }

    if(next<0 && next != END_NOT_FOUND){
        assert(s->last_index + next >= 0 );
        ff_h264_find_frame_end(s, &gb->raw[s->last_index + next], -next); //update state
    }

    gb->buf_size = buf_size;
    return next;
}

static int ff_raw_read_partial_packet(ParserContext *pc)
{
    int len= -1;

    if (!pc->eof_reached){
        len = read( pc->ifile, pc->data, pc->buffer_size);
//         printf("read task %d\t%d\n", pc->ifile, len); fflush(NULL);
        if (len < pc->buffer_size) {
            pc->eof_reached = 1;
        }
    }

    return len;
}

void av_read_frame_internal(ParserContext *pc, GetBitContext *gb){
    int len;
    uint8_t dummy_buf[FF_INPUT_BUFFER_PADDING_SIZE]={0};
    av_fast_malloc(&gb->raw, &gb->alloc_size, 2048+FF_INPUT_BUFFER_PADDING_SIZE);

    //Parsing is performed before read, since there are ussually leftovers from parsing the previous frame.
    for(;;) {
        if (pc->cur_len>0){
            len = h264_parse(pc, gb, pc->cur_ptr, pc->cur_len);
            if (len<0)
                len =0;
            //* increment read pointer */
            pc->cur_ptr += len;
            pc->cur_len -= len;

            if (gb->buf_size) {
                break;
            }
        }

        //check for ret and not parser->eof_reached as one "read" can contain more than 1 frame
        pc->size= ff_raw_read_partial_packet(pc);
        if (pc->size < 0) {
            pc->final_frame =1;
            /* return the last frames, if any */
            h264_parse(pc, gb, dummy_buf, 0);
            break;
        }
        pc->cur_ptr = pc->data;
        pc->cur_len = pc->size;
    }

    assert(gb->raw!=NULL);

}

ParserContext *get_parse_context(int ifile){
    ParserContext *pc = av_mallocz(sizeof(ParserContext));
    pc->buffer_size = 2048;
    pc->final_frame = 0;
    pc->cur_len= 0;
    pc->data = av_mallocz(2048 + FF_INPUT_BUFFER_PADDING_SIZE);
    pc->size = 2048;
    pc->eof_reached =0;
    pc->ifile = ifile;

    return pc;
}

void free_parse_context(ParserContext *pc){
    av_free(pc->data);
    av_free(pc);
}
