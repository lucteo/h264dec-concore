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

#ifndef H264_TYPES_H
#define H264_TYPES_H

#include "config.h"
#ifdef HAVE_LIBSDL2
#include <SDL2/SDL.h>
#endif

#include <pthread.h>
#include "avcodec.h"
#include "cabac.h"
#include "h264_dsp.h"
#include "h264_pred.h"
#include "get_bits.h"


#define MAX_REF_PIC_COUNT 16
#define MAX_DELAYED_PIC_COUNT 16

#define MAX_THREADS 80

//#define MAX_PIC_COUNT (4*(MAX_REF_PIC_COUNT+MAX_DELAYED_PIC_COUNT))

#define DPB_SIZE 33


//potsdam machine 8xX7560 without HT
// static int edb_affinity [16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
// static int edip_affinity[8] =  {16, 17, 18, 19, 20, 21, 22, 23};
//
// static int mbd_affinity[8][5] = {	{24, 32, 40, 48, 56},
// 							{25, 33, 41, 49, 57},
// 							{26, 34, 42, 50, 58},
// 							{27, 35, 43, 51, 59},
// 							{28, 36, 44, 52, 60},
// 							{29, 37, 45, 53, 61},
// 							{30, 38, 46, 54, 62},
// 							{31, 39, 47, 55, 63}, };

// static int edb_affinity [22] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 58, 59, 60, 61 ,62, 63};
// static int edip_affinity[10] =  {16, 17, 18, 19, 20, 21, 22, 23, 56, 57 };
//
// static int mbd_affinity[8][5] = {	{24, 32, 40, 48, 56},
// 							{25, 33, 41, 49, 57},
// 							{26, 34, 42, 50, 58},
// 							{27, 35, 43, 51, 59},
// 							{28, 36, 44, 52, 60},
// 							{29, 37, 45, 53, 61},
// 							{30, 38, 46, 54, 62},
// 							{31, 39, 47, 55, 63}, };
// //4 socket
// static int edip_affinity[5] = {0, 1, 2, 3, 56};
// static int edb_affinity [12] = {8, 9, 10, 11, 16, 17, 18, 19, 59, 58, 57, 51};
//
// static int mbd_affinity[4][5] = { {24, 32, 40, 48, 56},
// {25, 33, 41, 49, 57},
// {26, 34, 42, 50, 58},
// {27, 35, 43, 51, 59}, };

// static int edip_affinity[3] = {0, 1, 49};
// static int edb_affinity [6] = {8, 9, 16, 17, 56, 57};
//
// static int mbd_affinity[2][5] = { {24, 32, 40, 48, 56},
// {25, 33, 41, 49, 57}};

// static int edip_affinity[2] = {0, 8};
// static int edb_affinity [3] = {16, 24, 56};
//
// static int mbd_affinity[1][4] = { {32, 40, 48, 56},
// };

/// for ducks_take_off_2160p
// static int edip_affinity[2] = {0, 8};
// static int edb_affinity [3] = {16, 24, 32};
//
// static int mbd_affinity[1][4] = {{ 40, 48, 56, 32}};

// static int edip_affinity[3] = {0, 1, 57};
// static int edb_affinity [7] = {8, 9, 16, 17, 24, 25, 56};
//
// static int mbd_affinity[2][4] = { {32, 40, 48, 56},
// {33, 41, 49, 57}};

//4 socket
// static int edip_affinity[6]  = {0, 1, 2, 3, 59};
// static int edb_affinity [14] = {8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 58, 57};
//
// static int mbd_affinity[4][4] = { {32, 40, 48, 56},
// {33, 41, 49, 57},
// {34, 42, 50, 58},
// {35, 43, 51, 59}, };


// static int edb_affinity [29] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 59, 60, 61, 62, 63};
// static int edip_affinity[11] =  {24, 25, 26, 27, 28, 29, 30, 31, 63, 62, 61};
//
// static int mbd_affinity[8][4] = {{32, 40, 48, 56},
// 							{33, 41, 49, 57},
// 							{34, 42, 50, 58},
// 							{35, 43, 51, 59},
// 							{36, 44, 52, 60},
// 							{37, 45, 53, 61},
// 							{38, 46, 54, 62},
// 							{39, 47, 55, 63}, };

//potsdam machine 4xX7550 with HT
// int edip_affinity[16] = {0, 8, 16, 24, 	1, 9, 17, 25, 	2, 10, 18, 26,	3, 11, 19, 27 };
// int edb_affinity [16] = {1, 9, 17, 25, 	2, 10, 18, 26, 	6, 14, 22, 30,	7, 15, 23, 31 };
// int edip_affinity[16] = {58, 50, 42, 34, 	1, 9, 17, 25, 	2, 10, 18, 26,	3, 11, 19, 27 };
// int edb_affinity [16] = {57, 49, 41, 33, 	56, 48, 40, 32, 	6, 14, 22, 30,	7, 15, 23, 31 };
// //int edb_affinity [16] = {4, 12, 20, 28, 5, 13, 21, 29, 	6, 14, 22, 30,	7, 15, 23, 31 };
// //mb threads affinity on logical cores moving back to keep inteference with ed threads low
// int mbd_affinity[4][8] = {	{63, 62, 61, 60, 59, 58, 57, 56},
// 							{55, 54, 53, 52, 51, 50, 49, 48},
// 							{47, 46, 45, 44, 43, 42, 41, 40},
// 							{39, 38, 37, 36, 35, 34, 33, 32},
// 							};


// static int edip_affinity[2] = {0, 2};
// static int edb_affinity [4] = {1, 3, 2, 5};
//
// static int mbd_affinity[1][4] = {{ 4, 6, 7, 5}};

enum{
    PARSE=0,
    ENTROPY,
    REORDER,
    REORDER2,   //second mutex-cond pair used in reorder_thread
    MBDEC,
    OUTPUT,
    STAGES
};

//adhoc for profiling
enum{
    TOTAL=0,
    FRONT,
    ED,
    REC,
    PROFILE_STAGES
};

/* bit input */
/* buffer, buffer_end and size_in_bits must be present and used by every reader */

/* frame parsing */
typedef struct ParserContext {
    //int64_t offset;      ///< byte offset from starting packet start
    int ifile;
    int ofile;
    int buffer_size;
    int eof_reached;

    uint8_t *data;
    int   size;
    uint8_t *cur_ptr;
    int cur_len;

    int64_t frame_offset; /* offset of the current frame */
    int64_t cur_offset; /* current offset (incremented by each av_parser_parse()) */
    int64_t next_frame_offset; /* offset of the next frame */
    int pict_type;
    int repeat_pict;     //frame_duration = (1 + repeat_pict) * time_base. It is used by codecs like H.264 to display telecined material.
    int key_frame;  //Set by parser to 1 for key frames and 0 for non-key frames.
    int64_t pos;     // Byte position of currently parsed frame in stream.
    int64_t last_pos;  //Previous frame byte position.
    int final_frame;

    uint8_t overread[5];
    int overread_cnt;           ///< the number of bytes which where irreversibly read from the next frame
    int index;
    int last_index;
    int frame_start_found;
    uint32_t state;             ///< contains the last few bytes in MSB order
} ParserContext;

typedef struct NalContext {

    SPS *sps_buffers[MAX_SPS_COUNT];
    PPS *pps_buffers[MAX_PPS_COUNT];
    SPS sps; ///< current sps

    PictureInfo picture[16 + 1];  ///< Ref pic buffer used for deriving lists. Later linked with pic in dpb.
    PictureInfo *release_ref[MAX_MMCO_COUNT];
    PictureInfo *short_ref[32];
    PictureInfo *long_ref[32];
    int long_ref_count;  ///< number of actual long term references
    int short_ref_count; ///< number of actual short term references

    //POC stuff
    uint32_t coded_pic_num;
    int poc_lsb;
    int poc_msb;
    uint32_t poc_offset;
    int delta_poc;
    int frame_num;
    int prev_poc_msb;             ///< poc_msb of the last reference pic for POC type 0
    int prev_poc_lsb;             ///< poc_lsb of the last reference pic for POC type 0
    int frame_num_offset;         ///< for POC type 2
    int prev_frame_num_offset;    ///< for POC type 2
    int prev_frame_num;           ///< frame_num of the last pic for POC type 1/2

    int max_pic_num;
    int redundant_pic_count;
    int outputed_poc;
    int ip_id;
//   int b8_stride;             ///< 2*mb_width+1 used for some 8x8 block arrays to allow simple addressing
    int b4_stride;             ///< 4*mb_width+1 used for some 4x4 block arrays to allow simple addressing
    int mb_stride;             ///< mb_width+1 used for some arrays to allow simple addressing of left & top MBs without sig11
    int mb_width;
    int mb_height;
    int width;
    int height;

    int has_b_frames;
    //pic_struct in picture timing SEI message
    SEI_PicStructType sei_pic_struct;
    // Bit set of clock types for fields/frames in picture timing SEI message. For each found ct_type, appropriate bit is set (e.g., bit 1 for interlaced).
    int sei_ct_type;
    // dpb_output_delay in picture timing SEI message, see H.264 C.2.2
    int sei_dpb_output_delay;
    //cpb_removal_delay in picture timing SEI message, see H.264 C.1.2
    int sei_cpb_removal_delay;
    //recovery_frame_cnt from SEI message
    int sei_recovery_frame_cnt;
    // Timestamp stuff
    int sei_buffering_period_present;  ///< Buffering period SEI flag
    int initial_cpb_removal_delay[32]; ///< Initial timestamps for CPBs

} NalContext;

typedef struct EntropyContext{
    CABACContext c;

    H264Mb *m;
    int top_cbp;
    int left_cbp;
    int neighbor_transform_size; //number of neighbors (top and/or left) that used 8x8 dct

    uint32_t top_type;
    uint32_t left_type;
    uint32_t topright_type;
    uint32_t topleft_type;

    int curr_qscale;
    int chroma_qp[2]; //QPc
    int last_qscale_diff;

    uint32_t dequant4_buffer[6][52][16];
    uint32_t dequant8_buffer[2][52][64];
    uint32_t (*dequant4_coeff[6])[16];
    uint32_t (*dequant8_coeff[2])[64];

//     uint8_t (*non_zero_count_top)[32];
//     uint8_t (*non_zero_count)[32];
//     uint8_t (*non_zero_count_row[2])[32];

    uint8_t (*non_zero_count_top)[8];
    uint8_t (*non_zero_count)[8];
    uint8_t (*non_zero_count_row[2])[8];
    DECLARE_ALIGNED(8, uint8_t, non_zero_count_left[8]);

    uint8_t (*mvd_top[2])[2];
    uint8_t (*mvd[2])[2];
    uint8_t (*mvd_table[2][2])[2];

    uint8_t *direct_top;
    uint8_t *direct;
    uint8_t *direct_table[2];

    uint8_t *chroma_pred_mode_top;
    uint8_t *chroma_pred_mode;
    uint8_t *chroma_pred_mode_table[2];

    uint16_t *cbp_top;
    uint16_t *cbp;
    uint16_t *cbp_table[2];

    int8_t *qscale_top;
    int8_t *qscale;
    int8_t *qscale_table[2];

    int8_t *ref_index_top[2];
    int8_t *ref_index[2];
    int8_t *ref_index_table[2][2];

    uint32_t *mb_type_top;
    uint32_t *mb_type;
    uint32_t *mb_type_table[2];

    int b_stride;
    int mb_stride;
    int mb_width;
    int mb_height;

    uint8_t *zigzag_scan;
    uint8_t *zigzag_scan8x8;
    uint8_t direct_cache[5*8];

    DECLARE_ALIGNED(8, int8_t, intra4x4_pred_mode_cache[5*8]);
    DECLARE_ALIGNED(16, int16_t, mv_cache)[2][5*8][2];
    DECLARE_ALIGNED(8, int8_t, ref_cache)[2][5*8];
    DECLARE_ALIGNED(8, uint8_t, non_zero_count_cache)[6*8];
    DECLARE_ALIGNED(16, uint8_t, mvd_cache)[2][5*8][2];

} EntropyContext;

typedef struct H264Slice {
    PPS pps;                   ///< current pps
    PictureInfo* current_picture_info;
    DecodedPicture* curr_pic;
    int slice_num;

    int release_ref_cpn[MAX_MMCO_COUNT];
    int release_cnt;

    int qp_thresh;      ///< QP threshold to skip loopfilter
    int use_weight;
    int use_weight_chroma;
    int luma_log2_weight_denom;
    int chroma_log2_weight_denom;

    int16_t luma_weight[16][2][2];
    int16_t chroma_weight[16][2][2][2];
    int16_t implicit_weight[16][16][2];

    //poc number of ref_list int ref_poc[2][16]
    //In edslice this must becom Picture Info
    int ref_list_cpn[2][16];
    PictureInfo *ref_list[2][16];         ///Reordered version of default_ref_list according to picture reordering in slice header
    DecodedPicture *dp_ref_list[2][16];
    int ref_count[2];   ///< counts frames or fields, depending on current mb mode

    int slice_type;
    int slice_type_nos;
    int slice_alpha_c0_offset;
    int slice_beta_offset;
    int direct_8x8_inference_flag;

    uint8_t list_count;
    uint32_t coded_pic_num;

    int poc;
    int key_frame;
    int mmco_reset; //FIXME not used?

    ///stuff only needed for nal/entropy decoding
//     H264Mb *m;
//     GetBitContext *gb;
    int ip_id;
    int transform_bypass;
    int direct_spatial_mv_pred;
    int map_col_to_list0[2][16];
    int dist_scale_factor[16];

    int cabac_init_idc;
    int nal_ref_idc;
    int nal_unit_type;

    int ref2frm[2][64];  ///< reference to frame number lists, the first 2 are for -2,-1

    int qscale;

} H264Slice;

typedef struct {
    H264Slice slice;
    H264Mb *mbs;
    DecodedPicture *dp;
    GetBitContext gb;

    int lines_taken;
    int lines_total;
    int state;       // 0 free, 1 in use //1 wait for entropy, 2 wait for reconstruct.
    int initialized;

    int64_t deadline;
    int droppable;
    int drop;
} SliceBufferEntry;

typedef struct RingLineEntry{
    union{
    DECLARE_ALIGNED(64, volatile int32_t, mb_cnt);
    DECLARE_ALIGNED(64, int32_t, pad[16]);
    };
    SliceBufferEntry *sbe;
    int id;
    int line;
    TopBorder *top;
    struct RingLineEntry *prev_line;

} RingLineEntry;

// #if OMPSS
typedef struct SuperMBTask{
    int smb_x;
    int smb_y;
} SuperMBTask;

typedef struct SuperMBContext{
    int nsmb_width;             //number of super macroblocks in picture width
    int nsmb_height;            //number of super macroblocks in picture height
    int nsmb_3dheight;          //number of super macroblocks in picture height - max motion vertical vector
    int smb_width;              //width of a super macroblock
    int smb_height;             //height of a super macroblock
    int refcount;
    int index;
    SuperMBTask *smbs[2];
} SuperMBContext;
// #endif

//scratchpad for decoding a macroblock
typedef struct MBRecState{
    int8_t *ref_index_top[2];
    int8_t *ref_index[2];
    int16_t (*motion_val_top[2])[2];
    int16_t (*motion_val[2])[2];
    uint32_t *mb_type_top;
    uint32_t *mb_type;

    int8_t *list1_ref_index[2];
    int16_t (*list1_motion_val[2])[2];
    uint32_t *list1_mb_type;

    int8_t *intra4x4_pred_mode_top;
    int8_t *intra4x4_pred_mode;
#if !OMPSS
    int8_t intra4x4_pred_mode_left[4];
#endif
    int8_t *non_zero_count_top;
    int8_t *non_zero_count;
//     int8_t non_zero_count_left[8];


    unsigned int topleft_samples_available;
    unsigned int topright_samples_available;
    unsigned int top_samples_available;
    unsigned int left_samples_available;

    int top_type;
    int left_type;

    DECLARE_ALIGNED(8, int8_t, intra4x4_pred_mode_cache[5*8]);
    DECLARE_ALIGNED(16, int16_t, mv_cache)[2][5*8][2];
    DECLARE_ALIGNED(8, int8_t, ref_cache)[2][5*8];
    DECLARE_ALIGNED(8, uint8_t, non_zero_count_cache)[6*8];
    DECLARE_ALIGNED(16, uint8_t, mvd_cache)[2][5*8][2];

    DECLARE_ALIGNED(8, int16_t, bS)[2][4][4];
    uint8_t edges[2];

}MBRecState ;

typedef struct MBRecContext{
    DSPContext dsp;             ///< pointers for accelerated dsp functions
    H264DSPContext hdsp;
    H264PredContext hpc;

    MBRecState *mrs;
    RingLineEntry *rle;         //debug

    uint8_t *scratchpad_y;      ///implemented different on Cell
    uint8_t *scratchpad_cb;     ///implemented different on Cell
    uint8_t *scratchpad_cr;     ///implemented different on Cell

    int linesize;
    int uvlinesize;
    int mb_width;
    int mb_height;
    int mb_stride;
    int b_stride;
    int width;
    int height;

#if !OMPSS   // not used in OMPSS
    LeftBorder left;
    TopBorder *top;
    TopBorder *top_next; 	// next line top border
#endif
    /*
    .UU.YYYY
    .UU.YYYY
    .vv.YYYY
    .VV.YYYY
    */

    // block_offset[ 0..23] for frame macroblocks
    int block_offset[16+8];

} MBRecContext;

#ifdef HAVE_LIBSDL2
typedef struct SDLContext{
    int display;
    int fullscreen;
    pthread_t listen_thread;

    SDL_DisplayMode full;
    SDL_DisplayMode wind;


    SDL_Renderer *renderer;
    SDL_Rect rect;
    SDL_Rect win_rect;
    SDL_Window *window;
    double aspect;
    int win_w;
    int win_h;
    int resized;

    SDL_Texture *sbmap_texture;
    int showmap;
    int updatemap;
    int pause;

} SDLContext;
#endif

typedef struct OutputContext {
    int bit_buffer_size;
    uint8_t *bit_buffer;
    uint64_t video_size;
    int dropped_frames;
    int frame_number;
    DecodedPicture *delayed_pic[DPB_SIZE];
    int dp_cnt;

} OutputContext;

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t cond;
    SliceBufferEntry **queue;
    int size;
    int cnt;
    int fi;
    int fo;
} SliceBufferQueue;

typedef struct {
    pthread_mutex_t wslock;
    pthread_cond_t wscond;
    pthread_mutex_t swlock;
    pthread_cond_t swcond;
    RingLineEntry **queue;
    int size;
    int ready;
    int free;
    int fi;
    int fo;
} RingLineQueue;

#if HAVE_LIBSDL2
typedef struct {
    pthread_mutex_t sdl_lock;
    pthread_cond_t sdl_cond;
    SDL_Texture **queue;
    int size;
    int ready;
    int fi;
    int fo;
    int exit;
} SDLTextureQueue;
#endif
/**
* H264Context
*/
typedef struct H264Context{
    SliceBufferQueue sb_q[STAGES];
    RingLineQueue rl_q;

    pthread_mutex_t lock[STAGES];
    pthread_cond_t cond[STAGES];

    pthread_mutex_t task_lock;
    pthread_cond_t task_cond;

    pthread_attr_t ed_rec_attr[MAX_THREADS];
    pthread_t ed_rec_thr[MAX_THREADS];

    int init_threads;
    pthread_mutex_t ilock;
    pthread_cond_t icond;

    const char *file_name;
    int profile;
    int start;
    int touch_start;
    int setaff;
    int touch_done;
    int rl_side_touch;
    int statmbd;
    pthread_mutex_t slock;
    pthread_cond_t scond;
    pthread_mutex_t tlock;
    pthread_cond_t tcond;
    pthread_mutex_t tdlock;
    pthread_cond_t tdcond;

    int ed_ppe_threads;
    int threads;
    int smt;

    int acdpb_cnt;  //debug
    int reldpb_cnt;

    int sb_size;
    SliceBufferEntry *sb;               ///< Slice Syntax Buffer
    int free_sb_cnt;
    int slice_bufs;

    int max_dpb_cnt;
    DecodedPicture *dpb;       ///< Decoded Picture Buffer
    int free_dpb_cnt;

    int ifile;
    int ofile;
    int frame_width;
    int frame_height;
    int num_frames;
    int width;
    int height;
    int mb_width;
    int mb_height;
    int mb_stride;          ///< mb_width+1 used for some arrays to allow simple addressing of left & top MBs without sig11
    int b4_stride;
    int b_stride;

    int smb_height;
    int smb_width;
    pthread_mutex_t smb_lock;
    pthread_cond_t sdl_cond;
    pthread_mutex_t sdl_lock;
    SuperMBContext *smbc;

    int wave_order;
    int static_3d;
    int pipe_bufs;
    int ompss_rt;

    int controlfps;
    int64_t frametime;
    int64_t nexttime;
    int64_t framedelay;

    //shared tables used in entropy decoding
    uint8_t zigzag_scan[16];
    uint8_t zigzag_scan8x8[64];

    int verbose;
    int no_mbd;
    int display;
    int fullscreen;
    int quit;
#ifdef HAVE_LIBSDL2
    SDLTextureQueue sdlq;
    SDLContext *sdlc;
#endif

    struct timespec start_time[PROFILE_STAGES];
    struct timespec end_time[PROFILE_STAGES];
    double last_time[PROFILE_STAGES];
    double total_time[PROFILE_STAGES];

}H264Context;

#endif
