/*
 * Copyright 2021 Lucian Radu Teodorescu
 *
 * Based on the existing Starbench code, by TU Berlin
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

// #define TRACY_ENABLE 1

#include <concore/spawn.hpp>
#include <concore/global_executor.hpp>
#include <concore/pipeline.hpp>
#include <concore/task_group.hpp>
#include <concore/init.hpp>
#include <concore/profiling.hpp>
#include <concore/data/concurrent_queue.hpp>

#include "h264_types.h"
#include "h264_parser.h"
#include "h264_nal.h"
#include "h264_entropy.h"
#include "h264_rec.h"
#include "h264_pred_mode.h"
#include "h264_misc.h"
#undef NDEBUG
#include <assert.h>

#include <atomic>
#include <vector>

struct atomic_int : std::atomic<int> {
    atomic_int(int val = 0)
        : atomic(val) {}
    atomic_int(const atomic_int& other)
        : atomic(other.load()) {}
};

struct process_matrix {
    using cell_fun_t = std::function<void(int x, int y)>;

    void start(int w, int h, cell_fun_t cf, concore::task&& donet) {
        width = w;
        height = h;
        cell_fun = cf;
        done_task = std::move(donet);
        ref_counts.resize(h * w);
        for ( int y=0; y<h; y++ ) {
            for ( int x=0; x<w; x++ )
                ref_counts[y*w + x].store( x==0 || y==0 || x==w-1 ? 1 : 2 );
        }

        // Start with the first cell
        concore::spawn(create_cell_task(0, 0));
    }

private:
    int width;
    int height;
    cell_fun_t cell_fun;
    concore::task done_task;
    std::vector<atomic_int> ref_counts;

    concore::task create_cell_task(int x, int y) {
        auto f = [this, x, y] { cell_fun(x, y); };
        auto cont = [this, x, y] (std::exception_ptr) {
            // Spawn bottom task
            if (y < height - 1 && x > 0)
                unblock_cell(x - 1, y + 1);
            // Spawn right task
            if (x < width - 1)
                unblock_cell(x + 1, y, false);
            // Finish?
            if (y == height-1 && x == width-1 )
                concore::spawn(std::move(done_task), false);
        };
        return concore::task{f, {}, cont};
    }
    void unblock_cell(int x, int y, bool wake_workers = true) {
        int idx = y * width + x;
        if (ref_counts[idx]-- == 1)
            concore::spawn(create_cell_task(x, y), wake_workers);
    }
};

using MBRecCtxStash = concore::concurrent_queue<MBRecContext*>;

static constexpr int mb_line_chunk_size = 5;   // mb's per chunk

//! Holds all the data to decode a frame.
//! This will be used on all the stages of the pipeline.
struct FrameData {
    H264Slice slice;                                  // all phases
    GetBitContext gb{0};                              // parse, entropy
    EntropyContext* ec{nullptr};                      // entropy
    H264Mb* mbs{nullptr};                             // entropy, mb
    std::vector<MBRecContext*> mb_line_dec_ctx;       // the decoding context for a MB line
    process_matrix mb_processing;                     // matrix of MB chunks to process
    int frame_idx{0};                                 // the current frame index

    explicit FrameData(H264Context* h);
    ~FrameData();
};

FrameData::FrameData(H264Context* h) {
    CONCORE_PROFILING_FUNCTION();
    memset(&slice, 0, sizeof(H264Slice));
    mbs = (H264Mb*)malloc(h->mb_height * h->mb_width * sizeof(H264Mb));
    ec = get_entropy_context(h);
}
FrameData::~FrameData() {
    CONCORE_PROFILING_FUNCTION();
    free(mbs);
    free(gb.raw);
    if (gb.rbsp)
        free(gb.rbsp);
    free_entropy_context(ec);
}

using FrameDataPtr = std::shared_ptr<FrameData>;
using FrameDataStash = concore::concurrent_queue<FrameDataPtr, concore::queue_type::single_prod_single_cons>;

struct GlobalDecContext {
    H264Context* h;
    ParserContext* pc;
    NalContext* nc;
    OutputContext* out_ctx;
    FrameDataStash frames_stash;           //! data objects used for frame decoding
    MBRecCtxStash mbrec_ctx_stash;      //! objects used while processing lines; expensive to create
};

struct DecFrame {
    const int index;
    GlobalDecContext* const global_ctx;
    FrameDataPtr frame_data{};
};

void decode_slice_mb_chunk(GlobalDecContext* gctx, FrameData& fd, int chunk_x, int line_idx) {
    CONCORE_PROFILING_SCOPE_N("mb chunk");
    auto h = gctx->h;
    int start_x = chunk_x * mb_line_chunk_size;
    int end_x = std::min((chunk_x+1) * mb_line_chunk_size, h->mb_width);
    CONCORE_PROFILING_SET_TEXT_FMT(64, "frame=%d, line=%d, start=%d", fd.frame_idx, line_idx, start_x);

    auto& slice = fd.slice;
    auto& line_ctx = fd.mb_line_dec_ctx[line_idx];

    // First time: ensure we have a mbrec context
    if (chunk_x == 0) {
        if (!gctx->mbrec_ctx_stash.try_pop(line_ctx)) {
            line_ctx = get_mbrec_context(h);
            line_ctx->top_next = line_ctx->top =
                    (TopBorder*)malloc(h->mb_width * sizeof(TopBorder));
        }
        init_mbrec_context(line_ctx, line_ctx->mrs, &slice, line_idx);
    }

    static constexpr int headstart = 10;

    for (int i = start_x; i < end_x; i++) {
        // CONCORE_PROFILING_SCOPE_N("mb pixel");
        // CONCORE_PROFILING_SET_TEXT_FMT(64, "line=%d, col=%d", line_idx, i);
        H264Mb* m = &fd.mbs[i + line_idx * line_ctx->mb_width];
        h264_decode_mb_internal(line_ctx, line_ctx->mrs, &slice, m);
    }

    if (end_x == h->mb_width) {
        draw_edges(line_ctx, &slice, line_idx);

        // Done with the line; we don't need dec_ctx anymore
        gctx->mbrec_ctx_stash.push(std::move(line_ctx));
    }
}

void stage_parse(DecFrame& frm, concore::pipeline<DecFrame>& process) {
    CONCORE_PROFILING_FUNCTION();
    CONCORE_PROFILING_SET_TEXT_FMT(32, "frame=%d", frm.index);
    
    // Acquire a free decoding frame
    bool res = frm.global_ctx->frames_stash.try_pop(frm.frame_data);
    assert(res);

    // If we are not at the end, parse the frame
    if (!frm.global_ctx->pc->final_frame && !frm.global_ctx->h->quit) {
        memset(&frm.frame_data->slice, 0, sizeof(H264Slice));
        av_read_frame_internal(frm.global_ctx->pc, &frm.frame_data->gb);
        decode_nal_units(frm.global_ctx->nc, &frm.frame_data->slice, &frm.frame_data->gb);

        // Push a new frame through our pipeline
        process.push(DecFrame{frm.index+1, frm.global_ctx});
    } else
    {
        // Stop the line by releasing the decoding frame context
        frm.global_ctx->frames_stash.push(std::move(frm.frame_data));
    }
}

void stage_decode_slice_entropy(DecFrame& frm) {
    CONCORE_PROFILING_FUNCTION();
    CONCORE_PROFILING_SET_TEXT_FMT(32, "frame=%d", frm.index);
    if (!frm.frame_data)
        return;

    auto& fd = *frm.frame_data;
    int i, j;
    H264Slice *s = &fd.slice;
    GetBitContext *gb = &fd.gb;
    EntropyContext *ec = fd.ec;
    CABACContext *c = &ec->c;
    H264Mb *mbs = fd.mbs;

    if( !s->pps.cabac ){
        av_log(AV_LOG_ERROR, "Only cabac encoded streams are supported\n");
        return;
    }

    init_dequant_tables(s, ec);
    ec->curr_qscale = s->qscale;
    ec->last_qscale_diff = 0;
    ec->chroma_qp[0] = get_chroma_qp( s, 0, s->qscale);
    ec->chroma_qp[1] = get_chroma_qp( s, 1, s->qscale);

    /* realign */
    align_get_bits( gb );
    /* init cabac */
    ff_init_cabac_decoder( c, gb->buffer + get_bits_count(gb)/8, (get_bits_left(gb) + 7)/8);

    ff_h264_init_cabac_states(ec, s, c);

    for(j=0; j<ec->mb_height; j++){
        CONCORE_PROFILING_SCOPE_N("line");
        CONCORE_PROFILING_SET_TEXT_FMT(32, "line=%d", j);
        init_entropy_buf(ec, s, j);
        for(i=0; i<ec->mb_width; i++){
            int eos,ret;
            H264Mb *m = &mbs[i + j*ec->mb_width];
            //memset(m, 0, sizeof(H264Mb));
            m->mb_x=i;
            m->mb_y=j;
            ec->m = m;

            ret = ff_h264_decode_mb_cabac(ec, s, c);
            eos = get_cabac_terminate( c); (void) eos;

            if( ret < 0 || c->bytestream > c->bytestream_end + 2) {
                av_log(AV_LOG_ERROR, "error while decoding MB %d %d, bytestream (%td)\n", m->mb_x, m->mb_y, c->bytestream_end - c->bytestream);
                return;
            }
        }
    }
}

void stage_decode_slice_mb(DecFrame& frm) {
    CONCORE_PROFILING_FUNCTION();
    CONCORE_PROFILING_SET_TEXT_FMT(32, "frame=%d", frm.index);
    if (!frm.frame_data)
        return;

    auto& fd = *frm.frame_data;

    H264Slice *s = &fd.slice;
    H264Mb *mbs = fd.mbs;

    auto h = frm.global_ctx->h;
    for (int i=0; i<2; i++){
        for(int j=0; j< s->ref_count[i]; j++){
            if (s->ref_list_cpn[i][j] ==-1)
                continue;
            int k;
            for (k=0; k<h->max_dpb_cnt; k++){
                if(h->dpb[k].reference >= 2 && h->dpb[k].cpn == s->ref_list_cpn[i][j]){
                    s->dp_ref_list[i][j] = &h->dpb[k];
                    break;
                }
            }
        }
    }
    get_dpb_entry(h, s);

    if (!h->no_mbd) {
        // This will be broken into multiple tasks. Exchange continuation.
        auto cont = concore::exchange_cur_continuation();
        auto grp = concore::task_group::current_task_group();
        concore::task end_task{[] {}, grp, std::move(cont)};

        auto chunk_fun = [&frm] (int x, int y) {
            decode_slice_mb_chunk(frm.global_ctx, *frm.frame_data, x, y);
        };
        int width = h->mb_width/mb_line_chunk_size;
        fd.mb_processing.start(width, h->mb_height, chunk_fun, std::move(end_task));
    }
}

void stage_gen_output(DecFrame& frm) {
    CONCORE_PROFILING_FUNCTION();
    CONCORE_PROFILING_SET_TEXT_FMT(32, "frame=%d", frm.index);
    if (!frm.frame_data)
        return;

    auto& fd = *frm.frame_data;

    auto h = frm.global_ctx->h;
    H264Slice *s = &fd.slice;
    for (int i=0; i<s->release_cnt; i++){
        for(int j=0; j<h->max_dpb_cnt; j++){
            if(h->dpb[j].cpn== s->release_ref_cpn[i]){
                release_dpb_entry(h, &h->dpb[j], 2);
                break;
            }
        }
    }
    s->release_cnt = 0;

    auto oc = frm.global_ctx->out_ctx;
    auto out =output_frame(h, oc, s->curr_pic, h->ofile, h->frame_width, h->frame_height);
    if (out){
        release_dpb_entry(h, out, 1);
    }

    print_report(oc->frame_number, oc->dropped_frames, oc->video_size, 0, h->verbose);

    // Release the decoding frame context
    frm.global_ctx->frames_stash.push(std::move(frm.frame_data));
}

// The main loop of the file converter
extern "C" int h264_decode_concore_pipeline(H264Context* h) {
    GlobalDecContext ctx;
    ctx.h = h;
    ctx.pc = get_parse_context(h->ifile);
    ctx.nc = get_nal_context(h->width, h->height);
    ctx.out_ctx = get_output_context(h);

#if HAVE_LIBSDL2
    pthread_t sdl_thr;
    if (h->display) {
        pthread_create(&sdl_thr, NULL, sdl_thread, h);
    }
#endif

    av_start_timer();

    // Init concore with the right number of threads
    concore::init_data cfg;
    cfg.num_workers_ = h->threads - 1;
    concore::init(cfg);

    // Construct the decoding frame contexts
    int num_lines = ctx.h->mb_height;
    for (int i = 0; i < h->threads; i++) {
        auto frame = std::make_shared<FrameData>(h);
        frame->mb_line_dec_ctx.resize(num_lines);
        ctx.frames_stash.push(std::move(frame));
    }

    // Build a pipeline with proper stages
    concore::task_group group = concore::task_group::create();
    concore::pipeline<DecFrame> process{h->threads, group};
    auto in_order = concore::stage_ordering::in_order;
    auto conc = concore::stage_ordering::concurrent;
    process.add_stage(in_order, [&process](DecFrame& frm) { stage_parse(frm, process); });
    process.add_stage(conc, &stage_decode_slice_entropy);
    process.add_stage(in_order, &stage_decode_slice_mb);
    process.add_stage(in_order, &stage_gen_output);

    // Push the first frame through the pipeline
    process.push(DecFrame{0, &ctx});

    // Wait until we process all the pipeline
    concore::wait(group);

    // Output the remaining frames
    while (output_frame(h, ctx.out_ctx, NULL, h->ofile, h->frame_width, h->frame_height))
        ;

    print_report(ctx.out_ctx->frame_number, ctx.out_ctx->dropped_frames, ctx.out_ctx->video_size, 1,
            h->verbose);
    h->num_frames = ctx.out_ctx->frame_number;

#if HAVE_LIBSDL2
    if (h->display) {
        signal_sdl_exit(h);
        pthread_join(sdl_thr, NULL);
    }
#endif

    // Free our working objects
    MBRecContext* rc;
    while (ctx.mbrec_ctx_stash.try_pop(rc)) {
        free(rc->top);
        free_mbrec_context(rc);
    }
    free_output_context(ctx.out_ctx);
    free_parse_context(ctx.pc);
    free_nal_context(ctx.nc);

    return 0;
}
