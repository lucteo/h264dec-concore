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
#include <concore/task_group.hpp>
#include <concore/serializer.hpp>
#include <concore/n_serializer.hpp>
#include <concore/task_graph.hpp>
#include <concore/profiling.hpp>

#include "h264_types.h"
#include "h264_parser.h"
#include "h264_nal.h"
#include "h264_entropy.h"
#include "h264_rec.h"
#include "h264_pred_mode.h"
#include "h264_misc.h"
// #undef NDEBUG
#include <assert.h>

#include <thread>
#include <atomic>
#include <chrono>
#include <vector>

using namespace std::chrono_literals;

struct atomic_int : std::atomic<int> {
    atomic_int(int val)
        : atomic(val) {}
    atomic_int(const atomic_int& other)
        : atomic(other.load()) {}
};

using MBRecCtxStash = concore::concurrent_queue<MBRecContext*>;

struct GlobalDecContext {
    H264Context* h;
    ParserContext* pc;
    NalContext* nc;
    OutputContext* out_ctx;
    MBRecCtxStash mbrec_ctx_stash;
};

struct DecMBLine { //!< Holds data for decoding MB for one line
    atomic_int completed_x{0};
    atomic_int start_state{0}; // 0==stopped, 1==started, 2==started, cannot stop
    MBRecContext* dec_ctx{nullptr};
};

//! Holds all the data to decode a frame of data.
//! This will be passed along though all the stages of the pipeline
struct DecFrameContext {
    GlobalDecContext* global_data;                    // global decoding context
    H264Slice slice;                                  // all phases
    GetBitContext gb{0};                              // parse, entropy
    EntropyContext* ec{nullptr};                      // entropy
    H264Mb* mbs{nullptr};                             // entropy, mb
    std::vector<DecMBLine> mb_lines;                  // structure for mb_dec, one for each line
    concore::chained_task* mb_dec_done_task{nullptr}; // called when mb decoding is done
    int frame_idx{0};                                 // the current frame index

    explicit DecFrameContext(GlobalDecContext* ctx);
    DecFrameContext(const DecFrameContext& other);
    ~DecFrameContext();

    bool parse();
    void decode_slice_entropy();
    void decode_slice_mb();
    void decode_slice_mb_line(int line_idx);
    void gen_output();

    void start_line_if_stopped(int line_idx, bool until_end = false);
    bool try_stop_line(int line_idx);
};

constexpr int num_stages = 5; // parse, dec_entropy, mb_start, mb_end, output

struct TaskGraph {
    std::vector<DecFrameContext> frame_ctxs;
    concore::task_group group{concore::task_group::create()};
    std::vector<concore::chained_task> tasks;
    concore::chained_task never_execute_task;
    concore::chained_task cleanup_task;
    int cur_frame{0};

    TaskGraph(GlobalDecContext* global_data, int num_par_frames);
    void run();

private:
    concore::chained_task& task(int frame, int stage);
    void start_frame(DecFrameContext& frm);
};

DecFrameContext::DecFrameContext(GlobalDecContext* ctx)
    : global_data(ctx) {
    memset(&slice, 0, sizeof(H264Slice));
    mbs = (H264Mb*)malloc(ctx->h->mb_height * ctx->h->mb_width * sizeof(H264Mb));
    ec = get_entropy_context(ctx->h);
}
DecFrameContext::DecFrameContext(const DecFrameContext& other)
    : DecFrameContext(other.global_data) {}
DecFrameContext::~DecFrameContext() {
    free(mbs);
    free(gb.raw);
    if (gb.rbsp)
        free(gb.rbsp);
    free_entropy_context(ec);
}

bool DecFrameContext::parse() {
    CONCORE_PROFILING_FUNCTION();
    CONCORE_PROFILING_SET_TEXT_FMT(32, "frame=%d", frame_idx);

    if (!global_data->pc->final_frame && frame_idx < global_data->h->num_frames &&
            !global_data->h->quit) {
        memset(&slice, 0, sizeof(H264Slice));
        av_read_frame_internal(global_data->pc, &gb);
        decode_nal_units(global_data->nc, &slice, &gb);
        return true;
    } else
        return false;
}

void DecFrameContext::decode_slice_entropy() {
    CONCORE_PROFILING_FUNCTION();
    CONCORE_PROFILING_SET_TEXT_FMT(32, "frame=%d", slice.coded_pic_num);

    int i, j;
    CABACContext* c = &ec->c;

    if (!slice.pps.cabac) {
        av_log(AV_LOG_ERROR, "Only cabac encoded streams are supported\n");
        return;
    }

    init_dequant_tables(&slice, ec);
    ec->curr_qscale = slice.qscale;
    ec->last_qscale_diff = 0;
    ec->chroma_qp[0] = get_chroma_qp(&slice, 0, slice.qscale);
    ec->chroma_qp[1] = get_chroma_qp(&slice, 1, slice.qscale);

    /* realign */
    align_get_bits(&gb);
    /* init cabac */
    ff_init_cabac_decoder(c, gb.buffer + get_bits_count(&gb) / 8, (get_bits_left(&gb) + 7) / 8);

    ff_h264_init_cabac_states(ec, &slice, c);

    for (j = 0; j < ec->mb_height; j++) {
        CONCORE_PROFILING_SCOPE_N("line");
        CONCORE_PROFILING_SET_TEXT_FMT(32, "line=%d", j);
        init_entropy_buf(ec, &slice, j);
        for (i = 0; i < ec->mb_width; i++) {
            int eos, ret;
            H264Mb* m = &mbs[i + j * ec->mb_width];
            // memset(m, 0, sizeof(H264Mb));
            m->mb_x = i;
            m->mb_y = j;
            ec->m = m;

            ret = ff_h264_decode_mb_cabac(ec, &slice, c);
            eos = get_cabac_terminate(c);
            (void)eos;
            if (ret < 0 || c->bytestream > c->bytestream_end + 2) {
                av_log(AV_LOG_ERROR, "error while decoding MB %d %d, bytestream (%td)\n", m->mb_x,
                        m->mb_y, c->bytestream_end - c->bytestream);
                return;
            }
        }
    }
}

void DecFrameContext::decode_slice_mb() {
    CONCORE_PROFILING_FUNCTION();
    CONCORE_PROFILING_SET_TEXT_FMT(32, "frame=%d", slice.coded_pic_num);

    auto h = global_data->h;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < slice.ref_count[i]; j++) {
            if (slice.ref_list_cpn[i][j] == -1)
                continue;
            int k;
            for (k = 0; k < h->max_dpb_cnt; k++) {
                if (h->dpb[k].reference >= 2 && h->dpb[k].cpn == slice.ref_list_cpn[i][j]) {
                    slice.dp_ref_list[i][j] = &h->dpb[k];
                    break;
                }
            }
        }
    }
    get_dpb_entry(h, &slice);

    if (!h->no_mbd) {
        for (int i = 0; i < h->mb_height; i++) {
            mb_lines[i].completed_x.store(0);
            mb_lines[i].start_state.store(0);
        }
        DecMBLine* first_line = &mb_lines[0];
        first_line->start_state.store(0);
        start_line_if_stopped(0);
    }
}

void DecFrameContext::decode_slice_mb_line(int line_idx) {
    CONCORE_PROFILING_SCOPE_N("mb line");
    auto h = global_data->h;
    DecMBLine& line = mb_lines[line_idx];
    DecMBLine* next_line = line_idx < h->mb_height - 1 ? &mb_lines[line_idx + 1] : nullptr;
    DecMBLine* prev_line = line_idx > 0 ? &mb_lines[line_idx - 1] : nullptr;

    int start_x = line.completed_x.load();
    CONCORE_PROFILING_SET_TEXT_FMT(64, "frame=%d, line=%d, start=%d", frame_idx, line_idx, start_x);

    // First time: ensure we have a mbrec context
    if (start_x == 0) {
        if (!global_data->mbrec_ctx_stash.try_pop(line.dec_ctx)) {
            line.dec_ctx = get_mbrec_context(h);
            line.dec_ctx->top_next = line.dec_ctx->top =
                    (TopBorder*)malloc(h->mb_width * sizeof(TopBorder));
        }
        init_mbrec_context(line.dec_ctx, line.dec_ctx->mrs, &slice, line_idx);
    }

    static constexpr int headstart = 10;

    for (int i = start_x; i < h->mb_width; i++) {
        if (prev_line) {
            int prev_completed_x = prev_line->completed_x.load();
            if (prev_completed_x < h->mb_width && i + 1 >= prev_completed_x) {
                if (try_stop_line(line_idx))
                    return;
                assert(prev_line->completed_x.load() > i);
            }
        }
        if (next_line) {
            // Start decoding the next line if we have enough headstart
            int next_x = next_line->completed_x.load();
            if (next_x + headstart <= i)
                start_line_if_stopped(line_idx + 1);
        }

        CONCORE_PROFILING_SCOPE_N("mb pixel");
        CONCORE_PROFILING_SET_TEXT_FMT(64, "line=%d, col=%d", line_idx, i);
        H264Mb* m = &mbs[i + line_idx * line.dec_ctx->mb_width];
        h264_decode_mb_internal(line.dec_ctx, line.dec_ctx->mrs, &slice, m);

        line.completed_x.store(i + 1);
    }
    if (next_line)
        start_line_if_stopped(line_idx + 1, true);
    draw_edges(line.dec_ctx, &slice, line_idx);

    // Done with the line; we don't need dec_ctx anymore
    global_data->mbrec_ctx_stash.push(std::move(line.dec_ctx));

    if (line_idx == h->mb_height - 1)
        (*mb_dec_done_task)();
}
void DecFrameContext::gen_output() {
    CONCORE_PROFILING_FUNCTION();
    CONCORE_PROFILING_SET_TEXT_FMT(32, "frame=%d", frame_idx);

    auto h = global_data->h;
    for (int i = 0; i < slice.release_cnt; i++) {
        for (int j = 0; j < h->max_dpb_cnt; j++) {
            if (h->dpb[j].cpn == slice.release_ref_cpn[i]) {
                release_dpb_entry(h, &h->dpb[j], 2);
                break;
            }
        }
    }
    slice.release_cnt = 0;

    auto oc = global_data->out_ctx;
    auto out = output_frame(h, oc, slice.curr_pic, h->ofile, h->frame_width, h->frame_height);
    if (out)
        release_dpb_entry(h, out, 1);
    print_report(oc->frame_number, oc->dropped_frames, oc->video_size, 0, h->verbose);
}

void DecFrameContext::start_line_if_stopped(int line_idx, bool until_end) {
    int old = mb_lines[line_idx].start_state.exchange(until_end ? 2 : 1);
    if (old == 0) {
        concore::global_executor ex(concore::global_executor::prio_high);
        concore::execute(ex, [=]() { decode_slice_mb_line(line_idx); });
    }
}
bool DecFrameContext::try_stop_line(int line_idx) {
    CONCORE_PROFILING_FUNCTION();
    int old = 1;
    return mb_lines[line_idx].start_state.compare_exchange_strong(old, 0);
}

TaskGraph::TaskGraph(GlobalDecContext* global_data, int num_par_frames) {
    // Construct the decoding frame contexts
    frame_ctxs.reserve(static_cast<size_t>(num_par_frames));
    for (int i = 0; i < num_par_frames; i++)
        frame_ctxs.emplace_back(global_data);

    // Init the lines for each decoding frame context
    int num_lines = global_data->h->mb_height;
    assert(num_lines > 0);
    for (auto& frame : frame_ctxs)
        frame.mb_lines.resize(num_lines);

    // Create and initialize all the tasks
    // We will create tasks for all the stages for all the lines
    tasks.reserve(static_cast<size_t>(num_par_frames * num_stages));
    concore::global_executor e;
    for (int i = 0; i < num_par_frames; i++) {
        auto& frm = frame_ctxs[i];

        auto fun_parse = [this, &frm]() { start_frame(frm); };
        auto fun_dec_ent = [this, &frm]() { frm.decode_slice_entropy(); };
        auto fun_dec_mb = [this, &frm]() { frm.decode_slice_mb(); };
        auto fun_dec_mb_done = []() {};
        auto fun_out = [this, &frm]() { frm.gen_output(); };

        tasks.emplace_back(concore::chained_task({fun_parse, group}, e));
        tasks.emplace_back(concore::chained_task({fun_dec_ent, group}, e));
        tasks.emplace_back(concore::chained_task({fun_dec_mb, group}, e));
        tasks.emplace_back(concore::chained_task({fun_dec_mb_done, group}, e));
        tasks.emplace_back(concore::chained_task({fun_out, group}, e));

        frm.mb_dec_done_task = &task(i, 3);
    }
    auto cleanup_ftor = [=]() {
        CONCORE_PROFILING_SCOPE_N("cleanup tasks");
        tasks.clear();
        never_execute_task = concore::chained_task(); // clear the dependencies
    };
    cleanup_task = concore::chained_task(cleanup_ftor, e);
    never_execute_task = concore::chained_task([]() {}, e);
}

void TaskGraph::run() {
    // Create the dependencies for the first frame
    concore::add_dependency(task(0, 0), task(0, 1));
    concore::add_dependency(task(0, 1), task(0, 2));
    concore::add_dependency(task(0, 3), task(0, 4));

    // Spawn the first task, and wait for all the tasks to be completed
    concore::spawn(task(0, 0), false);
    concore::wait(group);
}

concore::chained_task& TaskGraph::task(int frame, int stage) {
    frame = (frame_ctxs.size() + frame) % frame_ctxs.size();
    return tasks[frame * num_stages + stage];
}

void TaskGraph::start_frame(DecFrameContext& frm) {
    CONCORE_PROFILING_FUNCTION();

    int frame_idx = cur_frame++;
    frm.frame_idx = frame_idx;
    if (frm.parse()) {
        // Ensure we have a dependency from the prev frame to this one
        concore::add_dependency(task(frame_idx - 1, 0), task(frame_idx, 0));

        // Dependencies between the current frame and the next frame
        concore::add_dependency(task(frame_idx, 0), task(frame_idx + 1, 0));
        concore::add_dependency(task(frame_idx, 3), task(frame_idx + 1, 2)); // end->start
        concore::add_dependency(task(frame_idx, 4), task(frame_idx + 1, 4));

        // Dependencies between the tasks for the next frame
        // The ones for the current frame are already created
        concore::add_dependency(task(frame_idx + 1, 0), task(frame_idx + 1, 1));
        concore::add_dependency(task(frame_idx + 1, 1), task(frame_idx + 1, 2));
        // 2 -> 3 is manually handled
        concore::add_dependency(task(frame_idx + 1, 3), task(frame_idx + 1, 4));

        // To keep the number of tasks below a threshold, we add a dependency between the last stage
        // of this frame and the first stage of a new frame
        concore::add_dependency(task(frame_idx + 1, 4), task(frame_idx - 1, 0));
    } else {
        // Don't continue with the next tasks; add a dependency that will never be cleared
        concore::add_dependency(never_execute_task, task(frame_idx, 1));
        concore::add_dependency(never_execute_task, task(frame_idx + 1, 0));
        concore::add_dependency(task(frame_idx - 1, 4), cleanup_task);
    }
}

// The main loop of the file converter
extern "C" int h264_decode_concore(H264Context* h) {
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
    TaskGraph g(&ctx, h->threads);
    g.run();

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
