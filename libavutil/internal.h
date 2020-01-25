/*
 * copyright (c) 2006 Michael Niedermayer <michaelni@gmx.at>
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
 * common internal API header
 */

#ifndef AVUTIL_INTERNAL_H
#define AVUTIL_INTERNAL_H

#if !defined(DEBUG) && !defined(NDEBUG)
#    define NDEBUG
#endif

#include <limits.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include "config.h"
#include "attributes.h"
#include "timer.h"



#ifndef INT16_MIN
#define INT16_MIN       (-0x7fff - 1)
#endif

#ifndef INT16_MAX
#define INT16_MAX       0x7fff
#endif

#ifndef INT32_MIN
#define INT32_MIN       (-0x7fffffff - 1)
#endif

#ifndef INT32_MAX
#define INT32_MAX       0x7fffffff
#endif

#ifndef UINT32_MAX
#define UINT32_MAX      0xffffffff
#endif

#ifndef INT64_MIN
#define INT64_MIN       (-0x7fffffffffffffffLL - 1)
#endif

#ifndef INT64_MAX
#define INT64_MAX INT64_C(9223372036854775807)
#endif

#ifndef UINT64_MAX
#define UINT64_MAX UINT64_C(0xFFFFFFFFFFFFFFFF)
#endif

#ifndef INT_BIT
#    define INT_BIT (CHAR_BIT * sizeof(int))
#endif

#ifndef offsetof
#    define offsetof(T, F) ((unsigned int)((char *)&((T *)0)->F))
#endif

/* Use to export labels from asm. */
#define LABEL_MANGLE(a) #a
#define LOCAL_MANGLE(a) #a
#define MANGLE(a) #a

// Use rip-relative addressing if compiling PIC code on x86-64.
// #if ARCH_X86_64 && defined(PIC)
// #    define LOCAL_MANGLE(a) #a "(%%rip)"
// #else
// #    define LOCAL_MANGLE(a) #a
// #endif
// 
// #define MANGLE(a) EXTERN_PREFIX LOCAL_MANGLE(a)

/* debug stuff */

/* dprintf macros */
#ifdef DEBUG
#    define dprintf(pctx, ...) av_log(pctx, AV_LOG_DEBUG, __VA_ARGS__)
#else
#    define dprintf(pctx, ...)
#endif

#define av_abort()      do { av_log(NULL, AV_LOG_ERROR, "Abort at %s:%d\n", __FILE__, __LINE__); abort(); } while (0)

/* math */


/* avoid usage of dangerous/inappropriate system functions */
// #undef  malloc
// #define malloc please_use_av_malloc
// #undef  free
// #define free please_use_av_free
#undef  realloc
#define realloc please_use_av_realloc
#undef  time
#define time time_is_forbidden_due_to_security_issues
#undef  rand
#define rand rand_is_forbidden_due_to_state_trashing_use_av_lfg_get
#undef  srand
#define srand srand_is_forbidden_due_to_state_trashing_use_av_lfg_init
#undef  random
#define random random_is_forbidden_due_to_state_trashing_use_av_lfg_get
#undef  sprintf
#define sprintf sprintf_is_forbidden_due_to_security_issues_use_snprintf
//#undef  exit
//#define exit exit_is_forbidden
#ifndef LIBAVFORMAT_BUILD

#undef  puts
#define puts please_use_av_log_instead_of_puts
#undef  perror
#define perror please_use_av_log_instead_of_perror
#endif

#define FF_ALLOC_OR_GOTO(p, size, label)\
{\
    p = av_malloc(size);\
    if (p == NULL && (size) != 0) {\
        av_log(AV_LOG_ERROR, "Cannot allocate memory.\n");\
        goto label;\
    }\
}

#define FF_ALLOCZ_OR_GOTO(p, size, label)\
{\
    p = av_mallocz(size);\
    if (p == NULL && (size) != 0) {\
        av_log(AV_LOG_ERROR, "Cannot allocate memory.\n");\
        goto label;\
    }\
}


/**
 * Returns NULL if CONFIG_SMALL is true, otherwise the argument
 * without modification. Used to disable the definition of strings
 * (for example AVCodec long_names).
 */
#if CONFIG_SMALL
#   define NULL_IF_CONFIG_SMALL(x) NULL
#else
#   define NULL_IF_CONFIG_SMALL(x) x
#endif

#endif /* AVUTIL_INTERNAL_H */
