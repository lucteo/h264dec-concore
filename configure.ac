#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT([h264_mt], [0.1], [cchi@cs.tu-berlin.de])
#AM_INIT_AUTOMAKE(AC_PACKAGE_NAME, AC_PACKAGE_VERSION)
AM_INIT_AUTOMAKE([-Wall -Werror foreign])

AC_CONFIG_SRCDIR([h264dec.c])
AC_PROG_RANLIB

# Checks for programs.
AC_GNU_SOURCE
AC_PROG_CC
AM_CONDITIONAL([HAVE_OMPSS], [test $CC = "sscc"])
AC_DEFINE([OMPSS], [0], [Define to 1 on when using the OmpSs compiler sscc])
if test $CC = "sscc";then
AC_DEFINE([OMPSS], [1], [Define to 1 on when using the OmpSs compiler sscc])
fi

#if [ test -n "${CFLAGS+x}" ] ; then
#    CFLAGS="-O3 -g"
#fi

# Checks for libraries.
AC_CHECK_LIB([pthread], [pthread_yield])
AC_CHECK_LIB([spe2], [spe_image_open])
AC_CHECK_LIB([sync], [mutex_init])
AC_CHECK_LIB([rt], [clock_gettime])

AC_ARG_ENABLE([sdl2], AS_HELP_STRING([--enable-sdl2], [Enable SDL2 playback]))
if test "$enable_sdl2" = "yes"; then
	AC_CHECK_LIB([SDL2], [SDL_CreateWindow], [], [echo "Error! libSDL2 required for playback." exit -1])
fi

if test "$enable_sdl2" = "yes"; then
	AC_CHECK_LIB([X11], [XInitThreads], [], [echo "Error! libX11 currently required for SDL2 workaround." exit -1])
fi

AC_ARG_ENABLE([sdl_ttf], AS_HELP_STRING([--enable-sdl_ttf], [Enable SDL_ttf for overlaying fonts]))
if test "$enable_sdl_ttf" = "yes"; then
    AC_CHECK_LIB([SDL_ttf], [TTF_Init], [], [echo "Error! libSDL_ttf required for font rendering." exit -1])
fi



AC_ARG_ENABLE([opencl], AS_HELP_STRING([--enable-opencl], [Enable GPU decoder]))
if test "$enable_opencl" = "yes"; then
	AC_CHECK_LIB([OpenCL], [clGetPlatformIDs], [], [echo "Error! libOpenCL required for GPU functionality." exit -1])
fi
AM_CONDITIONAL([HAVE_OPENCL], [test "$enable_opencl" = "yes"])


# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([stdint.h stdlib.h string.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T
AC_C_VOLATILE
AC_C_BIGENDIAN

# Checks for library functions.
AC_CHECK_FUNCS([malloc realloc memalign posix_memalign memmove memset])

AC_CANONICAL_HOST
AC_CANONICAL_BUILD

AC_MSG_CHECKING([for architecture])

AC_DEFINE([ARCH_ARM], [0], [Define to 1 on arm architectures.])
AC_DEFINE([ARCH_X86_32], [0], [Define to 1 on x86 architectures.])
AC_DEFINE([ARCH_X86_64], [0], [Define to 1 on x86_64 architectures.])
AC_DEFINE([ARCH_X86], [ARCH_X86_32 ||ARCH_X86_64], [True on x86])
AC_DEFINE([ARCH_PPC], [0], [Define to 1 on ppc architectures.])
AC_DEFINE([ARCH_PPC64], [0], [Define to 1 on ppc64 architectures.])
AC_DEFINE([ARCH_CELL], [0], [Define to 1 on cell architectures.])

if test "$enable_optimizations" != "no"; then
	case $build_cpu in
		arm )
			arch="arm"
			AC_MSG_RESULT([arm])
			AC_DEFINE([ARCH_ARM], [1], [Define to 1 on arm architectures.])
			;;
		i686 )
			arch="x86"
			AC_MSG_RESULT([x86])
			AC_DEFINE([ARCH_X86_32], [1], [Define to 1 on x86 architectures.])
			;;
		x86_64 )
			arch="x86_64"
			AC_MSG_RESULT([x86_64])
			AC_DEFINE([ARCH_X86_64], [1], [Define to 1 on x86 architectures.])
			;;
		powerpc64 )
			AC_DEFINE([HAVE_BIGENDIAN], [1], [Define to 1 on bigendian architectures.])
			if grep -E ^cpu /proc/cpuinfo | grep -q Cell ; then
				arch="cell"
				AC_MSG_RESULT([cell])
				AC_DEFINE([ARCH_CELL], [1], [Define to 1 on cell architectures.])
			else
				arch="powerpc64"
				AC_MSG_RESULT([ppc64])
				AC_DEFINE([ARCH_PPC64], [1], [Define to 1 on ppc64 architectures.])
			fi
			;;
		* )
			AC_MSG_RESULT([default (little endian).])
			;;
	esac
fi

AM_CONDITIONAL([HAVE_CELL], [test $arch = "cell"])

# Additional options
AC_ARG_ENABLE([optimizations], AS_HELP_STRING([--disable-optimizations], [Disable all architecture specific optimizations. Compiler optimizations are not disabled.]))

AC_DEFINE([HAVE_SSE], [0], [Define to 1 to enable sse optimizations.])
AC_DEFINE([HAVE_MMX], [0], [Define to 1 to enable mmx optimizations.])
AC_DEFINE([HAVE_MMX2], [0], [Define to 1 to enable mmx2 optimizations.])
AC_DEFINE([HAVE_SSSE3], [0], [Define to 1 to enable ssse3 optimizations.])
AC_DEFINE([HAVE_ALTIVEC], [0], [Define to 1 to enable altivec optimizations.])
AC_DEFINE([HAVE_NEON], [0], [Define to 1 to enable neon optimizations.])

AC_ARG_ENABLE([ssse3], AS_HELP_STRING([--enable-ssse3], [Enable ssse3 optimizations]))
if test "$enable_ssse3" = "yes"; then
	AC_DEFINE([HAVE_SSSE3], [1], [Define to 1 to enable ssse3 optimizations.])
	AC_DEFINE([HAVE_SSE], [1], [Define to 1 to enable sse optimizations.])
	AC_DEFINE([HAVE_MMX], [1], [Define to 1 to enable mmx optimizations.])
	AC_DEFINE([HAVE_MMX2], [1], [Define to 1 to enable mmx2 optimizations.])
	ARCH_SUBDIR=x86
fi

AC_ARG_ENABLE([sse], AS_HELP_STRING([--enable-sse], [Enable sse optimizations]))
if test "$enable_sse" = "yes"; then
	AC_DEFINE([HAVE_SSE], [1], [Define to 1 to enable sse optimizations.])
	AC_DEFINE([HAVE_MMX], [1], [Define to 1 to enable mmx optimizations.])
	AC_DEFINE([HAVE_MMX2], [1], [Define to 1 to enable mmx2 optimizations.])
	ARCH_SUBDIR=x86
fi

AC_ARG_ENABLE([altivec], AS_HELP_STRING([--enable-altivec], [Enable altivec optimizations]))
if test "$enable_altivec" = "yes"; then
	AC_DEFINE([HAVE_ALTIVEC], [1], [Define to 1 to enable altivec optimizations.])
	ARCH_SUBDIR="$ARCH_SUBDIR ppc"
	TMPCLAGS=$CFLAGS
	CFLAGS="$CFLAGS -maltivec"
	AC_CHECK_HEADERS(altivec.h)
	CFLAGS=$TMPCLAGS
fi

AC_ARG_ENABLE([neon], AS_HELP_STRING([--enable-neon], [Enable neon optimizations]))
if test "$enable_neon" = "yes"; then
	AC_DEFINE([HAVE_NEON], [1], [Define to 1 to enable neon optimizations.])
	ARCH_SUBDIR=arm
fi

AM_CONDITIONAL([HAVE_ARCH_SUBDIR], [test "$ARCH_SUBDIR" != ""])
AC_SUBST([ARCH_SUBDIR])

AC_DEFINE([HAVE_NEON], [0], [Define to 1 to enable neon optimizations.])

AC_CONFIG_HEADER([config.h])

AC_CONFIG_FILES([Makefile libavutil/Makefile libavcodec/Makefile libavcodec/x86/Makefile libavcodec/ppc/Makefile libavcodec/cell/Makefile])

AC_OUTPUT
