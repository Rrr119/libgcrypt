/* t-const-time.c - Tests for the constant-time comparison functions
 * Copyright (C) 2026 g10 Code GmbH
 *
 * This file is part of Libgcrypt.
 *
 * Libgcrypt is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * Libgcrypt is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, see <https://www.gnu.org/licenses/>.
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PGM "t-const-time"
#include "t-common.h"

/* Align the timing buffers to a cache line so that the measurements
   are not skewed by different cache-line split behavior of the
   individual buffers.  */
#if defined(__GNUC__)
# define ALIGN64 __attribute__ ((aligned (64)))
#else
# define ALIGN64
#endif

/* A sink to keep the compiler from optimizing the measured calls away.  */
static volatile unsigned int ct_sink;

/* A simple xorshift PRNG so that the test does not depend on the
   state of the Libgcrypt RNG and is reproducible.  */
static unsigned long long rng_state = 0x9e3779b97f4a7c15ULL;

static unsigned int
next_rand (void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (unsigned int)(rng_state >> 32);
}


/*
 * Correctness of gcry_ct_memequal and its compatibility with the
 * semantics of memcmp: equal buffers must compare equal, any single
 * bit difference must compare unequal.
 */
static void
test_memequal (void)
{
  unsigned char a[256], b[256];
  size_t len;
  int i;

  for (len = 0; len <= 200; len++)
    {
      for (i = 0; i < (int)sizeof a; i++)
        {
          a[i] = next_rand ();
          b[i] = next_rand ();
        }
      memcpy (b, a, len);

      if (gcry_ct_memequal (a, b, len) != 1)
        fail ("gcry_ct_memequal: equal buffers of len %u not equal\n",
              (unsigned int)len);

      if (len > 0)
        {
          size_t positions[3];

          positions[0] = 0;
          positions[1] = len / 2;
          positions[2] = len - 1;
          for (i = 0; i < 3; i++)
            {
              b[positions[i]] ^= 0x01;
              if (gcry_ct_memequal (a, b, len) != 0)
                fail ("gcry_ct_memequal: difference at byte %u of %u"
                      " not detected\n",
                      (unsigned int)positions[i], (unsigned int)len);
              b[positions[i]] ^= 0x01;
            }
        }
    }

  /* Cross-check against memcmp on random inputs.  */
  for (i = 0; i < 2000; i++)
    {
      int j;

      len = next_rand () % 200;
      for (j = 0; j < (int)len; j++)
        {
          a[j] = next_rand ();
          b[j] = next_rand ();
        }
      if (next_rand () & 1)
        memcpy (b, a, len);

      if (gcry_ct_memequal (a, b, len) != (unsigned int)(!memcmp (a, b, len)))
        fail ("gcry_ct_memequal: result differs from memcmp at len %u\n",
              (unsigned int)len);
    }

  if (verbose)
    info ("gcry_ct_memequal: correctness and memcmp compatibility OK\n");
}


/*
 * Correctness of gcry_ct_memequal_len, in particular the length
 * comparison: buffers with different lengths must never compare
 * equal, even if they share a common prefix.
 */
static void
test_memequal_len (void)
{
  static const size_t lengths[] =
    { 0, 1, 2, 7, 8, 9, 15, 16, 17, 32, 64, 128, 200 };
  unsigned char a[256], b[256];
  unsigned int i, j;
  int k;

  for (k = 0; k < (int)sizeof a; k++)
    {
      a[k] = next_rand ();
      b[k] = a[k];
    }

  for (i = 0; i < DIM (lengths); i++)
    for (j = 0; j < DIM (lengths); j++)
      {
        size_t len1 = lengths[i];
        size_t len2 = lengths[j];
        unsigned int expect = (len1 == len2);

        if (gcry_ct_memequal_len (a, len1, b, len2) != expect)
          fail ("gcry_ct_memequal_len: len1=%u len2=%u returned %u,"
                " expected %u\n",
                (unsigned int)len1, (unsigned int)len2, !expect, expect);

        /* A content difference in the common prefix must be detected
           regardless of the length combination.  */
        if (len1 && len2)
          {
            b[0] ^= 0x80;
            if (gcry_ct_memequal_len (a, len1, b, len2) != 0)
              fail ("gcry_ct_memequal_len: content difference not detected"
                    " (len1=%u len2=%u)\n",
                    (unsigned int)len1, (unsigned int)len2);
            b[0] ^= 0x80;
          }
      }

  if (verbose)
    info ("gcry_ct_memequal_len: length and content checks OK\n");
}


/*
 * Statistical timing tests.
 *
 * The idea: measure the wall-clock time of batches of comparisons for
 * two input classes (e.g. "buffers differ at the first byte" vs.
 * "buffers differ at the last byte"), interleaving the batches to
 * cancel out drift, and compare the batch-time distributions with
 * Welch's t-test.  An implementation with a secret-dependent early
 * exit shows up as a huge t statistic.
 */

#define N_BATCHES 61
#define BATCH_INNER 2000

/* Two input classes are considered distinguishable if both the paired
   sign-test z-score exceeds Z_THRESHOLD and the relative difference
   of the medians exceeds REL_THRESHOLD.  The thresholds are
   deliberately generous: an early-exit implementation differs by a
   factor proportional to the buffer length, which is orders of
   magnitude above these limits.  */
#define Z_THRESHOLD 5.0
#define REL_THRESHOLD 0.02

struct cmp_desc
{
  int with_len;             /* 0: gcry_ct_memequal, 1: gcry_ct_memequal_len */
  const void *p1;
  size_t len1;
  const void *p2;
  size_t len2;
};


static double
now_ns (void)
{
#if defined(HAVE_CLOCK_GETTIME) && defined(CLOCK_MONOTONIC)
  struct timespec ts;

  if (clock_gettime (CLOCK_MONOTONIC, &ts) == 0)
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
  {
    clock_t c = clock ();

    return (double)c * (1e9 / (double)CLOCKS_PER_SEC);
  }
}


static double
run_batch (const struct cmp_desc *desc, int inner)
{
  double t0, t1;
  unsigned int acc = 0;
  int i;

  t0 = now_ns ();
  if (desc->with_len)
    {
      for (i = 0; i < inner; i++)
        acc += gcry_ct_memequal_len (desc->p1, desc->len1,
                                     desc->p2, desc->len2);
    }
  else
    {
      for (i = 0; i < inner; i++)
        acc += gcry_ct_memequal (desc->p1, desc->p2, desc->len1);
    }
  t1 = now_ns ();

  ct_sink += acc;
  return t1 - t0;
}


/* Fill T1 and T2 with interleaved batch timings for D1 and D2.  */
static void
measure_pair (const struct cmp_desc *d1, const struct cmp_desc *d2,
              double *t1, double *t2, int nbatches, int inner)
{
  int k;

  /* Warm-up.  */
  run_batch (d1, inner);
  run_batch (d2, inner);

  for (k = 0; k < nbatches; k++)
    {
      if (k & 1)
        {
          t2[k] = run_batch (d2, inner);
          t1[k] = run_batch (d1, inner);
        }
      else
        {
          t1[k] = run_batch (d1, inner);
          t2[k] = run_batch (d2, inner);
        }
    }
}


/* Newton-based square root to avoid a libm dependency.  */
static double
my_sqrt (double x)
{
  double r;
  int i;

  if (x <= 0)
    return 0;
  r = x > 1 ? x : 1;
  for (i = 0; i < 60; i++)
    r = 0.5 * (r + x / r);
  return r;
}


static int
double_cmp (const void *a, const void *b)
{
  double da = *(const double *)a;
  double db = *(const double *)b;

  return (da > db) - (da < db);
}


static double
median_of (const double *x, int n)
{
  double sorted[N_BATCHES];

  memcpy (sorted, x, n * sizeof *sorted);
  qsort (sorted, n, sizeof *sorted, double_cmp);
  if (n & 1)
    return sorted[n / 2];
  return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}


/* Welch's t statistic for two samples; reported for information
   only.  */
static double
welch_t (const double *x, int n, const double *y, int m)
{
  double mx = 0, my = 0, vx = 0, vy = 0;
  double denom;
  int i;

  for (i = 0; i < n; i++)
    mx += x[i];
  mx /= n;
  for (i = 0; i < m; i++)
    my += y[i];
  my /= m;

  for (i = 0; i < n; i++)
    vx += (x[i] - mx) * (x[i] - mx);
  vx /= (n - 1);
  for (i = 0; i < m; i++)
    vy += (y[i] - my) * (y[i] - my);
  vy /= (m - 1);

  denom = my_sqrt (vx / n + vy / m);
  if (denom == 0)
    return 0;
  return (mx - my) / denom;
}


/* Paired sign test: batch K of the first group is paired with batch K
   of the second group (they are measured back to back, which cancels
   drift).  Returns the z-score of the sign statistic; a large
   absolute value means one group is consistently slower than the
   other.  */
static double
sign_z (const double *x, const double *y, int n)
{
  int greater = 0, less = 0;
  int i;

  for (i = 0; i < n; i++)
    {
      if (x[i] > y[i])
        greater++;
      else if (x[i] < y[i])
        less++;
    }

  if (!greater && !less)
      return 0;
  return (double)(greater - less) / my_sqrt ((double)(greater + less));
}


/* Return true if the two input classes are statistically
   distinguishable with our thresholds.  */
static int
distinguishable (const struct cmp_desc *d1, const struct cmp_desc *d2,
                 const char *what)
{
  double t1[N_BATCHES], t2[N_BATCHES];
  double pairdiff[N_BATCHES];
  double reldiff, t, z;
  int i;

  measure_pair (d1, d2, t1, t2, N_BATCHES, BATCH_INNER);
  t = welch_t (t1, N_BATCHES, t2, N_BATCHES);
  z = sign_z (t1, t2, N_BATCHES);
  /* Relative difference within each pair of back-to-back batches;
     using the median of the per-pair differences cancels clock drift
     and is robust against the one-sided spikes caused by
     interrupts.  */
  for (i = 0; i < N_BATCHES; i++)
    pairdiff[i] = 2 * (t1[i] - t2[i]) / (t1[i] + t2[i]);
  reldiff = median_of (pairdiff, N_BATCHES);
  if (t < 0)
    t = -t;
  if (z < 0)
    z = -z;
  if (reldiff < 0)
    reldiff = -reldiff;

  info ("%s: t=%.2f z=%.2f median-reldiff=%.4f%%\n",
        what, t, z, reldiff * 100);
  return z > Z_THRESHOLD && reldiff > REL_THRESHOLD;
}


static void
test_timing (void)
{
  unsigned char ref[128] ALIGN64;
  unsigned char equal[128] ALIGN64;
  unsigned char diff_first[128] ALIGN64;
  unsigned char diff_last[128] ALIGN64;
  struct cmp_desc d_equal, d_first, d_last, d_len64, d_len128;
  struct cmp_desc d_short, d_long;
  int i;
  int leaky = 0;

  for (i = 0; i < (int)sizeof ref; i++)
    ref[i] = next_rand ();
  memcpy (equal, ref, sizeof ref);
  memcpy (diff_first, ref, sizeof ref);
  memcpy (diff_last, ref, sizeof ref);
  diff_first[0] ^= 0x01;
  diff_last[sizeof ref - 1] ^= 0x01;

  d_equal.with_len = 0;
  d_equal.p1 = ref; d_equal.len1 = sizeof ref;
  d_equal.p2 = equal; d_equal.len2 = sizeof ref;

  d_first = d_equal; d_first.p2 = diff_first;
  d_last  = d_equal; d_last.p2  = diff_last;

  /* Positive control: comparing 32 bytes must be measurably faster
     than comparing 128 bytes.  If the environment is too noisy to
     detect even this, skip the leakage assertions instead of failing
     spuriously.  */
  d_short = d_equal; d_short.len1 = 32;
  d_long  = d_equal; d_long.len1  = 128;
  if (!distinguishable (&d_long, &d_short, "control: 128 vs 32 bytes"))
    {
      info ("timing resolution too low; skipping leakage assertions\n");
      return;
    }

  /* An early-exit comparison would be much faster for buffers
     differing at the first byte.  */
  if (distinguishable (&d_equal, &d_first, "equal vs diff-at-first-byte"))
    {
      fail ("gcry_ct_memequal: timing leaks whether buffers differ"
            " at the first byte\n");
      leaky = 1;
    }
  if (distinguishable (&d_last, &d_first, "diff-at-last vs diff-at-first"))
    {
      fail ("gcry_ct_memequal: timing leaks the position of the first"
            " differing byte\n");
      leaky = 1;
    }

  /* Length leakage: gcry_ct_memequal_len must not return early on a
     length mismatch, so (64 vs 64) and (64 vs 128) take the same
     time.  */
  d_len64.with_len = 1;
  d_len64.p1 = ref; d_len64.len1 = 64;
  d_len64.p2 = equal; d_len64.len2 = 64;
  d_len128 = d_len64; d_len128.len2 = 128;
  if (distinguishable (&d_len64, &d_len128, "len 64/64 vs 64/128"))
    {
      fail ("gcry_ct_memequal_len: timing leaks a length mismatch\n");
      leaky = 1;
    }

  if (!leaky && verbose)
    info ("constant-time timing checks OK\n");
}


int
main (int argc, char **argv)
{
  if (argc > 1 && !strcmp (argv[1], "--verbose"))
    verbose = 1;

  if (!gcry_check_version (GCRYPT_VERSION))
    die ("version mismatch\n");
  xgcry_control ((GCRYCTL_DISABLE_SECMEM, 0));
  xgcry_control ((GCRYCTL_ENABLE_QUICK_RANDOM, 0));
  xgcry_control ((GCRYCTL_INITIALIZATION_FINISHED, 0));

  test_memequal ();
  test_memequal_len ();
  test_timing ();

  if (error_count)
    die ("%d errors detected\n", error_count);
  return 0;
}
