use glam::{Vec2, Vec3, Vec3Swizzles};
use rayon::prelude::*;

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use crate::emit::{self, EMIT_BATCHES_PER_SEC};
use crate::obstacles::{
    self, build as build_obstacles, scene_is_animated, Obstacle, OBSTACLE_SCENE,
};
use crate::{HEIGHT, WIDTH};

/// Per-stage microsecond accumulators. Always recorded; `print_breakdown`
/// dumps them when `BENCH_BREAKDOWN=1`. `update_pos` covers the fused
/// update + constraint pass.
pub static T_UPDATE_POS: AtomicU64 = AtomicU64::new(0);
pub static T_FILL_BINS: AtomicU64 = AtomicU64::new(0);
pub static T_CHECK_COLLISIONS: AtomicU64 = AtomicU64::new(0);

pub fn reset_breakdown() {
    T_UPDATE_POS.store(0, Relaxed);
    T_FILL_BINS.store(0, Relaxed);
    T_CHECK_COLLISIONS.store(0, Relaxed);
}

pub fn print_breakdown(frames: usize) {
    let up = T_UPDATE_POS.load(Relaxed);
    let fb = T_FILL_BINS.load(Relaxed);
    let cc = T_CHECK_COLLISIONS.load(Relaxed);
    let total = up + fb + cc;
    let pct = |v: u64| {
        if total == 0 {
            0.0
        } else {
            100.0 * v as f64 / total as f64
        }
    };
    let per_frame = |v: u64| v / frames.max(1) as u64;
    eprintln!(
        "       breakdown (us/frame, avg over {frames}): update_pos={up_a} ({up_p:.1}%)  fill_bins={fb_a} ({fb_p:.1}%)  check_collisions={cc_a} ({cc_p:.1}%)",
        up_a = per_frame(up), up_p = pct(up),
        fb_a = per_frame(fb), fb_p = pct(fb),
        cc_a = per_frame(cc), cc_p = pct(cc),
    );
}

pub const MAX_PARTICLES: usize = 150000;
pub(crate) const MAX_RADIUS: f32 = 2.0;
pub const NB_THREAD: usize = 12;
pub(crate) const BIN_SIZE: usize = 4;
pub const BIN_W: usize = WIDTH / BIN_SIZE;
pub(crate) const BIN_H: usize = HEIGHT / BIN_SIZE;
pub(crate) const NUM_BIN: usize = BIN_W * BIN_H;
/// Inner-border thickness in world units. The wall constraint and pair
/// collision response need a non-zero buffer so particles can resolve
/// overlaps in 2D near walls instead of collapsing into a 1D line. The
/// renderer hides this buffer by zooming the camera to the playable area.
pub const BORDER_PADDING: f32 = 15.0;

/// Gravity acceleration in world units / s². Pure downward — any non-zero X
/// component shows up as a steady rightward (or leftward) drift that becomes
/// dominant once the floor saturates and particles can no longer dissipate
/// the lateral velocity through stacking gaps.
const GRAVITY: Vec2 = Vec2::new(0.0, 200.0);
/// Pushback factor for wall constraints. <1 leaves a residual overlap for the
/// next substep to clean up — softer visuals, more stable.
const WALL_DAMPING: f32 = 0.75;
/// Pair-collision response: each particle absorbs half the overlap, scaled by
/// `WALL_DAMPING` for consistency with the wall response.
const COLLISION_RESPONSE: f32 = 0.5 * WALL_DAMPING;

/// Smoothing radius for viscosity, in world units. Pairs farther apart than
/// this contribute nothing. Tied to `2 * BIN_SIZE` since that's what the
/// bin-pair iteration in `check_collisions` already covers — going wider
/// would force a 5×5 sweep instead of 3×3 and ~2.7× the pair work.
const SMOOTH_H: f32 = 2.0 * BIN_SIZE as f32;
const SMOOTH_H2: f32 = SMOOTH_H * SMOOTH_H;
const INV_SMOOTH_H: f32 = 1.0 / SMOOTH_H;
/// Viscosity strength. Each pair within `SMOOTH_H` averages a fraction of
/// their relative velocity into `last_pos` (Verlet velocity = pos - last_pos),
/// weighted by a quadratic falloff. 0.0 disables — and when disabled, all
/// the supporting infrastructure (`sorted_last_pos`, `last_delta_buf`, the
/// extra slice carving and inner-loop branch) is compile-time folded away
/// via `VISCOSITY_ENABLED`. Practical upper bound is around 0.3 — past that,
/// dense regions accumulate enough correction to oscillate or escape walls.
/// Stable up to 1.0 for sparse fluids.
const VISCOSITY_K: f32 = 0.02;
/// Compile-time master switch derived from `VISCOSITY_K`. The const-folder
/// turns every `if VISCOSITY_ENABLED { … }` into either the body or nothing,
/// so disabling viscosity is a true zero-cost change with no extra branches
/// or memory traffic.
const VISCOSITY_ENABLED: bool = VISCOSITY_K > 0.0;

const _: () = assert!((WIDTH / BIN_SIZE).is_multiple_of(NB_THREAD));
const _: () = assert!(MAX_RADIUS * 2.0 <= BIN_SIZE as f32);
const _: () = assert!(VISCOSITY_K >= 0.0 && VISCOSITY_K <= 1.0);

/// Bin-sorted particle slot on the collision hot path: 16 B (Vec3 pos + u32
/// original index). Using `u32` instead of `usize` avoids the 24 B alignment
/// padding `(Vec3, usize)` would cause, cutting memory traffic in
/// `check_collisions` by ~33%.
type SortedSlot = (Vec3, u32);

pub struct Bin {
    pub indexes: Vec<usize>,
}

impl Bin {
    fn new() -> Self {
        Bin {
            indexes: Vec::with_capacity(15),
        }
    }
}

pub struct Physics {
    len: usize,
    pos: Vec<Vec3>,
    last_pos: Vec<Vec2>,
    sorted_pos: Vec<SortedSlot>,
    /// `last_pos` scattered into bin-sorted order alongside `sorted_pos`.
    /// Lets the viscosity pass read each pair's velocity (= pos - last_pos)
    /// sequentially in the same hot loop instead of chasing random
    /// `last_pos[orig_idx]` indices.
    sorted_last_pos: Vec<Vec2>,
    /// Per-slot position-delta accumulator for Jacobi-style pair resolution.
    /// During `check_collisions` we read positions from `sorted_pos` and
    /// write deltas here, then add them back at the end. This decouples the
    /// solver from update ordering, killing the rightward Gauss-Seidel
    /// drift that's otherwise visible at saturation density.
    delta_buf: Vec<Vec2>,
    /// Per-slot last_pos-delta accumulator for the viscosity pass. Same
    /// Jacobi pattern as `delta_buf` but feeds back into `last_pos`.
    last_delta_buf: Vec<Vec2>,
    /// View of `sorted_pos` as `Vec<Bin>`, rebuilt lazily on demand for the UI.
    bins: Vec<Bin>,
    bins_dirty: bool,
    bin_start: Vec<usize>,
    bin_cursor: Vec<usize>,
    /// Per-particle bin index, computed once in `fill_bins` and reused for
    /// histogram + scatter (saves recomputing `(x*inv_bin)*BIN_H + y*inv_bin`).
    bin_idx: Vec<u32>,
    // --- Cold fields below: only touched on emit, never on the stepping hot path. ---
    /// Pre-generated radii for not-yet-emitted slots; `try_add_object`
    /// consumes `pending_radii[len]` and copies it into `pos[len].z`.
    /// Pre-generating keeps RNG ordering deterministic across runs.
    pending_radii: Vec<f32>,
    /// Fractional batch accumulator for `emit_flow_for`.
    emit_acc: f32,
    /// Active obstacle list, rebuilt from `OBSTACLE_SCENE` at construction.
    /// Animated scenes (rotating bar) overwrite entries each step.
    obstacles: Vec<Obstacle>,
    /// Per-bin obstacle indices (into `self.obstacles`). Built once for static
    /// scenes, rebuilt every step for animated ones, then read in the hot
    /// per-particle loop so each ball only tests obstacles it can actually
    /// touch — turning an O(N·M) loop into O(N · m_local).
    obstacle_bins: Vec<Vec<u32>>,
    /// Sim-time elapsed in seconds, accumulated from `step(dt)`. Drives all
    /// time-dependent obstacle/emitter scenes deterministically.
    sim_time: f32,
    /// Number of `emit_flow` calls so far. Drives the pulse/burst emitters.
    emit_count: u32,
}

impl Physics {
    pub fn new() -> Physics {
        // Pre-draw every radius up front so RNG ordering is deterministic
        // regardless of when each slot is later emitted by `try_add_object`.
        let pending_radii = (0..MAX_PARTICLES)
            .map(|_| quad_rand::gen_range(MAX_RADIUS * 0.5, MAX_RADIUS * 0.6))
            .collect();

        let bins = (0..NUM_BIN).map(|_| Bin::new()).collect();

        rayon::ThreadPoolBuilder::new()
            .num_threads(NB_THREAD)
            .build_global()
            .unwrap();

        let obstacles = build_obstacles(OBSTACLE_SCENE, 0.0);
        let mut obstacle_bins = vec![Vec::new(); NUM_BIN];
        obstacles::rebuild_bins(&obstacles, &mut obstacle_bins);

        Physics {
            len: 0,
            pos: vec![Vec3::ZERO; MAX_PARTICLES],
            last_pos: vec![Vec2::ZERO; MAX_PARTICLES],
            sorted_pos: vec![(Vec3::ZERO, 0u32); MAX_PARTICLES],
            sorted_last_pos: if VISCOSITY_ENABLED {
                vec![Vec2::ZERO; MAX_PARTICLES]
            } else {
                Vec::new()
            },
            delta_buf: vec![Vec2::ZERO; MAX_PARTICLES],
            last_delta_buf: if VISCOSITY_ENABLED {
                vec![Vec2::ZERO; MAX_PARTICLES]
            } else {
                Vec::new()
            },
            bins,
            bins_dirty: true,
            bin_start: vec![0; NUM_BIN + 1],
            bin_cursor: vec![0; NUM_BIN],
            bin_idx: vec![0u32; MAX_PARTICLES],
            pending_radii,
            emit_acc: 0.0,
            obstacles,
            obstacle_bins,
            sim_time: 0.0,
            emit_count: 0,
        }
    }

    /// Fused integration + wall/obstacle constraint. One pass keeps each
    /// particle's data hot in L1 across both phases.
    fn update_and_constrain(&mut self, dt: f32) {
        let factor = WALL_DAMPING;
        let acc = GRAVITY * (dt * dt);
        let xmax_base = WIDTH as f32 - BORDER_PADDING;
        let ymax_base = HEIGHT as f32 - BORDER_PADDING;
        let xmin_base = BORDER_PADDING;
        let ymin_base = BORDER_PADDING;

        // Refresh animated obstacles before the per-particle pass (rotating
        // bars rebuild their endpoints against the current sim_time), and
        // re-bin them so the lookup below stays in sync.
        if scene_is_animated(OBSTACLE_SCENE) {
            self.obstacles = build_obstacles(OBSTACLE_SCENE, self.sim_time);
            obstacles::rebuild_bins(&self.obstacles, &mut self.obstacle_bins);
        }
        let obstacles = &self.obstacles;
        let obstacle_bins = &self.obstacle_bins;
        let inv_bin = 1.0 / BIN_SIZE as f32;
        let bin_w_u32 = BIN_W as u32;
        let bin_h_u32 = BIN_H as u32;

        for i in 0..self.len {
            let mut p = self.pos[i];
            let r = p.z;

            let prev = self.last_pos[i];
            let cur_xy = p.xy();
            let diff = cur_xy - prev;
            self.last_pos[i] = cur_xy;
            let mut np = cur_xy + diff + acc;

            // Look up only the obstacles registered to this particle's bin.
            // `MAX_RADIUS * 2 ≤ BIN_SIZE` means a ball never spans more than
            // 2 bins on any axis; padding obstacle AABBs by `MAX_RADIUS` when
            // we register them guarantees that any obstacle the ball can
            // touch from `np` is registered in `np`'s bin.
            let bx = (np.x * inv_bin) as u32;
            let by = (np.y * inv_bin) as u32;
            if bx < bin_w_u32 && by < bin_h_u32 {
                let bi = (bx * bin_h_u32 + by) as usize;
                for &k in &obstacle_bins[bi] {
                    np = obstacles::resolve(np, r, &obstacles[k as usize]);
                }
            }

            let xmax = xmax_base - r;
            let xmin = xmin_base + r;
            let ymax = ymax_base - r;
            let ymin = ymin_base + r;
            let dx_over = (np.x - xmax).max(0.0);
            let dx_under = (xmin - np.x).max(0.0);
            let dy_over = (np.y - ymax).max(0.0);
            let dy_under = (ymin - np.y).max(0.0);
            np.x += factor * (dx_under - dx_over);
            np.y += factor * (dy_under - dy_over);

            // Hard clamp to the playable area as a last-resort safety net.
            // The soft constraint above leaves a residual overshoot (1-factor),
            // which is fine for physics but lets `fill_bins` panic with an
            // out-of-range bin index if a particle ever reaches the absolute
            // world bounds (e.g. from a misconfigured fluid force). Pinning
            // here costs one min/max per axis and guarantees `0 <= bin < BIN_*`.
            np.x = np.x.clamp(xmin, xmax);
            np.y = np.y.clamp(ymin, ymax);

            p.x = np.x;
            p.y = np.y;
            self.pos[i] = p;
        }
    }

    fn fill_bins(&mut self) {
        // Counting sort with `bin_start` as the prefix-sum array. The
        // `Vec<Bin>` view is rebuilt lazily on demand in `get_bins()`.
        self.bins_dirty = true;

        let inv_bin = 1.0 / BIN_SIZE as f32;

        // Compute bin indices in a separate pass first: the random scatter into
        // `bin_start` blocks the auto-vectoriser, but isolating the float→bin
        // math (cast via u32 to keep the SIMD width at 4 lanes) lets it run.
        let pos_slice = &self.pos[..self.len];
        let bin_idx = &mut self.bin_idx[..self.len];
        let bin_h_u32 = BIN_H as u32;
        for (i, p) in pos_slice.iter().enumerate() {
            let bx = (p.x * inv_bin) as u32;
            let by = (p.y * inv_bin) as u32;
            bin_idx[i] = by + bx * bin_h_u32;
        }

        for c in &mut self.bin_start[..NUM_BIN] {
            *c = 0;
        }
        for &b in bin_idx.iter() {
            self.bin_start[b as usize] += 1;
        }

        // Exclusive prefix sum in-place; sentinel at [NUM_BIN] = total.
        let mut acc = 0usize;
        for c in &mut self.bin_start[..NUM_BIN] {
            let v = *c;
            *c = acc;
            acc += v;
        }
        self.bin_start[NUM_BIN] = acc;
        debug_assert_eq!(acc, self.len);

        self.bin_cursor.copy_from_slice(&self.bin_start[..NUM_BIN]);
        if VISCOSITY_ENABLED {
            for i in 0..self.len {
                let b = self.bin_idx[i] as usize;
                let slot = self.bin_cursor[b];
                self.bin_cursor[b] = slot + 1;
                self.sorted_pos[slot] = (self.pos[i], i as u32);
                self.sorted_last_pos[slot] = self.last_pos[i];
            }
        } else {
            for i in 0..self.len {
                let b = self.bin_idx[i] as usize;
                let slot = self.bin_cursor[b];
                self.bin_cursor[b] = slot + 1;
                self.sorted_pos[slot] = (self.pos[i], i as u32);
            }
        }
    }

    /// Materialise `self.bins` from the compact (sorted_pos, bin_start) layout
    /// for UI consumers.
    fn rebuild_bins(&mut self) {
        for (bi, bin) in self.bins.iter_mut().enumerate() {
            bin.indexes.clear();
            let a = self.bin_start[bi];
            let b = self.bin_start[bi + 1];
            bin.indexes
                .extend(self.sorted_pos[a..b].iter().map(|sp| sp.1 as usize));
        }
        self.bins_dirty = false;
    }

    fn check_collisions(&mut self) {
        let chunk_size = NUM_BIN / NB_THREAD;
        let thread_width = BIN_W / NB_THREAD;

        // Stack array avoids a per-substep Vec alloc.
        let mut breakpoints_thread = [0usize; NB_THREAD];
        for (k, slot) in breakpoints_thread.iter_mut().enumerate() {
            *slot = unsafe { *self.bin_start.get_unchecked(k * chunk_size) };
        }

        // Jacobi pair resolution: zero the delta accumulator(s), then run
        // the parallel pair pass writing to `delta_buf` (and
        // `last_delta_buf` when viscosity is on) while reading immutably
        // from `sorted_pos` (and `sorted_last_pos`). The strip carving is
        // identical for both buffer pairs so no two threads ever touch
        // the same slot. After the pass, fold deltas back into the source
        // buffers. See the `delta_buf` field doc for why this is Jacobi
        // rather than Gauss-Seidel.
        for d in &mut self.delta_buf[..self.len] {
            *d = Vec2::ZERO;
        }
        if VISCOSITY_ENABLED {
            for d in &mut self.last_delta_buf[..self.len] {
                *d = Vec2::ZERO;
            }
        }

        // SAFETY (for the `get_unchecked` reads/writes below):
        // - `bin_start` has length `NUM_BIN + 1`, and every `bin_start[k]`
        //   access uses `k < NUM_BIN + 1`. The maximum we touch is
        //   `bin_start[col_b + 2]` where `col_b = bin1 + BIN_H` and
        //   `bin1 < NUM_BIN - 1`, so `col_b + 2 <= NUM_BIN + 1`.
        // - The four slice arguments cover exactly the contiguous bins
        //   assigned to this strip, with `offset = bin_start[first bin]`.
        //   Every i we index into them (`i1`, `i2`) comes from
        //   `bin_start[bin]` for a bin in this strip's range, so
        //   `offset <= i < offset + slice_pos.len()`.
        let check_slice = |slice_pos: &[SortedSlot],
                           slice_last: &[Vec2],
                           slice_delta: &mut [Vec2],
                           slice_last_delta: &mut [Vec2],
                           offset: usize,
                           start_x: usize,
                           width: usize,
                           wall: usize| {
            if slice_pos.is_empty() {
                return;
            }
            debug_assert_eq!(slice_pos.len(), slice_delta.len());
            debug_assert_eq!(slice_pos.len(), slice_last.len());
            debug_assert_eq!(slice_pos.len(), slice_last_delta.len());
            let bin_start = self.bin_start.as_slice();
            let x_lo = start_x + wall;
            let x_hi = (start_x + width) - wall;
            let y_lo = 1usize;
            let y_hi = BIN_H - 1;
            for bin_x in x_lo..x_hi {
                let col_base = bin_x * BIN_H;
                for bin_y in y_lo..y_hi {
                    let bin1 = col_base + bin_y;
                    // SAFETY: bin1+1 <= NUM_BIN (bin1 < NUM_BIN-1 because bin_y < BIN_H-1).
                    let i1_lo = unsafe { *bin_start.get_unchecked(bin1) };
                    let i1_hi = unsafe { *bin_start.get_unchecked(bin1 + 1) };
                    if i1_lo == i1_hi {
                        continue;
                    }

                    // Forward neighbours have 4 candidate bins with bin2 > bin1: the
                    // same-column bin1+1, plus the 3-row span bin1+BIN_H-1..=bin1+BIN_H+1.
                    // The right-column trio is contiguous in `bin_start`, so we collapse
                    // it into a single range — 2 range lookups instead of 4.
                    let row_lo = unsafe { *bin_start.get_unchecked(bin1 + 1) };
                    let row_hi = unsafe { *bin_start.get_unchecked(bin1 + 2) };
                    let col_b = bin1 + BIN_H;
                    let col_lo = unsafe { *bin_start.get_unchecked(col_b - 1) };
                    let col_hi = unsafe { *bin_start.get_unchecked(col_b + 2) };
                    let mut fwd_ranges: [(usize, usize); 2] = [(0, 0); 2];
                    let mut fwd_n = 0usize;
                    if row_lo < row_hi {
                        fwd_ranges[fwd_n] = (row_lo, row_hi);
                        fwd_n += 1;
                    }
                    if col_lo < col_hi {
                        fwd_ranges[fwd_n] = (col_lo, col_hi);
                        fwd_n += 1;
                    }

                    for i1 in i1_lo..i1_hi {
                        debug_assert!(i1 >= offset && i1 - offset < slice_pos.len());
                        let i1_local = i1 - offset;
                        let (pos1, _) = unsafe { *slice_pos.get_unchecked(i1_local) };
                        let pos1_z = pos1.z;
                        // Viscosity needs each particle's velocity; collision
                        // doesn't, so skip the load + subtract entirely when
                        // viscosity is compile-time off.
                        let vel1 = if VISCOSITY_ENABLED {
                            let last1 = unsafe { *slice_last.get_unchecked(i1_local) };
                            pos1.xy() - last1
                        } else {
                            Vec2::ZERO
                        };
                        // Accumulate i1's deltas in registers and fold them
                        // into the shared buffers once per i1, saving N-1
                        // stores per pair list.
                        let mut delta1 = Vec2::ZERO;
                        let mut last_delta1 = Vec2::ZERO;

                        // Body inlined below. We visit the same-bin range
                        // [i1+1, i1_hi) and the up-to-2 forward-bin ranges
                        // back-to-back, all with the same per-pair logic.
                        let mut process_pair = |i2: usize| {
                            let i2_local = i2 - offset;
                            let (pos2, _) = unsafe { *slice_pos.get_unchecked(i2_local) };
                            let v = pos1.xy() - pos2.xy();
                            let dist2 = v.length_squared();
                            let min_dist = pos1_z + pos2.z;
                            if dist2 < min_dist * min_dist {
                                // Hard collision: pushes pair apart so they
                                // stop overlapping. Folded form turns the
                                // 1×sqrt + 3×fdiv body into 1×sqrt + 1×fdiv.
                                let dist = dist2.sqrt();
                                let dn = COLLISION_RESPONSE * (dist - min_dist) / (dist * min_dist);
                                let v_scaled = v * dn;
                                delta1 -= v_scaled * pos2.z;
                                let i2_delta = v_scaled * pos1_z;
                                let slot = unsafe { slice_delta.get_unchecked_mut(i2_local) };
                                *slot += i2_delta;
                            } else if VISCOSITY_ENABLED && dist2 < SMOOTH_H2 {
                                // Viscosity: damp relative velocity by
                                // pulling each particle's `last_pos` toward
                                // the local mean. Velocity in Verlet is
                                // `pos - last_pos`, so adding `+α·rel_v/2`
                                // to last1 reduces vel1 by that amount.
                                // Quadratic falloff (max at touching, 0 at
                                // SMOOTH_H) so dense neighbour clusters
                                // can't sum to a runaway correction.
                                let dist = dist2.sqrt();
                                let s = (SMOOTH_H - dist) * INV_SMOOTH_H;
                                let visc_kernel = s * s;
                                let last2 = unsafe { *slice_last.get_unchecked(i2_local) };
                                let vel2 = pos2.xy() - last2;
                                let rel_v = vel1 - vel2;
                                let visc = VISCOSITY_K * visc_kernel * 0.5;
                                last_delta1 += rel_v * visc;
                                let lslot =
                                    unsafe { slice_last_delta.get_unchecked_mut(i2_local) };
                                *lslot -= rel_v * visc;
                            }
                        };

                        // Same-bin: start at i1+1 so i2 > i1 is guaranteed.
                        for i2 in (i1 + 1)..i1_hi {
                            process_pair(i2);
                        }
                        // Forward bins: bin2 > bin1 by construction.
                        for &(lo, hi) in &fwd_ranges[..fwd_n] {
                            for i2 in lo..hi {
                                process_pair(i2);
                            }
                        }

                        if delta1 != Vec2::ZERO {
                            let slot = unsafe { slice_delta.get_unchecked_mut(i1_local) };
                            *slot += delta1;
                        }
                        if VISCOSITY_ENABLED && last_delta1 != Vec2::ZERO {
                            let slot = unsafe { slice_last_delta.get_unchecked_mut(i1_local) };
                            *slot += last_delta1;
                        }
                    }
                }
            }
        };

        // Carve `sorted_pos` (read-only) and `delta_buf` (write-only) into
        // NB_THREAD disjoint sub-slices using identical breakpoints. When
        // viscosity is on, carve `sorted_last_pos` / `last_delta_buf` in
        // lock-step too — otherwise the viscosity buffers are empty and we
        // pass a `&[]` / `&mut []` placeholder (the inner branch is
        // compile-time off so the slice is never read). Rayon processes
        // the strips in parallel without a Mutex.
        type MainSub<'a> = (&'a [SortedSlot], &'a [Vec2], &'a mut [Vec2], &'a mut [Vec2], usize, usize);
        let sorted_pos: &[SortedSlot] = &self.sorted_pos[..];
        let sorted_last: &[Vec2] = if VISCOSITY_ENABLED {
            &self.sorted_last_pos[..]
        } else {
            &[]
        };
        let mut main_subs: Vec<MainSub> = Vec::with_capacity(NB_THREAD);
        let mut delta_rest: &mut [Vec2] = &mut self.delta_buf[..];
        let mut last_delta_rest: &mut [Vec2] = if VISCOSITY_ENABLED {
            &mut self.last_delta_buf[..]
        } else {
            &mut []
        };
        for slice_i in 0..NB_THREAD {
            let a = breakpoints_thread[slice_i];
            let b = if slice_i + 1 < NB_THREAD {
                breakpoints_thread[slice_i + 1]
            } else {
                self.len
            };
            let skip = a - main_subs.last().map(|s| s.4 + s.0.len()).unwrap_or(0);
            let (_, dafter) = delta_rest.split_at_mut(skip);
            let (dsub, dtail) = dafter.split_at_mut(b - a);
            let (lp_sub, lsub): (&[Vec2], &mut [Vec2]) = if VISCOSITY_ENABLED {
                let (_, lafter) = last_delta_rest.split_at_mut(skip);
                let (lsub, ltail) = lafter.split_at_mut(b - a);
                last_delta_rest = ltail;
                (&sorted_last[a..b], lsub)
            } else {
                (&[], &mut [])
            };
            main_subs.push((&sorted_pos[a..b], lp_sub, dsub, lsub, a, slice_i * thread_width));
            delta_rest = dtail;
        }
        main_subs
            .into_par_iter()
            .for_each(|(sp, sl, sd, sld, offset, start_x)| {
                check_slice(sp, sl, sd, sld, offset, start_x, thread_width, 1);
            });

        // Border pass: each strip processes the 2-column seam (start_x, start_x+1)
        // between two main-pass slices, needing 3×3 neighbours so the bin range is
        // [(start_x-1)*BIN_H, (start_x+3)*BIN_H). Adjacent border strips are spaced
        // `thread_width` columns apart (must be > 4 for non-overlap, asserted via
        // `BIN_W % NB_THREAD == 0`), so we carve disjoint sub-slices in parallel.
        let mut strips: Vec<(usize, usize, usize)> = Vec::with_capacity(NB_THREAD - 1);
        for slice_i in 0..(NB_THREAD - 1) {
            let start_x = (slice_i + 1) * thread_width - 1;
            let bin_lo = start_x.saturating_sub(1) * BIN_H;
            let bin_hi = (start_x + 3) * BIN_H;
            if bin_hi > NUM_BIN {
                continue;
            }
            let a = self.bin_start[bin_lo];
            let b = self.bin_start[bin_hi];
            if a < b {
                strips.push((a, b, start_x));
            }
        }
        // Strips are in increasing a-order; carve out non-overlapping
        // sub-slices for the delta buffers in lock-step (and the viscosity
        // buffers too when enabled — otherwise they're empty placeholders).
        let mut delta_rest: &mut [Vec2] = &mut self.delta_buf[..];
        let mut last_delta_rest: &mut [Vec2] = if VISCOSITY_ENABLED {
            &mut self.last_delta_buf[..]
        } else {
            &mut []
        };
        let mut rest_off: usize = 0;
        let mut sub_slices: Vec<MainSub> = Vec::with_capacity(strips.len());
        for &(a, b, start_x) in &strips {
            let skip = a - rest_off;
            let (_, dafter) = delta_rest.split_at_mut(skip);
            let (dsub, dtail) = dafter.split_at_mut(b - a);
            let (lp_sub, lsub): (&[Vec2], &mut [Vec2]) = if VISCOSITY_ENABLED {
                let (_, lafter) = last_delta_rest.split_at_mut(skip);
                let (lsub, ltail) = lafter.split_at_mut(b - a);
                last_delta_rest = ltail;
                (&sorted_last[a..b], lsub)
            } else {
                (&[], &mut [])
            };
            sub_slices.push((&sorted_pos[a..b], lp_sub, dsub, lsub, a, start_x));
            delta_rest = dtail;
            rest_off = b;
        }
        sub_slices
            .into_par_iter()
            .for_each(|(sp, sl, sd, sld, offset, start_x)| {
                check_slice(sp, sl, sd, sld, offset, start_x, 2, 0);
            });

        // Fold the delta buffer(s) back: position deltas land in
        // `sorted_pos` -> `pos` (collision). When viscosity is on, also
        // fold `last_delta_buf` into `last_pos` keyed by the particle's
        // original id. The two passes are split so the disabled path
        // doesn't even touch `last_pos`.
        if VISCOSITY_ENABLED {
            for slot_i in 0..self.len {
                let sp = &mut self.sorted_pos[slot_i];
                let d = self.delta_buf[slot_i];
                sp.0.x += d.x;
                sp.0.y += d.y;
                let id = sp.1 as usize;
                self.pos[id] = sp.0;
                self.last_pos[id] += self.last_delta_buf[slot_i];
            }
        } else {
            for slot_i in 0..self.len {
                let sp = &mut self.sorted_pos[slot_i];
                let d = self.delta_buf[slot_i];
                sp.0.x += d.x;
                sp.0.y += d.y;
                let id = sp.1 as usize;
                self.pos[id] = sp.0;
            }
        }
    }

    pub fn step(&mut self, dt: f32) {
        self.sim_time += dt;

        let t = Instant::now();
        self.update_and_constrain(dt);
        T_UPDATE_POS.fetch_add(t.elapsed().as_micros() as u64, Relaxed);

        let t = Instant::now();
        self.fill_bins();
        T_FILL_BINS.fetch_add(t.elapsed().as_micros() as u64, Relaxed);

        let t = Instant::now();
        self.check_collisions();
        T_CHECK_COLLISIONS.fetch_add(t.elapsed().as_micros() as u64, Relaxed);
    }

    /// Soft repel against an external collider (mouse cursor). Mirrors the
    /// circle branch of `obstacles::resolve` but operates in-place on every
    /// particle.
    pub fn avoid_obstacle(&mut self, pos: Vec2, size: f32) {
        for i in 0..self.len {
            let p = self.pos[i];
            let v = p.xy() - pos;
            let dist2 = v.length_squared();
            let min_dist = p.z + size;
            if dist2 < min_dist * min_dist {
                let dist = dist2.sqrt();
                let n = v / dist;
                self.pos[i] -= (n * obstacles::OBSTACLE_PUSH * (dist - min_dist)).extend(0.0);
            }
        }
    }

    /// Append a new particle with the next pre-drawn radius. Returns `false`
    /// when the buffer is full so emitters can stop early. Used by `emit::*`.
    pub(crate) fn try_add_object(&mut self, pos: Vec2, vel: Vec2) -> bool {
        if self.len >= MAX_PARTICLES {
            return false;
        }
        let radius = self.pending_radii[self.len];
        self.pos[self.len] = pos.extend(radius);
        self.last_pos[self.len] = pos - vel;
        self.len += 1;
        true
    }

    /// Rate-decoupled spawn driver: accumulates `dt * EMIT_BATCHES_PER_SEC`
    /// and drains whole batches. Spawn cadence stays tied to wall-clock time
    /// regardless of the caller's frame rate.
    pub fn emit_flow_for(&mut self, dt: f32) {
        self.emit_acc += dt * EMIT_BATCHES_PER_SEC;
        while self.emit_acc >= 1.0 {
            self.emit_flow();
            self.emit_acc -= 1.0;
        }
    }

    /// Emit one flow batch (~48 particles for the basic emitters). The active
    /// `EMIT_SCENE` decides the batch shape; bench drives it directly, GUI
    /// goes through `emit_flow_for` for fps-independence.
    pub fn emit_flow(&mut self) {
        let count = self.emit_count;
        self.emit_count = self.emit_count.wrapping_add(1);
        emit::dispatch(self, count);
    }

    pub fn nb_particles(&self) -> usize {
        self.len
    }

    pub fn get_bins(&mut self) -> &[Bin] {
        if self.bins_dirty {
            self.rebuild_bins();
        }
        &self.bins
    }

    /// Live particle slice, length = `nb_particles()`. Use this for rendering
    /// — the underlying buffer is sized to `MAX_PARTICLES` but only the prefix
    /// up to `len` is meaningful.
    pub fn active_points(&self) -> &[Vec3] {
        &self.pos[..self.len]
    }

    /// Renderable circle list (centre.xy, radius) for the active obstacle
    /// scene. Rects go through `obstacle_rects` and a separate pipeline.
    /// Animated scenes refresh per `step()`, so callers should re-read this
    /// each frame they want to draw.
    pub fn obstacle_circles(&self) -> Vec<Vec3> {
        obstacles::circles_for(&self.obstacles)
    }

    /// Per-instance attributes for the oriented-rect render pipeline.
    /// Returns an empty `Vec` if the active scene has no rects.
    pub fn obstacle_rects(&self) -> Vec<obstacles::RectInstance> {
        obstacles::rects_for(&self.obstacles)
    }
}
