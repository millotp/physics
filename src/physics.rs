use glam::{vec2, vec3, Vec2, Vec3, Vec3Swizzles};
use rayon::prelude::*;

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use crate::{HEIGHT, WIDTH};

/// Per-stage microsecond accumulators. Zero-cost in release unless
/// `BENCH_BREAKDOWN=1` is set at runtime (the timers are always recorded,
/// `print_breakdown` reads them on demand).
pub static T_UPDATE_POS: AtomicU64 = AtomicU64::new(0);
pub static T_APPLY_CONSTRAINT: AtomicU64 = AtomicU64::new(0);
pub static T_FILL_BINS: AtomicU64 = AtomicU64::new(0);
pub static T_CHECK_COLLISIONS: AtomicU64 = AtomicU64::new(0);

pub fn reset_breakdown() {
    T_UPDATE_POS.store(0, Relaxed);
    T_APPLY_CONSTRAINT.store(0, Relaxed);
    T_FILL_BINS.store(0, Relaxed);
    T_CHECK_COLLISIONS.store(0, Relaxed);
}

pub fn print_breakdown(frames: usize) {
    let up = T_UPDATE_POS.load(Relaxed);
    let ac = T_APPLY_CONSTRAINT.load(Relaxed);
    let fb = T_FILL_BINS.load(Relaxed);
    let cc = T_CHECK_COLLISIONS.load(Relaxed);
    let total = up + ac + fb + cc;
    let pct = |v: u64| if total == 0 { 0.0 } else { 100.0 * v as f64 / total as f64 };
    eprintln!(
        "       breakdown (us/frame, avg over {frames}): update_pos={up_a} ({up_p:.1}%)  apply_constraint={ac_a} ({ac_p:.1}%)  fill_bins={fb_a} ({fb_p:.1}%)  check_collisions={cc_a} ({cc_p:.1}%)",
        up_a = up / frames.max(1) as u64, up_p = pct(up),
        ac_a = ac / frames.max(1) as u64, ac_p = pct(ac),
        fb_a = fb / frames.max(1) as u64, fb_p = pct(fb),
        cc_a = cc / frames.max(1) as u64, cc_p = pct(cc),
    );
}

pub const MAX_PARTICLES: usize = 70000;
const MAX_RADIUS: f32 = 2.5;
pub const NB_THREAD: usize = 12;
const BIN_SIZE: usize = 5;
pub const BIN_W: usize = WIDTH / BIN_SIZE;
const BIN_H: usize = HEIGHT / BIN_SIZE;
const NUM_BIN: usize = BIN_W * BIN_H;
const BORDER_PADDING: f32 = 100.0;
const OBSTACLE_POS: Vec2 = vec2(850.0, 600.0);
const OBSTACLE_PADDING: f32 = 100.0;

const _: () = assert!((WIDTH / BIN_SIZE).is_multiple_of(NB_THREAD));
const _: () = assert!(MAX_RADIUS * 2.0 <= BIN_SIZE as f32);

/// One bin-sorted particle slot on the collision hot path.
///
/// `(Vec3, u32)` is laid out as 16 B (12 B for position + 4 B for the original
/// particle index) — vs. the previous `(Vec3, usize)` which alignment-padded to
/// 24 B. That cuts memory traffic in `check_collisions` by ~33% for the same
/// number of neighbour reads. The 32-bit index is enough since MAX_PARTICLES
/// fits well under `u32::MAX`.
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
    /// Bin-sorted particle data on the collision hot path. See `SortedSlot`
    /// for the layout rationale.
    sorted_pos: Vec<SortedSlot>,
    /// Rebuilt lazily from `sorted_pos` + `bin_start` when `get_bins()` is called.
    /// Not touched on the hot path.
    bins: Vec<Bin>,
    bins_dirty: bool,
    bin_start: Vec<usize>,
    /// Per-bin write cursor reused across scatter. Length = NUM_BIN.
    bin_cursor: Vec<usize>,
    /// Per-particle bin index, computed once in `fill_bins` and reused for
    /// histogram + scatter. Avoids recomputing `(x*inv_bin)*BIN_H + y*inv_bin`
    /// twice per particle (saves 2 mul + 2 cast).
    bin_idx: Vec<u32>,
}

impl Physics {
    pub fn new() -> Physics {
        let pos = (0..MAX_PARTICLES)
            .map(|_| {
                vec3(
                    0.0,
                    0.0,
                    quad_rand::gen_range(MAX_RADIUS * 0.5, MAX_RADIUS * 0.6),
                )
            })
            .collect::<Vec<Vec3>>();

        let bins = (0..NUM_BIN).map(|_| Bin::new()).collect();

        rayon::ThreadPoolBuilder::new()
            .num_threads(NB_THREAD)
            .build_global()
            .unwrap();

        Physics {
            len: 0,
            pos,
            last_pos: vec![Vec2::ZERO; MAX_PARTICLES],
            sorted_pos: vec![(Vec3::ZERO, 0u32); MAX_PARTICLES],
            bins,
            bins_dirty: true,
            bin_start: vec![0; NUM_BIN + 1],
            bin_cursor: vec![0; NUM_BIN],
            bin_idx: vec![0u32; MAX_PARTICLES],
        }
    }

    /// Fused update_pos + apply_constraint. Both stages stream over `pos`/`last_pos`
    /// and only touch `pos[i]`, so doing them in one pass keeps each particle's data
    /// hot in L1 across both phases (cuts memory traffic on this stage by ~half).
    fn update_and_constrain(&mut self, dt: f32) {
        let factor: f32 = 0.75;
        let acc = vec2(10.0, 200.0) * (dt * dt);
        let xmax_base = WIDTH as f32 - BORDER_PADDING;
        let ymax_base = HEIGHT as f32 - BORDER_PADDING;
        let xmin_base = BORDER_PADDING;
        let ymin_base = BORDER_PADDING;

        for i in 0..self.len {
            let mut p = self.pos[i];
            let r = p.z;

            let prev = self.last_pos[i];
            let cur_xy = p.xy();
            let diff = cur_xy - prev;
            self.last_pos[i] = cur_xy;
            let mut np = cur_xy + diff + acc;

            let v = np - OBSTACLE_POS;
            let dist2 = v.length_squared();
            let obs_min = r + OBSTACLE_PADDING;
            if dist2 < obs_min * obs_min {
                let dist = dist2.sqrt();
                let n = v / dist;
                np -= n * 0.1 * (dist - obs_min);
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

            p.x = np.x;
            p.y = np.y;
            self.pos[i] = p;
        }
    }

    fn fill_bins(&mut self) {
        // Counting sort: one histogram pass, prefix sum, one scatter pass.
        // Skip the Vec<Bin> rebuild — `check_collisions` only reads `bin_start`,
        // and `bins` is reconstructed lazily in `get_bins()` when the UI needs it.
        self.bins_dirty = true;

        let inv_bin = 1.0 / BIN_SIZE as f32;

        // 1a) Compute bin index per particle in a SIMD-friendly streaming pass.
        //     Splitting the bin-idx loop from the histogram lets the auto-vectoriser
        //     run on the float→bin maths (fmul + fcvtzs), since the random scatter
        //     into `bin_start` blocks vectorisation otherwise. Casting through u32
        //     (instead of usize/u64) keeps the SIMD width at 4 lanes — fcvtzs.4s
        //     instead of fcvtzu.2d, doubling the throughput on this stage.
        let pos_slice = &self.pos[..self.len];
        let bin_idx = &mut self.bin_idx[..self.len];
        let bin_h_u32 = BIN_H as u32;
        for (i, p) in pos_slice.iter().enumerate() {
            let bx = (p.x * inv_bin) as u32;
            let by = (p.y * inv_bin) as u32;
            bin_idx[i] = by + bx * bin_h_u32;
        }

        // 1b) Histogram into bin_start[0..NUM_BIN].
        for c in &mut self.bin_start[..NUM_BIN] {
            *c = 0;
        }
        for &b in bin_idx.iter() {
            self.bin_start[b as usize] += 1;
        }

        // 2) Exclusive prefix sum in-place; sentinel at [NUM_BIN] = total.
        let mut acc = 0usize;
        for c in &mut self.bin_start[..NUM_BIN] {
            let v = *c;
            *c = acc;
            acc += v;
        }
        self.bin_start[NUM_BIN] = acc;
        debug_assert_eq!(acc, self.len);

        // 3) Scatter using cached bin indices.
        self.bin_cursor.copy_from_slice(&self.bin_start[..NUM_BIN]);
        for i in 0..self.len {
            let b = self.bin_idx[i] as usize;
            let slot = self.bin_cursor[b];
            self.bin_cursor[b] = slot + 1;
            self.sorted_pos[slot] = (self.pos[i], i as u32);
        }
    }

    /// Rebuild `self.bins` from the compact (sorted_pos, bin_start) layout.
    /// Called on demand only (UI key events).
    fn rebuild_bins(&mut self) {
        for (bi, bin) in self.bins.iter_mut().enumerate() {
            bin.indexes.clear();
            let a = self.bin_start[bi];
            let b = self.bin_start[bi + 1];
            bin.indexes.extend(self.sorted_pos[a..b].iter().map(|sp| sp.1 as usize));
        }
        self.bins_dirty = false;
    }

    fn check_collisions(&mut self) {
        let chunk_size = NUM_BIN / NB_THREAD;
        let thread_width = BIN_W / NB_THREAD;

        // Avoid the per-substep Vec alloc — just probe NB_THREAD points in bin_start.
        let mut breakpoints_thread = [0usize; NB_THREAD];
        for (k, slot) in breakpoints_thread.iter_mut().enumerate() {
            *slot = unsafe { *self.bin_start.get_unchecked(k * chunk_size) };
        }

        let check_slice = |slice_pos: &mut [SortedSlot],
                           offset: usize,
                           start_x: usize,
                           width: usize,
                           wall: usize| {
            if slice_pos.is_empty() {
                return;
            }
            let bin_start = self.bin_start.as_slice();
            for bin_x in (start_x + wall)..((start_x + width) - wall) {
                let col_base = bin_x * BIN_H;
                for bin_y in 1..BIN_H - 1 {
                    let bin1 = col_base + bin_y;
                    // SAFETY: bin1+1 <= NUM_BIN (bin1 < NUM_BIN-1 because bin_y < BIN_H-1).
                    let i1_lo = unsafe { *bin_start.get_unchecked(bin1) };
                    let i1_hi = unsafe { *bin_start.get_unchecked(bin1 + 1) };
                    if i1_lo == i1_hi {
                        continue;
                    }

                    // Forward neighbours have exactly 4 candidates with bin2 > bin1:
                    //   (0,+1):     bin1+1                     - same column, +1 row
                    //   (+1,-1..1): bin1+BIN_H-1 .. bin1+BIN_H+1 - next column, 3 rows
                    // The 3 right-column bins are contiguous in `bin_start`, so we can
                    // collapse them into a single range using the prefix-sum property:
                    //   range = [bin_start[bin1+BIN_H-1], bin_start[bin1+BIN_H+2])
                    // Two range lookups instead of four — fewer indirect loads, fewer
                    // branches, smaller scratch array.
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
                        let (pos1, _) = slice_pos[i1 - offset];
                        let pos1_z = pos1.z;
                        // Batch the i1 write across all hits this iteration.
                        // i1 isn't read as i2 inside this i1 iteration (same-bin uses
                        // i2 > i1, fwd uses different bins), so deferring the write
                        // is safe. Saves repeated memory RMWs to the same slot.
                        let mut delta1 = Vec2::ZERO;

                        // Same-bin: start at i1+1 so i2 > i1 is guaranteed, no predicate.
                        for i2 in (i1 + 1)..i1_hi {
                            let (pos2, _) = unsafe { *slice_pos.get_unchecked(i2 - offset) };
                            let v = pos1.xy() - pos2.xy();
                            let dist2 = v.length_squared();
                            let min_dist = pos1_z + pos2.z;
                            if dist2 < min_dist * min_dist {
                                // Folded form: scale_k = delta * z_other / (dist * min_dist).
                                // Precomputing `delta / (dist * min_dist)` once turns the
                                // 1×sqrt + 3×fdiv body into 1×sqrt + 1×fdiv (4 fewer
                                // serial cycles on Apple Silicon's FP unit per hit).
                                let dist = dist2.sqrt();
                                let dn = (0.5 * 0.75) * (dist - min_dist) / (dist * min_dist);
                                let v_scaled = v * dn;
                                delta1 -= v_scaled * pos2.z;
                                let i2_delta = v_scaled * pos1_z;
                                let slot = unsafe { slice_pos.get_unchecked_mut(i2 - offset) };
                                slot.0.x += i2_delta.x;
                                slot.0.y += i2_delta.y;
                            }
                        }

                        // Forward neighbour bins (no i1>=i2 check needed).
                        for &(lo, hi) in &fwd_ranges[..fwd_n] {
                            for i2 in lo..hi {
                                let (pos2, _) = unsafe { *slice_pos.get_unchecked(i2 - offset) };
                                let v = pos1.xy() - pos2.xy();
                                let dist2 = v.length_squared();
                                let min_dist = pos1_z + pos2.z;
                                if dist2 < min_dist * min_dist {
                                    let dist = dist2.sqrt();
                                    let dn = (0.5 * 0.75) * (dist - min_dist) / (dist * min_dist);
                                    let v_scaled = v * dn;
                                    delta1 -= v_scaled * pos2.z;
                                    let i2_delta = v_scaled * pos1_z;
                                    let slot = unsafe { slice_pos.get_unchecked_mut(i2 - offset) };
                                    slot.0.x += i2_delta.x;
                                    slot.0.y += i2_delta.y;
                                }
                            }
                        }

                        if delta1 != Vec2::ZERO {
                            let slot = unsafe { slice_pos.get_unchecked_mut(i1 - offset) };
                            slot.0.x += delta1.x;
                            slot.0.y += delta1.y;
                        }
                    }
                }
            }
        };

        // Main pass: split sorted_pos into NB_THREAD contiguous chunks using
        // breakpoints_thread, one per column-strip. `split_at_mut` chain produces
        // NB_THREAD disjoint &mut sub-slices — same pattern as the border pass above,
        // replacing the old `par_bridge` (which serialises on an internal mutex).
        let mut main_subs: Vec<(&mut [SortedSlot], usize, usize)> =
            Vec::with_capacity(NB_THREAD);
        let mut rest: &mut [SortedSlot] = &mut self.sorted_pos[..];
        for slice_i in 0..NB_THREAD {
            let a = breakpoints_thread[slice_i];
            let b = if slice_i + 1 < NB_THREAD {
                breakpoints_thread[slice_i + 1]
            } else {
                self.len
            };
            let skip = a - main_subs.last().map(|s| s.1 + s.0.len()).unwrap_or(0);
            let (_, after) = rest.split_at_mut(skip);
            let (sub, tail) = after.split_at_mut(b - a);
            main_subs.push((sub, a, slice_i * thread_width));
            rest = tail;
        }
        main_subs
            .into_par_iter()
            .for_each(|(slice_pos, offset, start_x)| {
                check_slice(slice_pos, offset, start_x, thread_width, 1);
            });

        // Strips for bin1 in columns (start_x, start_x+1) need 3x3 in column range
        // (start_x-1)..=(start_x+2) => bins [ (start_x-1)*BIN_H, (start_x+3)*BIN_H ).
        // The old `skip(chunk_size/2) + ChunksMutIndices` misaligned: offset started at
        // bin 1200 while neighbors could be in 1920+ — `i2 - offset` underflowed and
        // `get_unchecked` read garbage (spurious bounces at inter-thread x boundaries).
        //
        // Border strips are column-disjoint: strip i covers x ∈ [(i+1)*tw-1, (i+1)*tw+3).
        // Neighbor strips are at (i+2)*tw-1, separated by tw columns (10 > 4), so their
        // sorted_pos ranges don't overlap → safe to parallelize by chaining split_at_mut.
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
        // Strips are in increasing a-order; carve out non-overlapping &mut sub-slices.
        let mut rest: &mut [SortedSlot] = &mut self.sorted_pos[..];
        let mut rest_off: usize = 0;
        let mut sub_slices: Vec<(&mut [SortedSlot], usize, usize)> =
            Vec::with_capacity(strips.len());
        for &(a, b, start_x) in &strips {
            let skip = a - rest_off;
            let (_, after) = rest.split_at_mut(skip);
            let (sub, tail) = after.split_at_mut(b - a);
            sub_slices.push((sub, a, start_x));
            rest = tail;
            rest_off = b;
        }
        sub_slices
            .into_par_iter()
            .for_each(|(slice, offset, start_x)| {
                check_slice(slice, offset, start_x, 2, 0);
            });

        for sp in self.sorted_pos.iter().take(self.len) {
            self.pos[sp.1 as usize] = sp.0;
        }
    }

    pub fn step(&mut self, dt: f32) {
        let t = Instant::now();
        self.update_and_constrain(dt);
        // Counted under update_pos so the breakdown still adds up; the merged
        // stage now covers what was previously update_pos + apply_constraint.
        T_UPDATE_POS.fetch_add(t.elapsed().as_micros() as u64, Relaxed);

        let t = Instant::now();
        self.fill_bins();
        T_FILL_BINS.fetch_add(t.elapsed().as_micros() as u64, Relaxed);

        let t = Instant::now();
        self.check_collisions();
        T_CHECK_COLLISIONS.fetch_add(t.elapsed().as_micros() as u64, Relaxed);
    }

    pub fn avoid_obstacle(&mut self, pos: Vec2, size: f32) {
        for i in 0..self.len {
            let v = self.pos[i].xy() - pos;
            let dist2 = v.length_squared();
            let min_dist = self.pos[i].z + size;
            if dist2 < min_dist * min_dist {
                let dist = dist2.sqrt();
                let n = v / dist;
                self.pos[i] -= (n * 0.1 * (dist - min_dist)).extend(0.0);
            }
        }
    }

    fn add_object(&mut self, pos: Vec2, vel: Vec2) {
        self.pos[self.len] = pos.extend(self.pos[self.len].z);
        self.last_pos[self.len] = pos - vel;
        self.len += 1;
    }

    pub fn emit_flow(&mut self) {
        let dir = vec2(2.0, 1.0).normalize();
        let space = MAX_RADIUS * 2.0 + 0.01;
        for i in 0..16 {
            let off_y = i as f32 * space;
            for j in 0..3 {
                if self.len >= MAX_PARTICLES {
                    break;
                }
                self.add_object(
                    vec2(200.0, 200.0 + off_y) + dir * space * j as f32,
                    dir * 2.2f32,
                );
            }
        }
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

    pub fn get_points(&self) -> &[Vec3] {
        &self.pos
    }
}
