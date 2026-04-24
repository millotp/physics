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
pub const NB_THREAD: usize = 24;
const BIN_SIZE: usize = 5;
pub const BIN_W: usize = WIDTH / BIN_SIZE;
const BIN_H: usize = HEIGHT / BIN_SIZE;
const NUM_BIN: usize = BIN_W * BIN_H;
const BORDER_PADDING: f32 = 100.0;
const OBSTACLE_POS: Vec2 = vec2(850.0, 600.0);
const OBSTACLE_PADDING: f32 = 100.0;

const _: () = assert!((WIDTH / BIN_SIZE).is_multiple_of(NB_THREAD));
const _: () = assert!(MAX_RADIUS * 2.0 <= BIN_SIZE as f32);

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
    sorted_pos: Vec<(Vec3, usize)>,
    /// Rebuilt lazily from `sorted_pos` + `bin_start` when `get_bins()` is called.
    /// Not touched on the hot path.
    bins: Vec<Bin>,
    bins_dirty: bool,
    bin_start: Vec<usize>,
    /// Per-bin write cursor reused across scatter. Length = NUM_BIN.
    bin_cursor: Vec<usize>,
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
            sorted_pos: vec![(Vec3::ZERO, 0); MAX_PARTICLES],
            bins,
            bins_dirty: true,
            bin_start: vec![0; NUM_BIN + 1],
            bin_cursor: vec![0; NUM_BIN],
        }
    }

    fn apply_constraint(&mut self) {
        let factor = 0.75;
        for i in 0..self.len {
            self.apply_obstacle_constraint(i);

            if self.pos[i].x > WIDTH as f32 - BORDER_PADDING - self.pos[i].z {
                self.pos[i].x += factor
                    * (WIDTH as f32 - BORDER_PADDING - self.pos[i].z - self.pos[i].x);
            }
            if self.pos[i].x < BORDER_PADDING + self.pos[i].z {
                self.pos[i].x += factor * (BORDER_PADDING + self.pos[i].z - self.pos[i].x);
            }
            if self.pos[i].y > HEIGHT as f32 - BORDER_PADDING - self.pos[i].z {
                self.pos[i].y += factor
                    * (HEIGHT as f32 - BORDER_PADDING - self.pos[i].z - self.pos[i].y);
            }
            if self.pos[i].y < BORDER_PADDING + self.pos[i].z {
                self.pos[i].y += factor * (BORDER_PADDING + self.pos[i].z - self.pos[i].y);
            }
        }
    }

    #[inline]
    fn apply_obstacle_constraint(&mut self, i: usize) {
        let v = self.pos[i].xy() - OBSTACLE_POS;
        let dist2 = v.length_squared();
        let min_dist = self.pos[i].z + OBSTACLE_PADDING;
        if dist2 < min_dist * min_dist {
            let dist = dist2.sqrt();
            let n = v / dist;
            self.pos[i] -= (n * 0.1 * (dist - min_dist)).extend(0.0);
        }
    }

    fn update_pos(&mut self, dt: f32) {
        for (i, pos) in self.pos.iter_mut().take(self.len).enumerate() {
            let diff = pos.xy() - self.last_pos[i];
            self.last_pos[i] = pos.xy();
            *pos += (diff + vec2(10.0, 200.0) * (dt * dt)).extend(0.0);
        }
    }

    fn fill_bins(&mut self) {
        // Counting sort: one histogram pass, prefix sum, one scatter pass.
        // Skip the Vec<Bin> rebuild — `check_collisions` only reads `bin_start`,
        // and `bins` is reconstructed lazily in `get_bins()` when the UI needs it.
        self.bins_dirty = true;

        // 1) Histogram into bin_start[0..NUM_BIN] (reusing it as the count buffer).
        for c in &mut self.bin_start[..NUM_BIN] {
            *c = 0;
        }
        let inv_bin = 1.0 / BIN_SIZE as f32;
        for i in 0..self.len {
            let p = self.pos[i];
            let bx = (p.x * inv_bin) as usize;
            let by = (p.y * inv_bin) as usize;
            self.bin_start[by + bx * BIN_H] += 1;
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

        // 3) Scatter. Cursors start at each bin's bin_start.
        self.bin_cursor.copy_from_slice(&self.bin_start[..NUM_BIN]);
        for i in 0..self.len {
            let p = self.pos[i];
            let bx = (p.x * inv_bin) as usize;
            let by = (p.y * inv_bin) as usize;
            let b = by + bx * BIN_H;
            let slot = self.bin_cursor[b];
            self.bin_cursor[b] = slot + 1;
            self.sorted_pos[slot] = (p, i);
        }
    }

    /// Rebuild `self.bins` from the compact (sorted_pos, bin_start) layout.
    /// Called on demand only (UI key events).
    fn rebuild_bins(&mut self) {
        for (bi, bin) in self.bins.iter_mut().enumerate() {
            bin.indexes.clear();
            let a = self.bin_start[bi];
            let b = self.bin_start[bi + 1];
            bin.indexes.extend(self.sorted_pos[a..b].iter().map(|sp| sp.1));
        }
        self.bins_dirty = false;
    }

    fn check_collisions(&mut self) {
        let chunk_size = NUM_BIN / NB_THREAD;
        let thread_width = BIN_W / NB_THREAD;

        let breakpoints_thread = self
            .bin_start
            .iter()
            .take(NUM_BIN)
            .step_by(chunk_size)
            .copied()
            .collect::<Vec<usize>>();

        let check_slice = |slice_pos: &mut [(Vec3, usize)],
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

                    // Collect only neighbours with bin index > bin1 (so all i2 > i1),
                    // plus the same-bin case handled separately below.
                    // `bin_start` is monotonic non-decreasing, so a neighbour bin_n with
                    // bin_n < bin1 has all its i2 < i1 -> nothing to do.
                    let mut fwd_ranges: [(usize, usize); 8] = [(0, 0); 8];
                    let mut fwd_n = 0usize;
                    for (off_i, off_j) in [
                        (-1i32, -1i32), (-1, 0), (-1, 1),
                        (0, -1), /* (0,0) same bin */ (0, 1),
                        (1, -1), (1, 0), (1, 1),
                    ] {
                        let nbx = (bin_x as i32 + off_i) as usize;
                        let nby = (bin_y as i32 + off_j) as usize;
                        let bin2 = nby + nbx * BIN_H;
                        if bin2 <= bin1 {
                            continue;
                        }
                        let (lo, hi) = unsafe {
                            (*bin_start.get_unchecked(bin2),
                             *bin_start.get_unchecked(bin2 + 1))
                        };
                        if lo < hi {
                            fwd_ranges[fwd_n] = (lo, hi);
                            fwd_n += 1;
                        }
                    }

                    for i1 in i1_lo..i1_hi {
                        let (pos1, _) = slice_pos[i1 - offset];
                        let pos1_z = pos1.z;

                        // Same-bin: start at i1+1 so i2 > i1 is guaranteed, no predicate.
                        for i2 in (i1 + 1)..i1_hi {
                            let (pos2, _) = unsafe { *slice_pos.get_unchecked(i2 - offset) };
                            let v = pos1.xy() - pos2.xy();
                            let dist2 = v.length_squared();
                            let min_dist = pos1_z + pos2.z;
                            if dist2 < min_dist * min_dist {
                                let dist = dist2.sqrt();
                                let inv_min = 1.0 / min_dist;
                                let delta = 0.5 * 0.75 * (dist - min_dist);
                                let scale1 = delta * pos2.z * inv_min / dist;
                                let scale2 = delta * pos1_z * inv_min / dist;
                                slice_pos[i1 - offset].0 -= (v * scale1).extend(0.0);
                                slice_pos[i2 - offset].0 += (v * scale2).extend(0.0);
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
                                    let inv_min = 1.0 / min_dist;
                                    let delta = 0.5 * 0.75 * (dist - min_dist);
                                    let scale1 = delta * pos2.z * inv_min / dist;
                                    let scale2 = delta * pos1_z * inv_min / dist;
                                    slice_pos[i1 - offset].0 -= (v * scale1).extend(0.0);
                                    slice_pos[i2 - offset].0 += (v * scale2).extend(0.0);
                                }
                            }
                        }
                    }
                }
            }
        };

        // Main pass: split sorted_pos into NB_THREAD contiguous chunks using
        // breakpoints_thread, one per column-strip. `split_at_mut` chain produces
        // NB_THREAD disjoint &mut sub-slices — same pattern as the border pass above,
        // replacing the old `par_bridge` (which serialises on an internal mutex).
        let mut main_subs: Vec<(&mut [(Vec3, usize)], usize, usize)> =
            Vec::with_capacity(NB_THREAD);
        let mut rest: &mut [(Vec3, usize)] = &mut self.sorted_pos[..];
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
        let mut rest: &mut [(Vec3, usize)] = &mut self.sorted_pos[..];
        let mut rest_off: usize = 0;
        let mut sub_slices: Vec<(&mut [(Vec3, usize)], usize, usize)> =
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
            self.pos[sp.1] = sp.0;
        }
    }

    pub fn step(&mut self, dt: f32) {
        let t = Instant::now();
        self.update_pos(dt);
        T_UPDATE_POS.fetch_add(t.elapsed().as_micros() as u64, Relaxed);

        let t = Instant::now();
        self.apply_constraint();
        T_APPLY_CONSTRAINT.fetch_add(t.elapsed().as_micros() as u64, Relaxed);

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
