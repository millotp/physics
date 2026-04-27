use glam::{vec2, Vec2, Vec3, Vec3Swizzles};
use rayon::prelude::*;

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

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

pub const MAX_PARTICLES: usize = 70000;
const MAX_RADIUS: f32 = 2.5;
pub const NB_THREAD: usize = 8;
const BIN_SIZE: usize = 5;
pub const BIN_W: usize = WIDTH / BIN_SIZE;
const BIN_H: usize = HEIGHT / BIN_SIZE;
const NUM_BIN: usize = BIN_W * BIN_H;
/// Inner-border thickness in world units. The wall constraint and pair
/// collision response need a non-zero buffer so particles can resolve
/// overlaps in 2D near walls instead of collapsing into a 1D line. The
/// renderer hides this buffer by zooming the camera to the playable area.
pub const BORDER_PADDING: f32 = 15.0;

/// Active scenes. Pick a combination by changing these two consts.
pub const EMIT_SCENE: EmitScene = EmitScene::TwoCrossing;
pub const OBSTACLE_SCENE: ObstacleScene = ObstacleScene::Pachinko;

/// How particles are spawned each `emit_flow` call.
#[allow(dead_code, reason = "variants are scene options selected via EMIT_SCENE")]
#[derive(Copy, Clone, Debug)]
pub enum EmitScene {
    /// Original: one diagonal stream from `(200, 200)`.
    Single,
    /// Two streams crossing in mid-air, one from each top corner.
    TwoCrossing,
    /// Single emitter, fan of velocities (cone of directions).
    Fountain,
    /// Big bursts every ~1 s instead of a constant trickle.
    Pulse,
    /// A long horizontal strip across the top, particles fall straight down.
    Rain,
    /// Curated mix: rain + a sweeping fountain, picked for visual impact.
    YouDecide,
}

/// What static / animated obstacles populate the playfield.
#[allow(
    dead_code,
    reason = "variants are scene options selected via OBSTACLE_SCENE"
)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ObstacleScene {
    /// Original: one big circular bumper at `(850, 600)`.
    Single,
    /// Pachinko: 5 rows of pegs in a triangle pattern.
    Pachinko,
    /// A long oblong (capsule) sweeping around the centre.
    RotatingBar,
    /// Ring of pegs around the centre.
    Ring,
    /// Three big circles at distinctive positions.
    FewCircles,
    /// Curated mix: a small pachinko on top of two rotating bars below.
    YouDecide,
}

/// Static and animated colliders. Capsule is a swept circle of radius `r`
/// between endpoints `a` and `b` — closest-point-on-segment is cheap and
/// covers oblong bars.
#[derive(Copy, Clone, Debug)]
enum Obstacle {
    Circle { center: Vec2, radius: f32 },
    Capsule { a: Vec2, b: Vec2, radius: f32 },
}

/// Per-step angular velocity for the rotating bar variants (rad / sim-second).
/// Stays deterministic since it's tied to the physics step count, not
/// wall-clock time.
const BAR_ANGULAR_VEL: f32 = 0.6;

/// Default emission rate, in flow-batches per second (one batch = 48 particles,
/// see `emit_flow`). 60/s matches the historical "one emit per 1/60 s frame"
/// tuning the rest of the params are calibrated for.
const EMIT_BATCHES_PER_SEC: f32 = 60.0;

/// Gravity acceleration in world units / s². Small X gives the flow a slight
/// horizontal drift; Y dominates.
const GRAVITY: Vec2 = vec2(10.0, 200.0);
/// Pushback factor for wall constraints. <1 leaves a residual overlap for the
/// next substep to clean up — softer visuals, more stable.
const WALL_DAMPING: f32 = 0.75;
/// Per-substep push factor for the static obstacle (and the mouse cursor).
const OBSTACLE_PUSH: f32 = 0.1;
/// Pair-collision response: each particle absorbs half the overlap, scaled by
/// `WALL_DAMPING` for consistency with the wall response.
const COLLISION_RESPONSE: f32 = 0.5 * WALL_DAMPING;

const _: () = assert!((WIDTH / BIN_SIZE).is_multiple_of(NB_THREAD));
const _: () = assert!(MAX_RADIUS * 2.0 <= BIN_SIZE as f32);

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
    /// View of `sorted_pos` as `Vec<Bin>`, rebuilt lazily on demand for the UI.
    bins: Vec<Bin>,
    bins_dirty: bool,
    bin_start: Vec<usize>,
    bin_cursor: Vec<usize>,
    /// Per-particle bin index, computed once in `fill_bins` and reused for
    /// histogram + scatter (saves recomputing `(x*inv_bin)*BIN_H + y*inv_bin`).
    bin_idx: Vec<u32>,
    // --- Cold fields below: only touched on emit, never on the stepping hot path. ---
    /// Pre-generated radii for not-yet-emitted slots; `add_object` consumes
    /// `pending_radii[len]` and copies it into `pos[len].z`. Pre-generating
    /// keeps RNG ordering deterministic across runs (seed=N reproduces).
    pending_radii: Vec<f32>,
    /// Fractional batch accumulator for `emit_flow_for`.
    emit_acc: f32,
    /// Active obstacle list, rebuilt from `OBSTACLE_SCENE` at construction.
    /// Animated scenes (rotating bar) overwrite entries each step.
    obstacles: Vec<Obstacle>,
    /// Sim-time elapsed in seconds, accumulated from `step(dt)`. Drives all
    /// time-dependent obstacle/emitter scenes deterministically.
    sim_time: f32,
    /// Number of `emit_flow` calls so far. Drives the pulse/burst emitters.
    emit_count: u32,
}

impl Physics {
    pub fn new() -> Physics {
        // Pre-draw every radius up front so RNG ordering is deterministic
        // regardless of when each slot is later emitted by `add_object`.
        let pending_radii = (0..MAX_PARTICLES)
            .map(|_| quad_rand::gen_range(MAX_RADIUS * 0.5, MAX_RADIUS * 0.6))
            .collect();

        let bins = (0..NUM_BIN).map(|_| Bin::new()).collect();

        rayon::ThreadPoolBuilder::new()
            .num_threads(NB_THREAD)
            .build_global()
            .unwrap();

        let obstacles = build_obstacles(OBSTACLE_SCENE, 0.0);

        Physics {
            len: 0,
            pos: vec![Vec3::ZERO; MAX_PARTICLES],
            last_pos: vec![Vec2::ZERO; MAX_PARTICLES],
            sorted_pos: vec![(Vec3::ZERO, 0u32); MAX_PARTICLES],
            bins,
            bins_dirty: true,
            bin_start: vec![0; NUM_BIN + 1],
            bin_cursor: vec![0; NUM_BIN],
            bin_idx: vec![0u32; MAX_PARTICLES],
            pending_radii,
            emit_acc: 0.0,
            obstacles,
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
        // bars rebuild their endpoints against the current sim_time).
        if scene_is_animated(OBSTACLE_SCENE) {
            self.obstacles = build_obstacles(OBSTACLE_SCENE, self.sim_time);
        }
        let obstacles = &self.obstacles;

        for i in 0..self.len {
            let mut p = self.pos[i];
            let r = p.z;

            let prev = self.last_pos[i];
            let cur_xy = p.xy();
            let diff = cur_xy - prev;
            self.last_pos[i] = cur_xy;
            let mut np = cur_xy + diff + acc;

            for obs in obstacles.iter() {
                np = resolve_obstacle(np, r, obs);
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
        for i in 0..self.len {
            let b = self.bin_idx[i] as usize;
            let slot = self.bin_cursor[b];
            self.bin_cursor[b] = slot + 1;
            self.sorted_pos[slot] = (self.pos[i], i as u32);
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

        // SAFETY (for the `get_unchecked` reads/writes below):
        // - `bin_start` has length `NUM_BIN + 1`, and every `bin_start[k]`
        //   access uses `k < NUM_BIN + 1`. The maximum we touch is
        //   `bin_start[col_b + 2]` where `col_b = bin1 + BIN_H` and
        //   `bin1 < NUM_BIN - 1`, so `col_b + 2 <= NUM_BIN + 1`.
        // - `slice_pos` covers exactly the contiguous bins assigned to this
        //   strip, with `offset = bin_start[first bin in strip]`. Every i
        //   we index into it (`i1`, `i2`) comes from `bin_start[bin]` for a
        //   bin in this strip's range, so `offset <= i < offset + slice_pos.len()`.
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
                        let (pos1, _) = unsafe { *slice_pos.get_unchecked(i1 - offset) };
                        let pos1_z = pos1.z;
                        // Accumulate locally and write i1 back once: i1 is never read
                        // as i2 within this iteration (same-bin uses i2 > i1, fwd is
                        // a different bin), so we can defer the RMW.
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
                                let dn = COLLISION_RESPONSE * (dist - min_dist) / (dist * min_dist);
                                let v_scaled = v * dn;
                                delta1 -= v_scaled * pos2.z;
                                let i2_delta = v_scaled * pos1_z;
                                let slot = unsafe { slice_pos.get_unchecked_mut(i2 - offset) };
                                slot.0.x += i2_delta.x;
                                slot.0.y += i2_delta.y;
                            }
                        }

                        // Forward bins: bin2 > bin1 by construction, no predicate needed.
                        for &(lo, hi) in &fwd_ranges[..fwd_n] {
                            for i2 in lo..hi {
                                let (pos2, _) = unsafe { *slice_pos.get_unchecked(i2 - offset) };
                                let v = pos1.xy() - pos2.xy();
                                let dist2 = v.length_squared();
                                let min_dist = pos1_z + pos2.z;
                                if dist2 < min_dist * min_dist {
                                    let dist = dist2.sqrt();
                                    let dn =
                                        COLLISION_RESPONSE * (dist - min_dist) / (dist * min_dist);
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

        // Carve `sorted_pos` into NB_THREAD disjoint &mut sub-slices, one per
        // column-strip. Chained `split_at_mut` lets Rayon process them in
        // parallel without a Mutex.
        let mut main_subs: Vec<(&mut [SortedSlot], usize, usize)> = Vec::with_capacity(NB_THREAD);
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

        // Border pass: each strip processes the 2-column seam (start_x, start_x+1)
        // between two main-pass slices, needing 3×3 neighbours so the bin range is
        // [(start_x-1)*BIN_H, (start_x+3)*BIN_H). Adjacent border strips are spaced
        // `thread_width` columns apart (10 > 4 here), so they never overlap and we
        // can carve disjoint &mut sub-slices for parallel processing.
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

    pub fn avoid_obstacle(&mut self, pos: Vec2, size: f32) {
        for i in 0..self.len {
            let p = self.pos[i];
            let v = p.xy() - pos;
            let dist2 = v.length_squared();
            let min_dist = p.z + size;
            if dist2 < min_dist * min_dist {
                let dist = dist2.sqrt();
                let n = v / dist;
                self.pos[i] -= (n * OBSTACLE_PUSH * (dist - min_dist)).extend(0.0);
            }
        }
    }

    fn add_object(&mut self, pos: Vec2, vel: Vec2) {
        let radius = self.pending_radii[self.len];
        self.pos[self.len] = pos.extend(radius);
        self.last_pos[self.len] = pos - vel;
        self.len += 1;
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

    /// Emit one flow batch (~48 particles for the basic emitters). This is
    /// the tuned unit of emission; the bench drives it directly, the GUI
    /// goes through `emit_flow_for` for fps-independence. The active scene
    /// (`EMIT_SCENE`) decides what shape the batch takes.
    pub fn emit_flow(&mut self) {
        let count = self.emit_count;
        self.emit_count = self.emit_count.wrapping_add(1);
        match EMIT_SCENE {
            EmitScene::Single => self.emit_single(),
            EmitScene::TwoCrossing => self.emit_two_crossing(),
            EmitScene::Fountain => self.emit_fountain(count),
            EmitScene::Pulse => self.emit_pulse(count),
            EmitScene::Rain => self.emit_rain(count),
            EmitScene::YouDecide => {
                // Rain on top, plus a sweeping fountain mixing things up.
                self.emit_rain(count);
                self.emit_fountain(count);
            }
        }
    }

    /// 16x3 diagonal stream from (200, 200), the historical default.
    fn emit_single(&mut self) {
        let dir = vec2(2.0, 1.0).normalize();
        let space = MAX_RADIUS * 2.0 + 0.01;
        for i in 0..16 {
            let off_y = i as f32 * space;
            for j in 0..3 {
                if self.len >= MAX_PARTICLES {
                    return;
                }
                self.add_object(
                    vec2(200.0, 200.0 + off_y) + dir * space * j as f32,
                    dir * 2.2,
                );
            }
        }
    }

    /// Two crossing streams from the upper-left and upper-right corners.
    /// Half the per-stream height of `Single` so the total batch stays ~48.
    fn emit_two_crossing(&mut self) {
        let space = MAX_RADIUS * 2.0 + 0.01;
        let dir_l = vec2(2.0, 1.0).normalize();
        let dir_r = vec2(-2.0, 1.0).normalize();
        for i in 0..8 {
            let off_y = i as f32 * space;
            for j in 0..3 {
                if self.len >= MAX_PARTICLES {
                    return;
                }
                self.add_object(
                    vec2(200.0, 200.0 + off_y) + dir_l * space * j as f32,
                    dir_l * 2.2,
                );
                if self.len >= MAX_PARTICLES {
                    return;
                }
                self.add_object(
                    vec2(WIDTH as f32 - 200.0, 200.0 + off_y) + dir_r * space * j as f32,
                    dir_r * 2.2,
                );
            }
        }
    }

    /// Fan-shaped fountain: 48 particles across a 60-degree cone, all from
    /// a single point. The cone direction sweeps slowly using `count` so the
    /// fountain feels alive without wall-clock time.
    fn emit_fountain(&mut self, count: u32) {
        let origin = vec2(WIDTH as f32 * 0.5, HEIGHT as f32 - 200.0);
        let speed = 6.0;
        // Sweep ±30deg around straight up, with a slow oscillation.
        let sweep = (count as f32 * 0.05).sin() * 0.3;
        let half_cone: f32 = std::f32::consts::FRAC_PI_6;
        let n = 48;
        for k in 0..n {
            if self.len >= MAX_PARTICLES {
                return;
            }
            let t = k as f32 / (n - 1) as f32;
            let ang = -std::f32::consts::FRAC_PI_2 + sweep + (t - 0.5) * 2.0 * half_cone;
            let dir = vec2(ang.cos(), ang.sin());
            self.add_object(origin + dir * MAX_RADIUS, dir * speed);
        }
    }

    /// Pulse: emit a dense ring of particles every 60 calls (~1 s real time
    /// at 60 batches/s). Quiet between bursts.
    fn emit_pulse(&mut self, count: u32) {
        if !count.is_multiple_of(60) {
            return;
        }
        let center = vec2(WIDTH as f32 * 0.5, 200.0);
        let n = 96;
        for k in 0..n {
            if self.len >= MAX_PARTICLES {
                return;
            }
            let ang = (k as f32) * std::f32::consts::TAU / (n as f32);
            let dir = vec2(ang.cos(), ang.sin());
            self.add_object(center + dir * 30.0, dir * 5.0);
        }
    }

    /// Rain: one row of 48 particles spread evenly across the top, falling
    /// straight down. `count` shifts the row by half a spacing each call so
    /// the grid pattern doesn't immediately visually resolve.
    fn emit_rain(&mut self, count: u32) {
        let n = 48;
        let span = WIDTH as f32 - 2.0 * BORDER_PADDING - 40.0;
        let step = span / (n - 1) as f32;
        let offset = if count.is_multiple_of(2) { 0.0 } else { step * 0.5 };
        let y = BORDER_PADDING + 10.0;
        for k in 0..n {
            if self.len >= MAX_PARTICLES {
                return;
            }
            let x = BORDER_PADDING + 20.0 + k as f32 * step + offset;
            self.add_object(vec2(x, y), vec2(0.0, 0.5));
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

    /// Live particle slice, length = `nb_particles()`. Use this for rendering
    /// — the underlying buffer is sized to `MAX_PARTICLES` but only the prefix
    /// up to `len` is meaningful.
    pub fn active_points(&self) -> &[Vec3] {
        &self.pos[..self.len]
    }

    /// Renderable disc list for the active obstacle scene. Each entry is
    /// `(center.xy, radius)` packed in a `Vec3`. Capsules are sampled into a
    /// chain of overlapping discs so the same instanced-circle renderer can
    /// draw them. Animated scenes refresh per `step()`, so callers should
    /// re-read this each frame they want to draw.
    pub fn obstacle_discs(&self) -> Vec<Vec3> {
        let mut out = Vec::new();
        for obs in self.obstacles.iter() {
            match *obs {
                Obstacle::Circle { center, radius } => {
                    out.push(center.extend(radius));
                }
                Obstacle::Capsule { a, b, radius } => {
                    let len = (b - a).length();
                    let n = (len / radius).ceil().max(2.0) as usize;
                    for k in 0..=n {
                        let t = k as f32 / n as f32;
                        let p = a + (b - a) * t;
                        out.push(p.extend(radius));
                    }
                }
            }
        }
        out
    }
}

/// Per-scene obstacle layout. The `time` argument lets animated scenes (e.g.
/// rotating bars) parameterise their geometry deterministically against the
/// caller's `sim_time`. Static scenes ignore `time`.
fn build_obstacles(scene: ObstacleScene, time: f32) -> Vec<Obstacle> {
    let cx = WIDTH as f32 * 0.5;
    let cy = HEIGHT as f32 * 0.5;
    match scene {
        ObstacleScene::Single => {
            vec![Obstacle::Circle {
                center: vec2(850.0, 600.0),
                radius: 100.0,
            }]
        }
        ObstacleScene::Pachinko => {
            // 5 staggered rows, ~30..50 pegs total. Top row is just below the
            // emitters; each row offsets by half the column spacing.
            let mut out = Vec::new();
            let rows = 6;
            let cols_top = 6;
            let row_dy = 110.0;
            let col_dx = 130.0;
            let top_y = 320.0;
            for r in 0..rows {
                let stagger = (r % 2) as f32 * 0.5 * col_dx;
                let cols = cols_top + r;
                let row_w = (cols - 1) as f32 * col_dx;
                let row_x0 = cx - row_w * 0.5 + stagger - col_dx * 0.5 * (r as f32 / rows as f32);
                let y = top_y + r as f32 * row_dy;
                for c in 0..cols {
                    let x = row_x0 + c as f32 * col_dx;
                    out.push(Obstacle::Circle {
                        center: vec2(x, y),
                        radius: 18.0,
                    });
                }
            }
            out
        }
        ObstacleScene::RotatingBar => {
            let half_len = 280.0;
            let ang = time * BAR_ANGULAR_VEL;
            let d = vec2(ang.cos(), ang.sin()) * half_len;
            let center = vec2(cx, cy);
            vec![Obstacle::Capsule {
                a: center - d,
                b: center + d,
                radius: 20.0,
            }]
        }
        ObstacleScene::Ring => {
            // 12 pegs on a circle of radius 220 around the centre.
            let n = 12;
            let radius_ring = 220.0;
            (0..n)
                .map(|k| {
                    let ang = k as f32 * std::f32::consts::TAU / n as f32;
                    Obstacle::Circle {
                        center: vec2(cx, cy) + vec2(ang.cos(), ang.sin()) * radius_ring,
                        radius: 30.0,
                    }
                })
                .collect()
        }
        ObstacleScene::FewCircles => vec![
            Obstacle::Circle {
                center: vec2(cx - 250.0, 500.0),
                radius: 80.0,
            },
            Obstacle::Circle {
                center: vec2(cx + 250.0, 500.0),
                radius: 80.0,
            },
            Obstacle::Circle {
                center: vec2(cx, 850.0),
                radius: 110.0,
            },
        ],
        ObstacleScene::YouDecide => {
            // Small pachinko on top + two counter-rotating bars below.
            let mut out = Vec::new();
            let rows = 3;
            let cols = 7;
            let col_dx = 140.0;
            let row_dy = 110.0;
            let top_y = 320.0;
            for r in 0..rows {
                let stagger = (r % 2) as f32 * 0.5 * col_dx;
                let row_w = (cols - 1) as f32 * col_dx;
                let row_x0 = cx - row_w * 0.5 + stagger;
                let y = top_y + r as f32 * row_dy;
                for c in 0..cols {
                    let x = row_x0 + c as f32 * col_dx;
                    out.push(Obstacle::Circle {
                        center: vec2(x, y),
                        radius: 18.0,
                    });
                }
            }
            let bar_half = 200.0;
            let ang_l = time * BAR_ANGULAR_VEL;
            let ang_r = -time * BAR_ANGULAR_VEL;
            let cl = vec2(cx - 280.0, 850.0);
            let cr = vec2(cx + 280.0, 850.0);
            let dl = vec2(ang_l.cos(), ang_l.sin()) * bar_half;
            let dr = vec2(ang_r.cos(), ang_r.sin()) * bar_half;
            out.push(Obstacle::Capsule {
                a: cl - dl,
                b: cl + dl,
                radius: 18.0,
            });
            out.push(Obstacle::Capsule {
                a: cr - dr,
                b: cr + dr,
                radius: 18.0,
            });
            out
        }
    }
}

/// Whether a scene needs its obstacle list rebuilt every step (rotating
/// bars, swept colliders) or whether the layout is static.
fn scene_is_animated(scene: ObstacleScene) -> bool {
    matches!(scene, ObstacleScene::RotatingBar | ObstacleScene::YouDecide)
}

/// Resolve a single obstacle against a particle at `np` (radius `r`).
/// Returns the corrected position; called once per obstacle per particle.
#[inline(always)]
fn resolve_obstacle(np: Vec2, r: f32, obs: &Obstacle) -> Vec2 {
    match *obs {
        Obstacle::Circle {
            center,
            radius: obs_r,
        } => {
            let v = np - center;
            let dist2 = v.length_squared();
            let min_dist = r + obs_r;
            if dist2 < min_dist * min_dist && dist2 > 0.0 {
                let dist = dist2.sqrt();
                let n = v / dist;
                np - n * OBSTACLE_PUSH * (dist - min_dist)
            } else {
                np
            }
        }
        Obstacle::Capsule {
            a,
            b,
            radius: obs_r,
        } => {
            // Closest point on segment ab to np.
            let ab = b - a;
            let len2 = ab.length_squared();
            let t = if len2 > 0.0 {
                ((np - a).dot(ab) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let closest = a + ab * t;
            let v = np - closest;
            let dist2 = v.length_squared();
            let min_dist = r + obs_r;
            if dist2 < min_dist * min_dist && dist2 > 0.0 {
                let dist = dist2.sqrt();
                let n = v / dist;
                np - n * OBSTACLE_PUSH * (dist - min_dist)
            } else {
                np
            }
        }
    }
}
