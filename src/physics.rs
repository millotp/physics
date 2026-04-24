use glam::{vec2, vec3, Vec2, Vec3, Vec3Swizzles};
use rayon::prelude::*;

use crate::{chunk_iter::ChunksMutIndices, HEIGHT, WIDTH};

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
    bins: Vec<Bin>,
    bin_start: Vec<usize>,
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
            bin_start: vec![0; NUM_BIN + 1],
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
        self.bins.iter_mut().for_each(|b| b.indexes.clear());

        for i in 0..self.len {
            let pos = self.pos[i].xy() / BIN_SIZE as f32;
            self.bins[pos.y as usize + pos.x as usize * BIN_H]
                .indexes
                .push(i);
        }

        let mut current_ind = 0;
        for (bi, bin) in self.bins.iter().enumerate() {
            for (i, &pi) in bin.indexes.iter().enumerate() {
                self.sorted_pos[current_ind + i] = (self.pos[pi], pi);
            }
            self.bin_start[bi] = current_ind;
            current_ind += bin.indexes.len();
        }
        self.bin_start[NUM_BIN] = self.len;
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
            for bin1 in (start_x * BIN_H)..((start_x + width) * BIN_H) {
                let bin_x = bin1 / BIN_H;
                let bin_y = bin1 % BIN_H;
                if bin_x < start_x + wall
                    || bin_x >= (start_x + width) - wall
                    || !(1..BIN_H - 1).contains(&bin_y)
                {
                    continue;
                }

                for i1 in self.bin_start[bin1]..self.bin_start[bin1 + 1] {
                    let (pos1, _) = slice_pos[i1 - offset];

                    for off_i in -1..=1 {
                        for off_j in -1..=1 {
                            let bin2 = (bin_y as i32 + off_j) as usize
                                + (bin_x as i32 + off_i) as usize * BIN_H;

                            for i2 in unsafe {
                                *self.bin_start.get_unchecked(bin2)
                                    ..*self.bin_start.get_unchecked(bin2 + 1)
                            } {
                                if i1 >= i2 {
                                    continue;
                                }

                                let (pos2, _) = unsafe { slice_pos.get_unchecked(i2 - offset) };
                                let v = pos1.xy() - pos2.xy();
                                let dist2 = v.length_squared();
                                let min_dist = pos1.z + pos2.z;
                                if dist2 < min_dist * min_dist {
                                    let dist = dist2.sqrt();
                                    let n = v / dist;
                                    let mass_ratio_1 = pos2.z / (pos1.z + pos2.z);
                                    let mass_ratio_2 = pos1.z / (pos1.z + pos2.z);
                                    let delta = 0.5 * 0.75 * (dist - min_dist);
                                    slice_pos[i1 - offset].0 -=
                                        (n * (mass_ratio_1 * delta)).extend(0.0);
                                    slice_pos[i2 - offset].0 +=
                                        (n * (mass_ratio_2 * delta)).extend(0.0);
                                }
                            }
                        }
                    }
                }
            }
        };

        ChunksMutIndices::new(&mut self.sorted_pos, &breakpoints_thread)
            .enumerate()
            .par_bridge()
            .for_each(|(slice_i, (slice_pos, breakpoint))| {
                check_slice(
                    slice_pos,
                    breakpoint,
                    slice_i * thread_width,
                    thread_width,
                    1,
                )
            });

        // Strips for bin1 in columns (start_x, start_x+1) need 3x3 in column range
        // (start_x-1)..=(start_x+2) => bins [ (start_x-1)*BIN_H, (start_x+3)*BIN_H ).
        // The old `skip(chunk_size/2) + ChunksMutIndices` misaligned: offset started at
        // bin 1200 while neighbors could be in 1920+ — `i2 - offset` underflowed and
        // `get_unchecked` read garbage (spurious bounces at inter-thread x boundaries).
        for slice_i in 0..(NB_THREAD - 1) {
            let start_x = (slice_i + 1) * thread_width - 1;
            let col_lo = start_x.saturating_sub(1);
            let bin_lo = col_lo * BIN_H;
            let bin_hi = (start_x + 3) * BIN_H;
            if bin_hi > NUM_BIN {
                continue;
            }
            let a = self.bin_start[bin_lo];
            let b = self.bin_start[bin_hi];
            if a < b {
                let slice = &mut self.sorted_pos[a..b];
                let offset = a;
                check_slice(slice, offset, start_x, 2, 0);
            }
        }

        for sp in self.sorted_pos.iter().take(self.len) {
            self.pos[sp.1] = sp.0;
        }
    }

    pub fn step(&mut self, dt: f32) {
        self.update_pos(dt);
        self.apply_constraint();
        self.fill_bins();
        self.check_collisions();
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

    pub fn get_bins(&self) -> &[Bin] {
        &self.bins
    }

    pub fn get_points(&self) -> &[Vec3] {
        &self.pos
    }
}
