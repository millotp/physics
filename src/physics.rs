use glam::{vec2, Vec2};
use rayon::prelude::*;

use crate::{chunk_iter::ChunksMutIndices, HEIGHT, WIDTH};

pub const MAX_PARTICLES: usize = 70000;
pub const MAX_RADIUS: f32 = 2.0;
pub const NB_THREAD: usize = 24;
const BIN_SIZE: usize = 5;
pub const BIN_W: usize = WIDTH / BIN_SIZE;
const BIN_H: usize = HEIGHT / BIN_SIZE;
const NUM_BIN: usize = BIN_W * BIN_H;

const _: () = assert!((WIDTH / BIN_SIZE) % NB_THREAD == 0);
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
    pos: Vec<Vec2>,
    last_pos: Vec<Vec2>,
    sorted_pos: Vec<(Vec2, usize)>,
    bins: Vec<Bin>,
    bin_start: Vec<usize>,
}

impl Physics {
    pub fn new() -> Physics {
        let pos = (0..MAX_PARTICLES)
            .map(|_| vec2(0.0, 0.0))
            .collect::<Vec<Vec2>>();

        let bins = (0..NUM_BIN).map(|_| Bin::new()).collect();

        rayon::ThreadPoolBuilder::new()
            .num_threads(NB_THREAD)
            .build_global()
            .unwrap();

        Physics {
            len: 0,
            pos,
            last_pos: vec![Vec2::ZERO; MAX_PARTICLES],
            sorted_pos: vec![(Vec2::ZERO, 0); MAX_PARTICLES],
            bins,
            bin_start: vec![0; NUM_BIN + 1],
        }
    }

    fn apply_constraint(&mut self) {
        let factor = 0.75;
        //let center = vec2(WIDTH as f32 / 2.0, HEIGHT as f32 / 2.00);
        for i in 0..self.len {
            /*let diff = center - self.pos[i];
            let len = diff.length();
            if len > 400.0 - self.radii[i] {
                let n = diff / len;
                self.pos[i] = center - n * (400.0 - self.radii[i]);
            }*/

            let v = self.pos[i] - vec2(850.0, 600.0);
            let dist2 = v.length_squared();
            let min_dist = MAX_RADIUS + 100.0;
            if dist2 < min_dist * min_dist {
                let dist = dist2.sqrt();
                let n = v / dist;
                self.pos[i] -= n * 0.1 * (dist - min_dist);
            }

            if self.pos[i].x > WIDTH as f32 - 100.0 - MAX_RADIUS {
                self.pos[i].x += factor * (WIDTH as f32 - 100.0 - MAX_RADIUS - self.pos[i].x);
            }
            if self.pos[i].x < 100.0 + MAX_RADIUS {
                self.pos[i].x += factor * (100.0 + MAX_RADIUS - self.pos[i].x);
            }
            if self.pos[i].y > HEIGHT as f32 - 100.0 - MAX_RADIUS {
                self.pos[i].y += factor * (HEIGHT as f32 - 100.0 - MAX_RADIUS - self.pos[i].y);
            }
            if self.pos[i].y < 100.0 + MAX_RADIUS {
                self.pos[i].y += factor * (100.0 + MAX_RADIUS - self.pos[i].y);
            }
        }
    }

    fn update_pos(&mut self, dt: f32) {
        for (i, pos) in self.pos.iter_mut().take(self.len).enumerate() {
            let diff = *pos - self.last_pos[i];
            self.last_pos[i] = *pos;
            *pos += diff + vec2(10.0, 200.0) * (dt * dt);
        }
    }

    fn fill_bins(&mut self) {
        self.bins.iter_mut().for_each(|b| b.indexes.clear());

        for i in 0..self.len {
            let pos = self.pos[i] / BIN_SIZE as f32;
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

        let check_slice = |slice_pos: &mut [(Vec2, usize)],
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
                    || bin_y < 1
                    || bin_y >= BIN_H - 1
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

                                let (pos2, _) = unsafe { *slice_pos.get_unchecked(i2 - offset) };
                                let v = pos1 - pos2;
                                let dist2 = v.length_squared();
                                let min_dist = 2.0 * MAX_RADIUS;
                                if dist2 < min_dist * min_dist {
                                    let dist = dist2.sqrt();
                                    let n = v / dist * (0.5 * 0.5 * 0.75) * (dist - min_dist);
                                    slice_pos[i1 - offset].0 -= n;
                                    slice_pos[i2 - offset].0 += n;
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

        let breakpoints_borders = self
            .bin_start
            .iter()
            .skip(chunk_size / 2)
            .take(NUM_BIN - chunk_size)
            .step_by(chunk_size)
            .copied()
            .collect::<Vec<usize>>();

        // check collisions across thread borders
        ChunksMutIndices::new(&mut self.sorted_pos, &breakpoints_borders)
            .enumerate()
            .par_bridge()
            .for_each(|(slice_i, (slice_pos, breakpoint))| {
                check_slice(
                    slice_pos,
                    breakpoint,
                    (slice_i + 1) * thread_width - 1,
                    2,
                    0,
                )
            });

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
            let v = self.pos[i] - pos;
            let dist2 = v.length_squared();
            let min_dist = MAX_RADIUS + size;
            if dist2 < min_dist * min_dist {
                let dist = dist2.sqrt();
                let n = v / dist;
                self.pos[i] -= n * 0.1 * (dist - min_dist);
            }
        }
    }

    fn add_object(&mut self, pos: Vec2, vel: Vec2) {
        self.pos[self.len] = pos;
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

    pub fn get_points(&self) -> &[Vec2] {
        &self.pos
    }
}
