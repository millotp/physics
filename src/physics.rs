use glam::{vec2, vec3, Vec2, Vec3};
use rayon::prelude::*;

use crate::{chunk_iter::ChunksMutIndices, BORDER, HEIGHT, WIDTH};

pub const MAX_PARTICLES: usize = 70000;
const MAX_RADIUS: f32 = 1.5;
pub const NB_THREAD: usize = 10;
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

#[derive(Clone, Copy, Default)]
pub struct Ball {
    pub pos: Vec2,
    pub last_pos: Vec2,
    pub radius: f32,
    pub neighboors: u16,
}

pub struct Physics {
    len: usize,
    balls: Vec<Ball>,
    sorted_balls: Vec<(Ball, usize)>,
    bins: Vec<Bin>,
    bin_start: Vec<usize>,
}

impl Physics {
    pub fn new() -> Physics {
        let balls = (0..MAX_PARTICLES)
            .map(|_| Ball {
                pos: vec2(0.0, 0.0),
                last_pos: Vec2::ZERO,
                radius: quad_rand::gen_range(MAX_RADIUS * 0.9, MAX_RADIUS),
                neighboors: 0,
            })
            .collect::<Vec<Ball>>();

        let bins = (0..NUM_BIN).map(|_| Bin::new()).collect();

        //rayon::ThreadPoolBuilder::new()
        //    .num_threads(NB_THREAD)
        //    .build_global()
        //   .unwrap();

        Physics {
            len: 0,
            balls,
            sorted_balls: vec![(Ball::default(), 0); MAX_PARTICLES],
            bins,
            bin_start: vec![0; NUM_BIN + 1],
        }
    }

    fn update_pos(&mut self, gravity: Vec2, dt: f32) {
        self.balls.iter_mut().take(self.len).for_each(|b| {
            let diff = b.pos - b.last_pos;
            b.last_pos = b.pos;
            b.neighboors = 0;
            b.pos += diff + gravity * (dt * dt);
        });
    }

    fn apply_constraint(&mut self) {
        let factor = 0.75;
        self.balls.par_iter_mut().take(self.len).for_each(|b| {
            let v = b.pos - vec2(250.0, 700.0);
            let dist2 = v.length_squared();
            let min_dist = b.radius + 100.0;
            if dist2 < min_dist * min_dist {
                let dist = dist2.sqrt();
                let n = v / dist;
                b.pos -= n * 0.1 * (dist - min_dist);
            }

            if b.pos.x > WIDTH as f32 - BORDER - b.radius {
                b.pos.x += factor * (WIDTH as f32 - BORDER - b.radius - b.pos.x);
            }
            if b.pos.x < BORDER + b.radius {
                b.pos.x += factor * (BORDER + b.radius - b.pos.x);
            }
            if b.pos.y > HEIGHT as f32 - BORDER - b.radius {
                b.pos.y += factor * (HEIGHT as f32 - BORDER - b.radius - b.pos.y);
            }
            if b.pos.y < BORDER + b.radius {
                b.pos.y += factor * (BORDER + b.radius - b.pos.y);
            }
        });
    }

    fn fill_bins(&mut self) {
        self.bins.iter_mut().for_each(|b| b.indexes.clear());

        for i in 0..self.len {
            let pos = self.balls[i].pos / BIN_SIZE as f32;
            self.bins[pos.y as usize + pos.x as usize * BIN_H]
                .indexes
                .push(i);
        }

        let mut current_ind = 0;
        for (bi, bin) in self.bins.iter().enumerate() {
            for (i, &pi) in bin.indexes.iter().enumerate() {
                self.sorted_balls[current_ind + i] = (self.balls[pi], pi);
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

        let check_slice = |slice_balls: &mut [(Ball, usize)],
                           offset: usize,
                           start_x: usize,
                           width: usize,
                           wall: usize| {
            if slice_balls.is_empty() {
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
                    let (ball1, _) = slice_balls[i1 - offset];

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

                                let (ball2, _) = unsafe { slice_balls.get_unchecked(i2 - offset) };
                                let v = ball1.pos - ball2.pos;
                                let dist2 = v.length_squared();
                                let min_dist = ball1.radius + ball2.radius;
                                if dist2 < min_dist * min_dist {
                                    let dist = dist2.sqrt();
                                    let n = v / dist;
                                    let mass_ratio_1 = ball2.radius / (ball1.radius + ball2.radius);
                                    let mass_ratio_2 = ball1.radius / (ball1.radius + ball2.radius);
                                    let delta = 0.5 * 0.75 * (dist - min_dist);
                                    slice_balls[i1 - offset].0.pos -= n * (mass_ratio_1 * delta);
                                    slice_balls[i2 - offset].0.pos += n * (mass_ratio_2 * delta);
                                    slice_balls[i1 - offset].0.neighboors += 1;
                                    slice_balls[i2 - offset].0.neighboors += 1;
                                }
                            }
                        }
                    }
                }
            }
        };

        ChunksMutIndices::new(&mut self.sorted_balls, &breakpoints_thread)
            .enumerate()
            .par_bridge()
            .for_each(|(slice_i, (slice_balls, breakpoint))| {
                check_slice(
                    slice_balls,
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
        ChunksMutIndices::new(&mut self.sorted_balls, &breakpoints_borders)
            .enumerate()
            .par_bridge()
            .for_each(|(slice_i, (slice_balls, breakpoint))| {
                check_slice(
                    slice_balls,
                    breakpoint,
                    (slice_i + 1) * thread_width - 1,
                    2,
                    0,
                )
            });

        for sp in self.sorted_balls.iter().take(self.len) {
            self.balls[sp.1] = sp.0;
        }
    }

    pub fn step(&mut self, gravity: Vec2, dt: f32) {
        self.update_pos(gravity, dt);
        self.apply_constraint();
        self.fill_bins();
        self.check_collisions();
    }

    pub fn avoid_obstacle(&mut self, pos: Vec2, size: f32) {
        for i in 0..self.len {
            let v = self.balls[i].pos - pos;
            let dist2 = v.length_squared();
            let min_dist = self.balls[i].radius + size;
            if dist2 < min_dist * min_dist {
                let dist = dist2.sqrt();
                let n = v / dist;
                self.balls[i].pos -= n * 0.1 * (dist - min_dist);
            }
        }
    }

    fn add_object(&mut self, pos: Vec2, vel: Vec2) {
        self.balls[self.len].pos = pos;
        self.balls[self.len].last_pos = pos - vel;
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

    pub fn get_balls(&self) -> &[Ball] {
        &self.balls
    }

    pub fn get_points(&self) -> Vec<Vec2> {
        self.balls.iter().map(|b| b.pos).collect()
    }

    pub fn get_radius(&self) -> Vec<f32> {
        self.balls.iter().map(|b| b.radius).collect()
    }

    pub fn get_colors(&self) -> Vec<Vec3> {
        self.balls
            .iter()
            .map(|b| {
                vec3(
                    b.neighboors as f32 / 10.0,
                    0.7 - b.neighboors as f32 / 15.0,
                    0.9 - b.neighboors as f32 / 15.0,
                )
            })
            .collect()
    }
}
