#![feature(slice_ptr_len)]
#![feature(raw_slice_split)]

use std::{
    f32::consts::PI,
    fs::File,
    io::{Read, Write},
    marker::PhantomData,
    time::Instant,
};

use miniquad::*;

use image::io::Reader as ImageReader;

use glam::{vec2, vec3, Mat4, Vec2, Vec3, Vec3Swizzles};
use scoped_threadpool::Pool;

const MAX_PARTICLES: usize = 50000;
const CIRCLE_SIDES: usize = 12;
const WIDTH: i32 = 1200;
const HEIGHT: i32 = 1200;
const BIN_SIZE: f32 = 10.0;
const NB_THREAD: usize = 20;

struct Bin {
    indexes: Vec<usize>,
}

impl Bin {
    fn new() -> Self {
        Bin {
            indexes: Vec::with_capacity(50),
        }
    }
}

enum UpdateCommand {
    OneFrame,
    Continue,
    Stop,
}

struct Stage {
    pipeline: Pipeline,
    bindings: Bindings,

    len: usize,
    pos: Vec<Vec3>,
    last_pos: Vec<Vec2>,
    sorted_pos: Vec<(Vec3, usize)>,
    bin_start: Vec<usize>,
    num_bin: usize,
    colors: Vec<Vec3>,
    bins: Vec<Bin>,
    bin_w: usize,
    bin_h: usize,
    last_frame: Instant,
    frame_count: usize,
    mouse_pressed: bool,
    mouse_pos: Vec2,
    can_update: UpdateCommand,
    pool: Pool,
}

impl Stage {
    pub fn new(ctx: &mut Context) -> Stage {
        assert!((WIDTH / (BIN_SIZE as i32)) % NB_THREAD as i32 == 0);
        quad_rand::srand(1);

        let mut vertices = vec![0f32; (CIRCLE_SIDES + 1) * 2];
        for i in 0..CIRCLE_SIDES {
            let angle = i as f32 * (2f32 * PI) / CIRCLE_SIDES as f32;
            vertices[i * 2 + 2] = angle.cos();
            vertices[i * 2 + 3] = angle.sin();
        }

        // vertex buffer for static geometry
        let geometry_vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &vertices);

        let indices = (0..CIRCLE_SIDES)
            .flat_map(|i| [0, i as u16 + 1, ((i + 1) % CIRCLE_SIDES + 1) as u16])
            .collect::<Vec<u16>>();

        let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &indices);

        let mut colors = vec![Vec3::ZERO; MAX_PARTICLES];
        let file = File::open("colors.txt");
        match file {
            Ok(mut file) => {
                let mut data: Vec<u8> = Vec::new();
                file.read_to_end(&mut data).unwrap();
                if data.len() == 3 * MAX_PARTICLES {
                    colors = data
                        .chunks(3)
                        .map(|col| {
                            vec3(
                                col[0] as f32 / 255.0,
                                col[1] as f32 / 255.0,
                                col[2] as f32 / 255.0,
                            )
                        })
                        .collect();
                } else {
                    println!(
                        "colors.txt file is meant for {} particles, generating random colors",
                        data.len() / 3
                    );
                    for col in colors.iter_mut() {
                        *col = vec3(
                            quad_rand::gen_range(0.0, 1.0),
                            quad_rand::gen_range(0.0, 1.0),
                            quad_rand::gen_range(0.0, 1.0),
                        );
                    }
                }
            }
            _ => {
                for col in colors.iter_mut() {
                    *col = vec3(
                        quad_rand::gen_range(0.0, 1.0),
                        quad_rand::gen_range(0.0, 1.0),
                        quad_rand::gen_range(0.0, 1.0),
                    );
                }
            }
        }

        let colors_vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &colors);

        let pos = (0..MAX_PARTICLES)
            .map(|_| vec3(0.0, 0.0, quad_rand::gen_range(2.0, 2.5)))
            .collect::<Vec<Vec3>>();

        // empty, dynamic instance-data vertex buffer
        let positions_vertex_buffer = Buffer::stream(
            ctx,
            BufferType::VertexBuffer,
            MAX_PARTICLES * std::mem::size_of::<Vec3>(),
        );

        let bindings = Bindings {
            vertex_buffers: vec![
                geometry_vertex_buffer,
                positions_vertex_buffer,
                colors_vertex_buffer,
            ],
            index_buffer,
            images: vec![],
        };

        let shader = Shader::new(ctx, shader::VERTEX, shader::FRAGMENT, shader::meta()).unwrap();

        let pipeline = Pipeline::new(
            ctx,
            &[
                BufferLayout::default(),
                BufferLayout {
                    step_func: VertexStep::PerInstance,
                    ..Default::default()
                },
                BufferLayout {
                    step_func: VertexStep::PerInstance,
                    ..Default::default()
                },
            ],
            &[
                VertexAttribute::with_buffer("pos", VertexFormat::Float2, 0),
                VertexAttribute::with_buffer("pos_radius", VertexFormat::Float3, 1),
                VertexAttribute::with_buffer("color0", VertexFormat::Float3, 2),
            ],
            shader,
        );

        let bin_w = (WIDTH as f32 / BIN_SIZE).ceil() as usize;
        let bin_h = (HEIGHT as f32 / BIN_SIZE).ceil() as usize;

        let num_bin = bin_w * bin_h;

        let bins = (0..num_bin).map(|_| Bin::new()).collect();

        Stage {
            pipeline,
            bindings,
            len: 0,
            pos,
            sorted_pos: vec![(Vec3::ZERO, 0); MAX_PARTICLES],
            last_pos: vec![Vec2::ZERO; MAX_PARTICLES],
            bin_start: vec![0; num_bin + 1],
            colors,
            bins,
            bin_w,
            bin_h,
            num_bin,
            last_frame: Instant::now(),
            frame_count: 0,
            mouse_pressed: false,
            mouse_pos: Vec2::ZERO,
            can_update: UpdateCommand::Stop,
            pool: Pool::new(NB_THREAD as u32),
        }
    }

    fn apply_constraint(&mut self) {
        //let center = vec2(WIDTH as f32 / 2.0, HEIGHT as f32 / 2.00);
        for i in 0..self.len {
            /*let diff = center - self.pos[i];
            let len = diff.length();
            if len > 400.0 - self.radii[i] {
                let n = diff / len;
                self.pos[i] = center - n * (400.0 - self.radii[i]);
            }*/

            let v = self.pos[i].xy() - vec2(850.0, 600.0);
            let dist2 = v.length_squared();
            let min_dist = self.pos[i].z + 100.0;
            if dist2 < min_dist * min_dist {
                let dist = dist2.sqrt();
                let n = v / dist;
                self.pos[i] -= (n * 0.1 * (dist - min_dist)).extend(0.0);
            }

            let factor = 0.75;

            if self.pos[i].x > WIDTH as f32 - 100.0 - self.pos[i].z {
                self.pos[i].x += factor * (WIDTH as f32 - 100.0 - self.pos[i].z - self.pos[i].x);
            }
            if self.pos[i].x < 100.0 + self.pos[i].z {
                self.pos[i].x += factor * (100.0 + self.pos[i].z - self.pos[i].x);
            }
            if self.pos[i].y > HEIGHT as f32 - 100.0 - self.pos[i].z {
                self.pos[i].y += factor * (HEIGHT as f32 - 100.0 - self.pos[i].z - self.pos[i].y);
            }
            if self.pos[i].y < 100.0 + self.pos[i].z {
                self.pos[i].y += factor * (100.0 + self.pos[i].z - self.pos[i].y);
            }
        }
    }

    fn update_pos(&mut self, dt: f32) {
        for i in 0..self.len {
            let diff = self.pos[i].xy() - self.last_pos[i];
            self.last_pos[i] = self.pos[i].xy();
            self.pos[i] += (diff + vec2(10.0, 200.0) * (dt * dt)).extend(0.0);
        }
    }

    fn fill_bins(&mut self) {
        self.bins.iter_mut().for_each(|b| b.indexes.clear());

        for i in 0..self.len {
            let pos = self.pos[i].xy() / BIN_SIZE;
            self.bins[pos.y.floor() as usize + pos.x.floor() as usize * self.bin_h]
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
        self.bin_start[self.num_bin] = self.len;
    }

    fn avoid_obstacle(&mut self, pos: Vec2, size: f32) {
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

    fn check_collisions(&mut self) {
        let chunk_size = self.num_bin / NB_THREAD;
        let thread_width = self.bin_w / NB_THREAD;

        let breakpoints_thread = self
            .bin_start
            .iter()
            .take(self.num_bin)
            .step_by(chunk_size)
            .map(|&i| i)
            .collect::<Vec<usize>>();

        let check_slice = |slice_pos: &mut [(Vec3, usize)],
                           offset: usize,
                           start_x: usize,
                           width: usize,
                           wall: usize| {
            if slice_pos.len() == 0 {
                return;
            }
            for break_i in (start_x * self.bin_h)..((start_x + width) * self.bin_h) {
                let bin_x = break_i / self.bin_h;
                let bin_y = break_i % self.bin_h;
                if bin_x < start_x + wall
                    || bin_x >= (start_x + width) - wall
                    || bin_y < 1
                    || bin_y >= self.bin_h - 1
                {
                    continue;
                }

                for i1 in self.bin_start[break_i]..self.bin_start[break_i + 1] {
                    let (pos1, _) = slice_pos[i1 - offset];

                    for off_i in -1..=1 {
                        for off_j in -1..=1 {
                            let bin2ind = (bin_y as i32 + off_j) as usize
                                + (bin_x as i32 + off_i) as usize * self.bin_h;

                            for i2 in self.bin_start[bin2ind]..self.bin_start[bin2ind + 1] {
                                if i1 >= i2 {
                                    continue;
                                }

                                let (pos2, _) = slice_pos[i2 - offset];
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

        for odd in 0..2 {
            self.pool.scoped(|scope| {
                for (slice_i, (slice_pos, breakpoint)) in
                    ChunksMutIndices::new(&mut self.sorted_pos, &breakpoints_thread)
                        .enumerate()
                        .filter(|(i, _)| i % 2 == odd)
                {
                    scope.execute(move || {
                        check_slice(
                            slice_pos,
                            breakpoint,
                            slice_i * thread_width,
                            thread_width,
                            1,
                        )
                    });
                }
            });
        }

        let breakpoints_borders = self
            .bin_start
            .iter()
            .skip(chunk_size / 2)
            .take(self.num_bin - chunk_size)
            .step_by(chunk_size)
            .map(|&i| i)
            .collect::<Vec<usize>>();

        // check collisions across thread borders
        self.pool.scoped(|scope| {
            for (bid, (slice_pos, breakpoint)) in
                ChunksMutIndices::new(&mut self.sorted_pos, &breakpoints_borders).enumerate()
            {
                scope.execute(move || {
                    check_slice(slice_pos, breakpoint, (bid + 1) * thread_width - 1, 2, 0)
                });
            }
        });

        for i in 0..self.len {
            self.pos[self.sorted_pos[i].1] = self.sorted_pos[i].0;
        }
    }

    fn add_object(&mut self, pos: Vec2, vel: Vec2) {
        self.pos[self.len] = pos.extend(self.pos[self.len].z);
        self.last_pos[self.len] = pos - vel;
        self.len += 1;
    }
}

impl EventHandler for Stage {
    fn update(&mut self, _: &mut Context) {
        match self.can_update {
            UpdateCommand::Stop => return,
            _ => (),
        }

        let start = Instant::now();
        let dt = 1. / 60.;

        // emit new particles
        let dir = vec2(2.0, 1.0).normalize();
        let space = 8.0;
        for i in 0..8 {
            let off_y = i as f32 * space;
            for j in 0..5 {
                if self.len >= MAX_PARTICLES {
                    break;
                }
                self.add_object(
                    vec2(200.0, 200.0 + off_y) + dir * space * j as f32,
                    dir * 5.0,
                );
            }
        }

        // update particle positions
        for _ in 0..8 {
            self.update_pos(dt / 8.0);
            self.apply_constraint();
            self.fill_bins();
            self.check_collisions();
        }

        if self.mouse_pressed {
            self.avoid_obstacle(self.mouse_pos, 50.0);
        }

        self.frame_count += 1;
        if self.frame_count % 30 == 0 {
            println!(
                "objects: {}, fps: {}, time to update: {}",
                self.len,
                1000000 / self.last_frame.elapsed().as_micros(),
                start.elapsed().as_micros()
            );
        }
        self.last_frame = Instant::now();

        match self.can_update {
            UpdateCommand::OneFrame => self.can_update = UpdateCommand::Stop,
            _ => (),
        }
    }

    fn mouse_motion_event(&mut self, _ctx: &mut Context, x: f32, y: f32) {
        if self.mouse_pressed {
            self.mouse_pos = vec2(x, y);
        }
    }

    fn mouse_button_down_event(&mut self, _: &mut Context, button: MouseButton, x: f32, y: f32) {
        if button == MouseButton::Left {
            self.mouse_pos = vec2(x, y);
            self.mouse_pressed = true;
        }
    }

    fn mouse_button_up_event(&mut self, _: &mut Context, button: MouseButton, _: f32, _: f32) {
        if button == MouseButton::Left {
            self.mouse_pressed = false;
        }
    }

    fn key_down_event(&mut self, ctx: &mut Context, keycode: KeyCode, _: KeyMods, _: bool) {
        match keycode {
            KeyCode::B => {
                let cols = self
                    .bins
                    .iter()
                    .map(|_| {
                        vec3(
                            quad_rand::gen_range(0.0, 1.0),
                            quad_rand::gen_range(0.0, 1.0),
                            quad_rand::gen_range(0.0, 1.0),
                        )
                    })
                    .collect::<Vec<Vec3>>();

                for (bi, bin) in self.bins.iter().enumerate() {
                    for &i in bin.indexes.iter() {
                        self.colors[i] = cols[bi];
                    }
                }

                let colors_vertex_buffer =
                    Buffer::immutable(ctx, BufferType::VertexBuffer, &self.colors);
                self.bindings.vertex_buffers[2] = colors_vertex_buffer;
            }
            KeyCode::C => {
                for i in 0..self.len {
                    self.colors[i] = vec3(
                        self.pos[i].x / WIDTH as f32,
                        (self.pos[i].y - 400.0) / HEIGHT as f32,
                        (self.pos[i].x + self.pos[i].y) / 2000.0,
                    );
                }

                let colors_vertex_buffer =
                    Buffer::immutable(ctx, BufferType::VertexBuffer, &self.colors);
                self.bindings.vertex_buffers[2] = colors_vertex_buffer;
            }
            KeyCode::T => {
                let cols = (0..NB_THREAD)
                    .map(|_| {
                        vec3(
                            quad_rand::gen_range(0.0, 1.0),
                            quad_rand::gen_range(0.0, 1.0),
                            quad_rand::gen_range(0.0, 1.0),
                        )
                    })
                    .collect::<Vec<Vec3>>();

                for (bi, bin) in self.bins.iter().enumerate() {
                    for &i in bin.indexes.iter() {
                        self.colors[i] = cols[(bi / self.bin_w) / (self.bin_w / NB_THREAD)];
                    }
                }

                let colors_vertex_buffer =
                    Buffer::immutable(ctx, BufferType::VertexBuffer, &self.colors);
                self.bindings.vertex_buffers[2] = colors_vertex_buffer;
            }
            KeyCode::S => {
                let mut file = File::create("colors.txt").unwrap();
                let data = self
                    .colors
                    .iter()
                    .flat_map(|c| c.to_array())
                    .map(|v| (v * 255.0).clamp(0.0, 255.0) as u8)
                    .collect::<Vec<u8>>();
                file.write_all(&data).unwrap();
                println!("colors written to file");
            }
            KeyCode::I => {
                let img = ImageReader::open("anime.jpg")
                    .unwrap()
                    .decode()
                    .unwrap()
                    .to_rgb32f();

                for i in 0..self.len {
                    self.colors[i] = match self.pos[i].y < 400.0 {
                        true => Vec3::ONE,
                        false => {
                            let x = ((self.pos[i].x - 100.0) / (WIDTH - 200) as f32
                                * img.width() as f32) as u32;
                            let y = ((self.pos[i].y - 400.0) / (HEIGHT - 500) as f32
                                * img.height() as f32) as u32;
                            let pixel = img.get_pixel(x, y);
                            vec3(pixel[0], pixel[1], pixel[2])
                        }
                    }
                }
                let colors_vertex_buffer =
                    Buffer::immutable(ctx, BufferType::VertexBuffer, &self.colors);
                self.bindings.vertex_buffers[2] = colors_vertex_buffer;

                println!("loaded image");
            }
            KeyCode::N => self.can_update = UpdateCommand::OneFrame,
            KeyCode::Space => {
                self.can_update = match self.can_update {
                    UpdateCommand::OneFrame => UpdateCommand::Continue,
                    UpdateCommand::Continue => UpdateCommand::Stop,
                    UpdateCommand::Stop => UpdateCommand::Continue,
                }
            }
            _ => (),
        }
    }

    fn draw(&mut self, ctx: &mut Context) {
        self.bindings.vertex_buffers[1].update(ctx, &self.pos);

        let proj = Mat4::orthographic_lh(0.0, WIDTH as f32, HEIGHT as f32, 0.0, 0.0, 1.0);

        ctx.begin_default_pass(Default::default());

        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&self.bindings);
        ctx.apply_uniforms(&shader::Uniforms { mvp: proj });
        ctx.draw(0, (CIRCLE_SIDES * 3) as i32, self.len as i32);
        ctx.end_render_pass();

        ctx.commit_frame();
    }
}

fn main() {
    miniquad::start(
        conf::Conf {
            window_width: WIDTH,
            window_height: HEIGHT,
            high_dpi: false,
            ..Default::default()
        },
        |ctx| Box::new(Stage::new(ctx)),
    );
}

mod shader {
    use miniquad::*;

    pub const VERTEX: &str = r#"#version 100
    attribute vec2 pos;
    attribute vec3 pos_radius;
    attribute vec3 color0;

    varying lowp vec4 color;

    uniform mat4 mvp;

    void main() {
        vec4 pos = vec4(pos * pos_radius.z + pos_radius.xy, 0.0, 1.0);
        gl_Position = mvp * pos;
        color = vec4(color0, 1.0);
    }
    "#;

    pub const FRAGMENT: &str = r#"#version 100
    varying lowp vec4 color;

    void main() {
        gl_FragColor = color;
    }
    "#;

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: vec![],
            uniforms: UniformBlockLayout {
                uniforms: vec![UniformDesc::new("mvp", UniformType::Mat4)],
            },
        }
    }

    #[repr(C)]
    pub struct Uniforms {
        pub mvp: glam::Mat4,
    }
}

struct ChunksMutIndices<'a, T: 'a> {
    v: *mut [T],
    breakpoints: &'a [usize],
    curr_ind: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: 'a> ChunksMutIndices<'a, T> {
    #[inline]
    fn new(slice: &'a mut [T], breakpoints: &'a [usize]) -> Self {
        Self {
            v: slice,
            breakpoints,
            curr_ind: 0,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> Iterator for ChunksMutIndices<'a, T> {
    type Item = (&'a mut [T], usize);

    #[inline]
    fn next(&mut self) -> Option<(&'a mut [T], usize)> {
        if self.v.is_empty() || self.curr_ind >= self.breakpoints.len() {
            None
        } else {
            let siz = if self.curr_ind < self.breakpoints.len() - 1 {
                self.breakpoints[self.curr_ind + 1] - self.breakpoints[self.curr_ind]
            } else {
                self.v.len()
            };
            self.curr_ind += 1;
            // SAFETY: The self.v contract ensures that any split_at_mut is valid.
            let (head, tail) = unsafe { self.v.split_at_mut(siz) };
            self.v = tail;
            // SAFETY: Nothing else points to or will point to the contents of this slice.
            Some(unsafe { (&mut *head, self.breakpoints[self.curr_ind - 1]) })
        }
    }
}
