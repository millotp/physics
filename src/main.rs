use std::{
    cell::RefCell,
    f32::consts::PI,
    fs::File,
    io::{Read, Write},
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use crossbeam_channel::{bounded, Receiver, Sender};
use miniquad::*;

use image::io::Reader as ImageReader;

use glam::{vec2, vec3, Mat4, Vec2, Vec3};

const MAX_PARTICLES: usize = 20000;
const CIRCLE_SIDES: usize = 12;
const WIDTH: i32 = 1200;
const HEIGHT: i32 = 1200;
const BIN_SIZE: f32 = 10.0;
const NB_THREAD: usize = 4;

#[derive(Clone)]
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

struct Stage {
    pipeline: Pipeline,
    bindings: Bindings,

    pos: Arc<Mutex<Vec<Vec2>>>,
    last_pos: Vec<Vec2>,
    acc: Vec<Vec2>,
    radii: Arc<Vec<f32>>,
    colors: Vec<Vec3>,
    bins: Arc<RefCell<Vec<Bin>>>,
    bin_w: usize,
    last_frame: Instant,
    mouse_pressed: bool,
    mouse_pos: Vec2,
    start_collide_tx: Vec<Sender<()>>,
    done_collide_rx: Receiver<()>,
}

impl Stage {
    pub fn new(ctx: &mut Context) -> Box<Stage> {
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

        // empty, dynamic instance-data vertex buffer
        let positions_vertex_buffer = Buffer::stream(
            ctx,
            BufferType::VertexBuffer,
            MAX_PARTICLES * std::mem::size_of::<Vec2>(),
        );

        let mut colors = vec![Vec3::ZERO; MAX_PARTICLES];
        let file = File::open("colors.txt");
        match file {
            Ok(mut file) => {
                let mut data: Vec<u8> = Vec::new();
                file.read_to_end(&mut data).unwrap();
                assert!(data.len() == 3 * MAX_PARTICLES);
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
            }
            _ => {
                for i in 0..MAX_PARTICLES {
                    colors[i] = vec3(
                        quad_rand::gen_range(0.0, 1.0),
                        quad_rand::gen_range(0.0, 1.0),
                        quad_rand::gen_range(0.0, 1.0),
                    );
                }
            }
        }

        let colors_vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &colors);

        let radii = Arc::new(
            (0..MAX_PARTICLES)
                .map(|_| quad_rand::gen_range(2.0, 2.5))
                .collect::<Vec<f32>>(),
        );

        let radii_vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &*radii);

        let bindings = Bindings {
            vertex_buffers: vec![
                geometry_vertex_buffer,
                positions_vertex_buffer,
                colors_vertex_buffer,
                radii_vertex_buffer,
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
                BufferLayout {
                    step_func: VertexStep::PerInstance,
                    ..Default::default()
                },
            ],
            &[
                VertexAttribute::with_buffer("pos", VertexFormat::Float2, 0),
                VertexAttribute::with_buffer("inst_pos", VertexFormat::Float2, 1),
                VertexAttribute::with_buffer("color0", VertexFormat::Float3, 2),
                VertexAttribute::with_buffer("radius", VertexFormat::Float1, 3),
            ],
            shader,
        );

        let bin_w = (WIDTH as f32 / BIN_SIZE).ceil() as usize;
        let bin_h = (HEIGHT as f32 / BIN_SIZE).ceil() as usize;

        let bins = Arc::new(RefCell::new(Vec::with_capacity(bin_w * bin_h)));

        for _ in 0..(bin_w * bin_h) {
            bins.borrow_mut().push(Bin::new());
        }

        let pos = Arc::new(Mutex::new(Vec::with_capacity(MAX_PARTICLES)));

        let (start_collide_tx, start_collide_rx): (Vec<Sender<()>>, Vec<Receiver<()>>) =
            (0..NB_THREAD).map(|_| bounded(1)).unzip();
        let (done_collide_tx, done_collide_rx) = bounded(1);

        let stage = Box::new(Stage {
            pipeline,
            bindings,
            pos: Arc::clone(&pos),
            last_pos: Vec::with_capacity(MAX_PARTICLES),
            acc: Vec::with_capacity(MAX_PARTICLES),
            colors,
            radii: Arc::clone(&radii),
            bins: Arc::clone(&bins),
            bin_w,
            last_frame: Instant::now(),
            mouse_pressed: false,
            mouse_pos: Vec2::ZERO,
            start_collide_tx,
            done_collide_rx,
        });

        for (tid, slice) in bins
            .borrow()
            .chunks(bins.borrow().len() / NB_THREAD)
            .enumerate()
        {
            let section_h = bin_h / NB_THREAD;
            let srx = start_collide_rx[tid].clone();
            let dtx = done_collide_tx.clone();

            let pos = Arc::clone(&pos);
            let rad = Arc::clone(&radii);

            thread::spawn(move || loop {
                srx.recv().unwrap();
                for bin_j in 0..section_h {
                    if bin_j < 1 || bin_j >= bin_h - 1 {
                        continue;
                    }
                    for bin_i in 1..(bin_w - 1) {
                        for off_i in (-1i32)..2 {
                            for off_j in (-1i32)..2 {
                                let bin1 = bin_i + bin_j * bin_w;
                                let bin2 = (bin_i as i32 + off_i) as usize
                                    + (bin_j as i32 + off_j) as usize * bin_w;

                                for &i in slice[bin1].indexes.iter() {
                                    for &j in slice[bin2].indexes.iter() {
                                        if i >= j {
                                            continue;
                                        }
                                        let mut p = pos.lock().unwrap();
                                        let v = p[i] - p[j];
                                        let dist2 = v.length_squared();
                                        let min_dist = rad[i] + rad[j];
                                        if dist2 < min_dist * min_dist {
                                            let dist = dist2.sqrt();
                                            let n = v / dist;
                                            let mass_ratio_i = rad[i] / (rad[i] + rad[j]);
                                            let mass_ratio_j = rad[j] / (rad[i] + rad[j]);
                                            let delta = 0.5 * 0.75 * (dist - min_dist);
                                            p[i] -= n * (mass_ratio_j * delta);
                                            p[j] += n * (mass_ratio_i * delta);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                dtx.send(()).unwrap();
            });
        }

        return stage;
    }

    fn apply_gravity(&mut self) {
        for i in 0..self.acc.len() {
            self.acc[i] = self.acc[i] + vec2(10.0, 200.0);
        }
    }

    fn apply_constraint(&mut self) {
        let mut p = self.pos.lock().unwrap();
        //let center = vec2(WIDTH as f32 / 2.0, HEIGHT as f32 / 2.00);
        for i in 0..p.len() {
            /*let diff = center - self.pos[i];
            let len = diff.length();
            if len > 400.0 - self.radii[i] {
                let n = diff / len;
                self.pos[i] = center - n * (400.0 - self.radii[i]);
            }*/
            let v = p[i] - vec2(850.0, 600.0);
            let dist2 = v.length_squared();
            let min_dist = self.radii[i] + 100.0;
            if dist2 < min_dist * min_dist {
                let dist = dist2.sqrt();
                let n = v / dist;
                p[i] -= n * 0.1 * (dist - min_dist);
            }

            let factor = 0.75;

            if p[i].x > WIDTH as f32 - 100.0 - self.radii[i] {
                p[i].x += factor * (WIDTH as f32 - 100.0 - self.radii[i] - p[i].x);
            }
            if p[i].x < 100.0 + self.radii[i] {
                p[i].x += factor * (100.0 + self.radii[i] - p[i].x);
            }
            if p[i].y > HEIGHT as f32 - 100.0 - self.radii[i] {
                p[i].y += factor * (HEIGHT as f32 - 100.0 - self.radii[i] - p[i].y);
            }
            if p[i].y < 100.0 + self.radii[i] {
                p[i].y += factor * (100.0 + self.radii[i] - p[i].y);
            }
        }
    }

    fn update_pos(&mut self, dt: f32) {
        let mut p = self.pos.lock().unwrap();
        for i in 0..p.len() {
            let diff = p[i] - self.last_pos[i];
            self.last_pos[i] = p[i];
            p[i] += diff + self.acc[i] * (dt * dt);
            self.acc[i] = Vec2::ZERO;
        }
    }

    fn fill_bins(&mut self) {
        self.bins
            .borrow_mut()
            .iter_mut()
            .for_each(|b| b.indexes.clear());

        let p = self.pos.lock().unwrap();

        for i in 0..p.len() {
            let pos = p[i] / BIN_SIZE;
            self.bins.borrow_mut()[pos.x as usize + pos.y as usize * self.bin_w]
                .indexes
                .push(i);
        }
    }

    fn avoid_obstacle(&mut self, pos: Vec2, size: f32) {
        let mut p = self.pos.lock().unwrap();
        for i in 0..p.len() {
            let v = p[i] - pos;
            let dist2 = v.length_squared();
            let min_dist = self.radii[i] + size;
            if dist2 < min_dist * min_dist {
                let dist = dist2.sqrt();
                let n = v / dist;
                p[i] -= n * 0.1 * (dist - min_dist);
            }
        }
    }

    /*
    fn check_collisions_bin(&mut self, bin1: usize, bin2: usize) {
        for &i in self.bins[bin1].indexes.iter() {
            for &j in self.bins[bin2].indexes.iter() {
                if i >= j {
                    continue;
                }
                let v = self.pos[i] - self.pos[j];
                let dist2 = v.length_squared();
                let min_dist = self.radii[i] + self.radii[j];
                if dist2 < min_dist * min_dist {
                    let dist = dist2.sqrt();
                    let n = v / dist;
                    let mass_ratio_i = self.radii[i] / (self.radii[i] + self.radii[j]);
                    let mass_ratio_j = self.radii[j] / (self.radii[i] + self.radii[j]);
                    let delta = 0.5 * 0.75 * (dist - min_dist);
                    self.pos[i] -= n * (mass_ratio_j * delta);
                    self.pos[j] += n * (mass_ratio_i * delta);
                }
            }
        }
    }
    */

    fn check_collisions(&mut self) {
        for i in 0..(NB_THREAD / 2) {
            self.start_collide_tx[i * 2].send(()).unwrap();
        }
        for _ in 0..(NB_THREAD / 2) {
            self.done_collide_rx.recv().unwrap();
        }

        for i in 0..(NB_THREAD / 2) {
            self.start_collide_tx[i * 2 + 1].send(()).unwrap();
        }
        for _ in 0..(NB_THREAD / 2) {
            self.done_collide_rx.recv().unwrap();
        }

        /*
                self.bin_order
                    .clone()
                    .into_iter()
                    .for_each(|(bin1, bin2)| self.check_collisions_bin(bin1, bin2));
        */
    }

    fn add_object(&mut self, pos: Vec2, vel: Vec2) {
        self.pos.lock().unwrap().push(pos);
        self.last_pos.push(pos - vel);
        self.acc.push(Vec2::ZERO);
    }
}

impl EventHandler for Stage {
    fn update(&mut self, _: &mut Context) {
        // let elapsed = self.last_frame.elapsed().as_micros();
        // let dt = (elapsed as f32 / 1000000f32).min(0.02);
        let dt = 1. / 60.;
        /*
        println!(
            "{}, {}",
            1000000 / self.last_frame.elapsed().as_micros(),
            self.pos.len()
        );*/
        self.last_frame = Instant::now();

        // emit new particles
        let dir = vec2(2.0, 1.0).normalize();
        for i in 0..8 {
            let off_y = i as f32 * 8.0;
            for j in 0..5 {
                if self.acc.len() >= MAX_PARTICLES {
                    break;
                }
                self.add_object(vec2(200.0, 200.0 + off_y) + dir * 8.0 * j as f32, dir * 1.0);
            }
        }

        // update particle positions
        for _ in 0..8 {
            self.apply_gravity();
            self.update_pos(dt / 8.0);
            self.apply_constraint();
            self.fill_bins();
            self.check_collisions();
        }

        if self.mouse_pressed {
            self.avoid_obstacle(self.mouse_pos, 50.0);
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
            KeyCode::C => {
                let p = self.pos.lock().unwrap();
                for i in 0..p.len() {
                    self.colors[i] = vec3(
                        p[i].x / WIDTH as f32,
                        (p[i].y - 400.0) / HEIGHT as f32,
                        (p[i].x + p[i].y) / 2000.0,
                    );
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
                file.write(&data).unwrap();
                println!("colors written to file");
            }
            KeyCode::I => {
                let img = ImageReader::open("anime.jpg")
                    .unwrap()
                    .decode()
                    .unwrap()
                    .to_rgb32f();

                let p = self.pos.lock().unwrap();
                for i in 0..p.len() {
                    self.colors[i] = match p[i].y < 400.0 {
                        true => Vec3::ONE,
                        false => {
                            let x = ((p[i].x - 100.0) / (WIDTH - 200) as f32 * img.width() as f32)
                                as u32;
                            let y = ((p[i].y - 400.0) / (HEIGHT - 500) as f32 * img.height() as f32)
                                as u32;
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
            _ => (),
        }
    }

    fn draw(&mut self, ctx: &mut Context) {
        self.bindings.vertex_buffers[1].update(ctx, &self.pos.lock().unwrap()[..]);

        let proj = Mat4::orthographic_lh(0.0, WIDTH as f32, HEIGHT as f32, 0.0, 0.0, 1.0);

        ctx.begin_default_pass(Default::default());

        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&self.bindings);
        ctx.apply_uniforms(&shader::Uniforms { mvp: proj });
        ctx.draw(0, (CIRCLE_SIDES * 3) as i32, self.acc.len() as i32);
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
        |mut ctx| Stage::new(&mut ctx),
    );
}

mod shader {
    use miniquad::*;

    pub const VERTEX: &str = r#"#version 100
    attribute vec2 pos;
    attribute vec2 inst_pos;
    attribute vec3 color0;
    attribute float radius;

    varying lowp vec4 color;

    uniform mat4 mvp;

    void main() {
        vec4 pos = vec4(pos * radius + inst_pos, 0.0, 1.0);
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