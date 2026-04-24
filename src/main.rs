#![feature(raw_slice_split)]

use std::{
    f32::consts::PI,
    fs::File,
    io::{Read, Write},
    time::Instant,
};

mod chunk_iter;
mod physics;
mod shader;

use miniquad::*;

use image::ImageReader;

use glam::{vec2, vec3, Mat4, Vec2, Vec3};
use physics::{Physics, BIN_W, MAX_PARTICLES, NB_THREAD};

const CIRCLE_SIDES: usize = 12;
const WIDTH: usize = 1200;
const HEIGHT: usize = 1200;

enum UpdateCommand {
    OneFrame,
    Continue,
    Stop,
    Quit,
}

struct Stage {
    ctx: Box<dyn RenderingBackend>,
    pipeline: Pipeline,
    bindings: Bindings,

    physics: Physics,
    colors: Vec<Vec3>,
    last_frame: Instant,
    frame_count: usize,
    mouse_pressed: bool,
    mouse_pos: Vec2,
    can_update: UpdateCommand,
    accumulate_time: u128,
}

impl Stage {
    pub fn new() -> Stage {
        let mut ctx: Box<dyn RenderingBackend> = window::new_rendering_backend();

        quad_rand::srand(1);

        let mut vertices = vec![0f32; (CIRCLE_SIDES + 1) * 2];
        for i in 0..CIRCLE_SIDES {
            let angle = i as f32 * (2f32 * PI) / CIRCLE_SIDES as f32;
            vertices[i * 2 + 2] = angle.cos();
            vertices[i * 2 + 3] = angle.sin();
        }

        let geometry_vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&vertices),
        );

        let indices = (0..CIRCLE_SIDES)
            .flat_map(|i| [0, i as u16 + 1, ((i + 1) % CIRCLE_SIDES + 1) as u16])
            .collect::<Vec<u16>>();
        let index_buffer = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&indices),
        );

        let physics = Physics::new();

        let colors = Stage::init_colors();
        let colors_vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&colors),
        );

        let positions_vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Stream,
            BufferSource::empty::<Vec3>(MAX_PARTICLES),
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

        let shader = ctx
            .new_shader(
                ShaderSource::Glsl {
                    vertex: shader::VERTEX,
                    fragment: shader::FRAGMENT,
                },
                shader::meta(),
            )
            .unwrap();

        let pipeline = ctx.new_pipeline(
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
            PipelineParams::default(),
        );

        Stage {
            ctx,
            pipeline,
            bindings,
            physics,
            colors,
            last_frame: Instant::now(),
            frame_count: 0,
            mouse_pressed: false,
            mouse_pos: Vec2::ZERO,
            can_update: UpdateCommand::Continue,
            accumulate_time: 0,
        }
    }

    fn init_colors() -> Vec<Vec3> {
        let file = File::open("colors.txt");
        match file {
            Ok(mut file) => {
                let mut data: Vec<u8> = Vec::new();
                file.read_to_end(&mut data).unwrap();
                if data.len() == 3 * MAX_PARTICLES {
                    data.chunks(3)
                        .map(|col| {
                            vec3(
                                col[0] as f32 / 255.0,
                                col[1] as f32 / 255.0,
                                col[2] as f32 / 255.0,
                            )
                        })
                        .collect()
                } else {
                    println!(
                        "colors.txt file is meant for {} particles, generating random colors",
                        data.len() / 3
                    );
                    Stage::init_random_colors()
                }
            }
            _ => Stage::init_random_colors(),
        }
    }

    fn init_random_colors() -> Vec<Vec3> {
        (0..MAX_PARTICLES)
            .map(|_| {
                vec3(
                    quad_rand::gen_range(0.0, 1.0),
                    quad_rand::gen_range(0.0, 1.0),
                    quad_rand::gen_range(0.0, 1.0),
                )
            })
            .collect()
    }
}

impl EventHandler for Stage {
    fn update(&mut self) {
        match self.can_update {
            UpdateCommand::Stop => return,
            UpdateCommand::Quit => {
                window::quit();
                return;
            }
            _ => (),
        }

        let start = Instant::now();
        let dt = 1. / 60.;

        self.physics.emit_flow();

        for _ in 0..10 {
            self.physics.step(dt / 10.0);
        }

        if self.mouse_pressed {
            self.physics.avoid_obstacle(self.mouse_pos, 50.0);
        }

        self.frame_count += 1;
        self.accumulate_time += self.last_frame.elapsed().as_micros();
        if self.frame_count % 30 == 0 {
            println!(
                "objects: {}, fps: {}, time to update: {}",
                self.physics.nb_particles(),
                1000000 / (self.accumulate_time / 30),
                start.elapsed().as_micros()
            );
            self.accumulate_time = 0;
        }
        self.last_frame = Instant::now();

        if let UpdateCommand::OneFrame = self.can_update {
            self.can_update = UpdateCommand::Stop;
        }
    }

    fn mouse_motion_event(&mut self, x: f32, y: f32) {
        if self.mouse_pressed {
            self.mouse_pos = vec2(x, y);
        }
    }

    fn mouse_button_down_event(&mut self, button: MouseButton, x: f32, y: f32) {
        if button == MouseButton::Left {
            self.mouse_pos = vec2(x, y);
            self.mouse_pressed = true;
        }
    }

    fn mouse_button_up_event(&mut self, button: MouseButton, _: f32, _: f32) {
        if button == MouseButton::Left {
            self.mouse_pressed = false;
        }
    }

    fn key_down_event(&mut self, keycode: KeyCode, _: KeyMods, _: bool) {
        let random_color = || {
            vec3(
                quad_rand::gen_range(0.0, 1.0),
                quad_rand::gen_range(0.0, 1.0),
                quad_rand::gen_range(0.0, 1.0),
            )
        };

        match keycode {
            KeyCode::B => {
                quad_rand::srand(2);
                let palette = (0..self.physics.get_bins().len())
                    .map(|_| random_color())
                    .collect::<Vec<Vec3>>();

                for (bi, bin) in self.physics.get_bins().iter().enumerate() {
                    for &i in bin.indexes.iter() {
                        self.colors[i] = palette[bi];
                    }
                }

                let colors_vertex_buffer = self.ctx.new_buffer(
                    BufferType::VertexBuffer,
                    BufferUsage::Immutable,
                    BufferSource::slice(&self.colors),
                );
                self.bindings.vertex_buffers[2] = colors_vertex_buffer;
            }
            KeyCode::T => {
                quad_rand::srand(3);
                let palette = (0..NB_THREAD)
                    .map(|_| random_color())
                    .collect::<Vec<Vec3>>();

                for (bi, bin) in self.physics.get_bins().iter().enumerate() {
                    for &i in bin.indexes.iter() {
                        self.colors[i] = palette[(bi / BIN_W) / (BIN_W / NB_THREAD)];
                    }
                }

                let colors_vertex_buffer = self.ctx.new_buffer(
                    BufferType::VertexBuffer,
                    BufferUsage::Immutable,
                    BufferSource::slice(&self.colors),
                );
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

                let nb = self.physics.nb_particles();
                let points: Vec<Vec3> = self
                    .physics
                    .get_points()
                    .iter()
                    .take(nb)
                    .map(|p| {
                        p.clamp(
                            vec3(100.0 + p.z, 100.0 + p.z, p.z),
                            vec3(WIDTH as f32 - 100.0 - p.z, HEIGHT as f32 - 100.0 - p.z, p.z),
                        )
                    })
                    .collect();

                for (i, point) in points.iter().enumerate() {
                    self.colors[i] = match point.y < 400.0 {
                        true => Vec3::ONE,
                        false => {
                            let x = ((point.x - 100.0) / (WIDTH - 200) as f32
                                * img.width() as f32) as u32;
                            let y = ((point.y - 400.0) / (HEIGHT - 500) as f32
                                * img.height() as f32) as u32;
                            let pixel = img.get_pixel(x, y);
                            vec3(pixel[0], pixel[1], pixel[2])
                        }
                    }
                }

                let colors_vertex_buffer = self.ctx.new_buffer(
                    BufferType::VertexBuffer,
                    BufferUsage::Immutable,
                    BufferSource::slice(&self.colors),
                );
                self.bindings.vertex_buffers[2] = colors_vertex_buffer;

                println!("loaded image");
            }
            KeyCode::N => self.can_update = UpdateCommand::OneFrame,
            KeyCode::Space => {
                self.can_update = match self.can_update {
                    UpdateCommand::Continue => UpdateCommand::Stop,
                    _ => UpdateCommand::Continue,
                }
            }
            KeyCode::Escape => self.can_update = UpdateCommand::Quit,
            _ => (),
        }
    }

    fn draw(&mut self) {
        let points = self.physics.get_points().to_vec();
        self.ctx.buffer_update(
            self.bindings.vertex_buffers[1],
            BufferSource::slice(&points),
        );

        let proj = Mat4::orthographic_lh(0.0, WIDTH as f32, HEIGHT as f32, 0.0, 0.0, 1.0);

        self.ctx.begin_default_pass(Default::default());

        self.ctx.apply_pipeline(&self.pipeline);
        self.ctx.apply_bindings(&self.bindings);
        self.ctx
            .apply_uniforms(UniformsSource::table(&shader::Uniforms { mvp: proj }));
        self.ctx.draw(
            0,
            (CIRCLE_SIDES * 3) as i32,
            self.physics.nb_particles() as i32,
        );
        self.ctx.end_render_pass();

        self.ctx.commit_frame();
    }
}

fn main() {
    miniquad::start(
        conf::Conf {
            window_width: WIDTH as i32,
            window_height: HEIGHT as i32,
            high_dpi: false,
            ..Default::default()
        },
        || Box::new(Stage::new()),
    );
}
