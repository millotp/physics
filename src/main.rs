use std::{
    f32::consts::PI,
    fs::File,
    io::{Read, Write},
    time::Instant,
};

mod physics;
mod shader;

use miniquad::*;

use image::ImageReader;

use glam::{vec2, vec3, Mat4, Vec2, Vec3, Vec3Swizzles};
use physics::{Physics, BIN_W, BORDER_PADDING, MAX_PARTICLES, NB_THREAD};

const CIRCLE_SIDES: usize = 12;
/// Logical simulation world size, in physics units. The bin grid, particle
/// coordinates and ortho projection all use these values.
const WIDTH: usize = 1200;
const HEIGHT: usize = 1200;
/// On-screen window size, in points. Independent of `WIDTH`/`HEIGHT`: the
/// ortho projection maps the world into the window via NDC. Kept square so
/// the world's square aspect ratio is preserved on screen.
const WINDOW_SIZE: i32 = 800;

#[derive(Copy, Clone)]
enum UpdateCommand {
    OneFrame,
    Continue,
    Stop,
    Quit,
}

/// Soft cap on the number of discs needed to draw any obstacle scene
/// (capsules expand into a small chain of overlapping circles). Way more
/// than the busiest current scene needs.
const MAX_OBSTACLE_DISCS: usize = 256;

struct Stage {
    ctx: Box<dyn RenderingBackend>,
    pipeline: Pipeline,
    bindings: Bindings,
    /// Bindings reused for the obstacle draw call: same circle geometry +
    /// index buffer, but the per-instance buffers are dedicated to obstacles.
    obstacle_bindings: Bindings,
    /// Streaming buffer for obstacle (center.xy, radius) packed as `Vec3`.
    obstacle_pos_buffer: BufferId,

    physics: Physics,
    colors: Vec<Vec3>,
    last_frame: Instant,
    frame_count: usize,
    mouse_pressed: bool,
    mouse_pos: Vec2,
    update_mode: UpdateCommand,
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

        // Obstacle rendering: a streaming pos-buffer, an immutable dark-grey
        // colors buffer, and a separate `Bindings` reusing the same circle
        // geometry. The `Stream` usage lets animated scenes refresh
        // positions per frame without re-creating the buffer.
        let obstacle_pos_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Stream,
            BufferSource::empty::<Vec3>(MAX_OBSTACLE_DISCS),
        );
        let obstacle_colors: Vec<Vec3> = (0..MAX_OBSTACLE_DISCS)
            .map(|_| vec3(0.35, 0.35, 0.40))
            .collect();
        let obstacle_color_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&obstacle_colors),
        );
        let obstacle_bindings = Bindings {
            vertex_buffers: vec![
                geometry_vertex_buffer,
                obstacle_pos_buffer,
                obstacle_color_buffer,
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
            obstacle_bindings,
            obstacle_pos_buffer,
            physics,
            colors,
            last_frame: Instant::now(),
            frame_count: 0,
            mouse_pressed: false,
            mouse_pos: Vec2::ZERO,
            update_mode: UpdateCommand::Continue,
            accumulate_time: 0,
        }
    }

    fn init_colors() -> Vec<Vec3> {
        let mut file = match File::open("colors.txt") {
            Ok(file) => file,
            Err(_) => return Stage::init_random_colors(),
        };

        let expected_len = (3 * MAX_PARTICLES) as u64;
        let len = match file.metadata() {
            Ok(meta) => meta.len(),
            Err(_) => return Stage::init_random_colors(),
        };
        if len != expected_len {
            println!(
                "colors.txt file is meant for {} particles, generating random colors",
                len / 3
            );
            return Stage::init_random_colors();
        }

        let mut data: Vec<u8> = Vec::with_capacity(expected_len as usize);
        file.read_to_end(&mut data).unwrap();
        data.chunks(3)
            .map(|col| {
                vec3(
                    col[0] as f32 / 255.0,
                    col[1] as f32 / 255.0,
                    col[2] as f32 / 255.0,
                )
            })
            .collect()
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
        match self.update_mode {
            UpdateCommand::Stop => return,
            UpdateCommand::Quit => {
                window::quit();
                return;
            }
            _ => (),
        }

        let start = Instant::now();
        // 120 Hz panel + vsync: 5 substeps of dt=1/600 advance 1/120 s of sim
        // time per call, matching real-time. Each substep stays at dt=1/600 s,
        // the value the physics is tuned for.
        let dt = 1. / 120.;

        self.physics.emit_flow_for(dt);

        for _ in 0..5 {
            self.physics.step(dt / 5.0);
        }

        if self.mouse_pressed {
            self.physics.avoid_obstacle(self.mouse_pos, 50.0);
        }

        self.frame_count += 1;
        self.accumulate_time += self.last_frame.elapsed().as_micros();
        if self.frame_count.is_multiple_of(30) {
            println!(
                "objects: {}, fps: {}, time to update: {}",
                self.physics.nb_particles(),
                1_000_000 / (self.accumulate_time / 30),
                start.elapsed().as_micros()
            );
            self.accumulate_time = 0;
        }
        self.last_frame = Instant::now();

        if let UpdateCommand::OneFrame = self.update_mode {
            self.update_mode = UpdateCommand::Stop;
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

                // Playable area in world coords. Particles are clamped here
                // before sampling the image so we don't read out of bounds.
                let pad = BORDER_PADDING;
                let world_w = WIDTH as f32 - 2.0 * pad;
                // Reserve the top of the playable area for a white "sky" band
                // (matches the original look). The image is mapped onto the
                // remainder.
                let sky_h: f32 = 300.0;
                let img_top = pad + sky_h;
                let img_h = HEIGHT as f32 - pad - img_top;

                let points: Vec<Vec3> = self
                    .physics
                    .active_points()
                    .iter()
                    .map(|p| {
                        let xy = p.xy().clamp(
                            vec2(pad + p.z, pad + p.z),
                            vec2(WIDTH as f32 - pad - p.z, HEIGHT as f32 - pad - p.z),
                        );
                        xy.extend(p.z)
                    })
                    .collect();

                for (i, point) in points.iter().enumerate() {
                    self.colors[i] = if point.y < img_top {
                        Vec3::ONE
                    } else {
                        let u = (point.x - pad) / world_w;
                        let v = (point.y - img_top) / img_h;
                        let x = (u * img.width() as f32) as u32;
                        let y = (v * img.height() as f32) as u32;
                        let pixel = img.get_pixel(x, y);
                        vec3(pixel[0], pixel[1], pixel[2])
                    };
                }

                let colors_vertex_buffer = self.ctx.new_buffer(
                    BufferType::VertexBuffer,
                    BufferUsage::Immutable,
                    BufferSource::slice(&self.colors),
                );
                self.bindings.vertex_buffers[2] = colors_vertex_buffer;

                println!("loaded image");
            }
            KeyCode::N => self.update_mode = UpdateCommand::OneFrame,
            KeyCode::Space => {
                self.update_mode = match self.update_mode {
                    UpdateCommand::Continue => UpdateCommand::Stop,
                    UpdateCommand::Stop => UpdateCommand::Continue,
                    other => other,
                };
            }
            KeyCode::Escape => self.update_mode = UpdateCommand::Quit,
            _ => (),
        }
    }

    fn draw(&mut self) {
        self.ctx.buffer_update(
            self.bindings.vertex_buffers[1],
            BufferSource::slice(self.physics.active_points()),
        );

        let obstacle_discs = self.physics.obstacle_discs();
        let obstacle_count = obstacle_discs.len().min(MAX_OBSTACLE_DISCS);
        if obstacle_count > 0 {
            self.ctx.buffer_update(
                self.obstacle_pos_buffer,
                BufferSource::slice(&obstacle_discs[..obstacle_count]),
            );
        }

        // Camera: zoom to the playable area (i.e. inside `BORDER_PADDING`)
        // so the wall buffer that physics needs doesn't show up as a visible
        // black border in the window.
        let proj = Mat4::orthographic_lh(
            BORDER_PADDING,
            WIDTH as f32 - BORDER_PADDING,
            HEIGHT as f32 - BORDER_PADDING,
            BORDER_PADDING,
            0.0,
            1.0,
        );

        self.ctx.begin_default_pass(Default::default());

        self.ctx.apply_pipeline(&self.pipeline);
        self.ctx
            .apply_uniforms(UniformsSource::table(&shader::Uniforms { mvp: proj }));

        // Obstacles first so particles render on top of them.
        if obstacle_count > 0 {
            self.ctx.apply_bindings(&self.obstacle_bindings);
            self.ctx
                .draw(0, (CIRCLE_SIDES * 3) as i32, obstacle_count as i32);
        }

        self.ctx.apply_bindings(&self.bindings);
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
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--bench") {
        run_bench(&args);
        return;
    }
    miniquad::start(
        conf::Conf {
            // The window is in screen points, decoupled from `WIDTH`/`HEIGHT`
            // (the simulation's logical world). The ortho projection maps the
            // 1200x1200 world onto whatever the framebuffer is, so circles
            // stay round and the window can be any square size that fits on
            // the user's display.
            window_width: WINDOW_SIZE,
            window_height: WINDOW_SIZE,
            // HiDPI on so the framebuffer matches the display's physical
            // pixel density (crisp on Retina). Aspect stays square since
            // both axes scale by the same dpi factor.
            high_dpi: true,
            // Lock the window to a square so the simulation can never be
            // squashed by a manual resize.
            window_resizable: false,
            // Vsync ON. Software-only fps caps overshoot routinely on macOS, so
            // we let the display drive the cadence and tune `update()` accordingly.
            platform: conf::Platform {
                swap_interval: Some(1),
                ..Default::default()
            },
            ..Default::default()
        },
        || Box::new(Stage::new()),
    );
}

/// Headless deterministic benchmark: per "frame" we emit one batch and run
/// 10 substeps of `dt = 1/60 / 10`. Defaults give a ~5 s run at ~28k particles.
///
/// Usage: `physics --bench [frames=100] [warmup=500] [seed=1]`
fn run_bench(args: &[String]) {
    fn parse_or<T: std::str::FromStr>(s: Option<&String>, d: T) -> T {
        s.and_then(|v| v.parse().ok()).unwrap_or(d)
    }
    let positional: Vec<&String> = args.iter().skip(1).filter(|a| *a != "--bench").collect();
    let frames: usize = parse_or(positional.first().copied(), 100usize);
    let warmup: usize = parse_or(positional.get(1).copied(), 500usize);
    let seed: u64 = parse_or(positional.get(2).copied(), 1u64);

    quad_rand::srand(seed);
    let mut physics = Physics::new();
    let dt: f32 = 1.0 / 60.0;

    for _ in 0..warmup {
        physics.emit_flow();
        for _ in 0..10 {
            physics.step(dt / 10.0);
        }
    }

    // Reset timers after warmup so only the measured window counts.
    physics::reset_breakdown();

    let mut samples: Vec<u128> = Vec::with_capacity(frames);
    let total_start = Instant::now();
    for _ in 0..frames {
        let t = Instant::now();
        physics.emit_flow();
        for _ in 0..10 {
            physics.step(dt / 10.0);
        }
        samples.push(t.elapsed().as_micros());
    }
    let total = total_start.elapsed().as_micros();

    samples.sort_unstable();
    let n = samples.len().max(1);
    let p = |q: f64| samples[(q * (n - 1) as f64).round() as usize];
    let min = samples[0];
    let median = p(0.50);
    let p95 = p(0.95);
    let p99 = p(0.99);
    let max = *samples.last().unwrap();
    let mean: u128 = samples.iter().sum::<u128>() / n as u128;

    eprintln!(
        "bench: particles_end={} frames={} warmup={} seed={}",
        physics.nb_particles(),
        frames,
        warmup,
        seed
    );
    eprintln!(
        "       per-frame us: min={min} median={median} mean={mean} p95={p95} p99={p99} max={max}"
    );
    eprintln!(
        "       total_ms={:.2} fps_avg={:.1}",
        total as f64 / 1000.0,
        frames as f64 * 1_000_000.0 / total as f64
    );
    if std::env::var("BENCH_BREAKDOWN").is_ok() {
        physics::print_breakdown(frames * 10);
    }
    // One-line machine-readable summary on stdout:
    println!(
        "particles_end={} frames={} min={min} median={median} mean={mean} p95={p95} p99={p99} max={max} total_ms={:.2} fps_avg={:.1}",
        physics.nb_particles(),
        frames,
        total as f64 / 1000.0,
        frames as f64 * 1_000_000.0 / total as f64
    );
}

/// Headless replay of the GUI's `update()` body. Identical control flow,
/// no rendering or input. Prints a position signature so determinism can
/// be asserted byte-perfectly across runs of the same scene.
fn run_gui_sig(args: &[String]) {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).skip(1).collect();
    let parse_or = |s: Option<&&String>, default: u64| -> u64 {
        s.and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let frames: u64 = parse_or(positional.first(), 600);
    let seed: u64 = parse_or(positional.get(1), 1);

    quad_rand::srand(seed);
    let mut physics = Physics::new();
    let dt: f32 = 1.0 / 120.0;
    for _ in 0..frames {
        physics.emit_flow_for(dt);
        for _ in 0..5 {
            physics.step(dt / 5.0);
        }
    }

    let mut sx: f64 = 0.0;
    let mut sy: f64 = 0.0;
    for p in physics.active_points() {
        sx += p.x as f64;
        sy += p.y as f64;
    }
    println!(
        "gui_sig frames={} seed={} n={} sum_x={:.4} sum_y={:.4}",
        frames,
        seed,
        physics.nb_particles(),
        sx,
        sy
    );
}
