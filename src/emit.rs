//! Particle emission scenes. Each scene is a function over `&mut Physics`
//! that calls `try_add_object` until it runs out of slots or finishes its
//! batch. `dispatch` is the public entry point — physics calls it once per
//! `emit_flow`.

use glam::vec2;

use crate::physics::{Physics, BORDER_PADDING, MAX_RADIUS};
use crate::{HEIGHT, WIDTH};

/// Active emit scene. Pick a variant to change the spawn pattern.
pub const EMIT_SCENE: EmitScene = EmitScene::Stream;

/// Default emission rate, in flow-batches per second (one batch = 48 particles,
/// see `dispatch`).
pub const EMIT_BATCHES_PER_SEC: f32 = 120.0;

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
    /// Narrow column of particles dropping from the top centre. Designed
    /// to be the entry point for path-following obstacle scenes like
    /// `ObstacleScene::Cascade`.
    Stream,
}

/// Emit one flow batch. The active scene (`EMIT_SCENE`) decides what shape
/// the batch takes; `count` lets time-varying scenes (sweeps, bursts) stay
/// deterministic against the physics step counter.
pub fn dispatch(physics: &mut Physics, count: u32) {
    match EMIT_SCENE {
        EmitScene::Single => single(physics),
        EmitScene::TwoCrossing => two_crossing(physics),
        EmitScene::Fountain => fountain(physics, count),
        EmitScene::Pulse => pulse(physics, count),
        EmitScene::Rain => rain(physics, count),
        EmitScene::Stream => stream(physics, count),
    }
}

/// 16x3 diagonal stream from (200, 200), the historical default.
fn single(physics: &mut Physics) {
    let speed = 1.0;
    let dir = vec2(2.0, 1.0).normalize();
    let space = MAX_RADIUS * 2.0 + 0.01;
    for i in 0..16 {
        let off_y = i as f32 * space;
        for j in 0..3 {
            if !physics.try_add_object(
                vec2(200.0, 200.0 + off_y) + dir * space * j as f32,
                dir * speed,
            ) {
                return;
            }
        }
    }
}

/// Two crossing streams from the upper-left and upper-right corners.
/// Half the per-stream height of `single` so the total batch stays ~48.
fn two_crossing(physics: &mut Physics) {
    let speed = 1.0;
    let space = MAX_RADIUS * 2.0 + 0.01;
    let dir_l = vec2(2.0, 1.0).normalize();
    let dir_r = vec2(-2.0, 1.0).normalize();
    for i in 0..8 {
        let off_y = i as f32 * space;
        for j in 0..3 {
            if !physics.try_add_object(
                vec2(200.0, 200.0 + off_y) + dir_l * space * j as f32,
                dir_l * speed,
            ) {
                return;
            }
            if !physics.try_add_object(
                vec2(WIDTH as f32 - 200.0, 200.0 + off_y) + dir_r * space * j as f32,
                dir_r * speed,
            ) {
                return;
            }
        }
    }
}

/// Fan-shaped fountain: 48 particles across a 60-degree cone, all from
/// a single point. The cone direction sweeps slowly using `count` so the
/// fountain feels alive without wall-clock time.
fn fountain(physics: &mut Physics, count: u32) {
    let speed = 1.0;
    let origin = vec2(WIDTH as f32 * 0.5, HEIGHT as f32 - 200.0);
    let sweep = (count as f32 * 0.05).sin() * 0.3;
    let half_cone: f32 = std::f32::consts::FRAC_PI_6;
    let n = 48;
    for k in 0..n {
        let t = k as f32 / (n - 1) as f32;
        let ang = -std::f32::consts::FRAC_PI_2 + sweep + (t - 0.5) * 2.0 * half_cone;
        let dir = vec2(ang.cos(), ang.sin());
        if !physics.try_add_object(origin + dir * MAX_RADIUS, dir * speed) {
            return;
        }
    }
}

/// Pulse: emit a dense ring of particles once per second (the modulus
/// matches `EMIT_BATCHES_PER_SEC` so the cadence is wall-clock-stable).
/// Quiet between bursts.
fn pulse(physics: &mut Physics, count: u32) {
    if !count.is_multiple_of(EMIT_BATCHES_PER_SEC as u32) {
        return;
    }
    let speed = 1.0;
    let center = vec2(WIDTH as f32 * 0.5, 200.0);
    let n = 96;
    for k in 0..n {
        let ang = (k as f32) * std::f32::consts::TAU / (n as f32);
        let dir = vec2(ang.cos(), ang.sin());
        if !physics.try_add_object(center + dir * 30.0, dir * speed) {
            return;
        }
    }
}

/// Stateless splitmix64-style hash. Maps a 64-bit key to a uniform 64-bit
/// output without touching any global RNG state, so emitters can pull
/// per-particle deterministic jitter purely from their `(count, k, channel)`
/// indices. Same `key` always produces the same output, across runs.
fn hash_u64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Sample a deterministic uniform `[-1, 1]` float from `(count, k, channel)`.
/// `channel` separates independent jitter streams (x/y/vx/speed) so they
/// don't correlate visually.
fn jitter(count: u32, k: u32, channel: u32) -> f32 {
    let key = (count as u64) << 32 | ((k as u64) << 8) | channel as u64;
    let h = hash_u64(key);
    let unit = (h >> 40) as f32 / ((1u32 << 24) as f32);
    unit * 2.0 - 1.0
}

/// Stream: a narrow vertical column dropping from the top centre. 4 wide
/// by 12 tall = 48 particles per batch, jittered for variation. Pairs
/// well with path-following obstacle layouts where the entry point is
/// fixed.
fn stream(physics: &mut Physics, count: u32) {
    let speed = 1.0;
    let cols: u32 = 4;
    let rows: u32 = 12;
    let space = MAX_RADIUS * 2.0 + 0.5;
    let column_width = (cols - 1) as f32 * space;
    let x0 = WIDTH as f32 * 0.5 - column_width * 0.5;
    let y0 = BORDER_PADDING + 10.0;
    for r in 0..rows {
        for c in 0..cols {
            let k = r * cols + c;
            let jx = jitter(count, k, 0);
            let jy = jitter(count, k, 1);
            let jvx = jitter(count, k, 2);
            let jvs = jitter(count, k, 3);
            let x = x0 + c as f32 * space + jx * space * 0.3;
            let y = y0 + r as f32 * space + jy * 1.5;
            let vx = jvx * 0.05 * speed;
            let vy = speed * (1.0 + jvs * 0.1);
            if !physics.try_add_object(vec2(x, y), vec2(vx, vy)) {
                return;
            }
        }
    }
}

/// Rain: one row of 48 particles across the top, falling roughly downward.
/// Position and velocity are jittered per drop using a deterministic hash
/// of `(count, k)`, so the row doesn't look like a grid and streaks
/// don't all align — but the simulation is still bit-perfect across runs.
fn rain(physics: &mut Physics, count: u32) {
    let speed = 1.0;
    let n: u32 = 48;
    let span = WIDTH as f32 - 2.0 * BORDER_PADDING - 40.0;
    let step = span / (n - 1) as f32;
    let y0 = BORDER_PADDING + 10.0;
    for k in 0..n {
        let jx = jitter(count, k, 0);
        let jy = jitter(count, k, 1);
        let jvx = jitter(count, k, 2);
        let jvs = jitter(count, k, 3);
        let x = BORDER_PADDING + 20.0 + k as f32 * step + jx * step * 0.5;
        let y = y0 + jy * 5.0;
        let vx = jvx * 0.15 * speed;
        let vy = speed * (1.0 + jvs * 0.15);
        if !physics.try_add_object(vec2(x, y), vec2(vx, vy)) {
            return;
        }
    }
}
