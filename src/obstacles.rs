//! Obstacle types and the per-scene layout. Owns the static/animated
//! geometry and the spatial-bin acceleration structure used by the per-particle
//! collision pass in `physics`.

use glam::{vec2, Vec2, Vec3};

use crate::physics::{BIN_H, BIN_SIZE, BIN_W, MAX_RADIUS, NUM_BIN};
use crate::{HEIGHT, WIDTH};

/// Active obstacle scene. Pick a variant to change the playfield layout.
pub const OBSTACLE_SCENE: ObstacleScene = ObstacleScene::Cascade;

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
    /// Five small rotating crosses (two perpendicular rects per cross),
    /// each spinning at a different phase.
    RotatingCrosses,
    /// Ring of pegs around the centre.
    Ring,
    /// Three big circles at distinctive positions.
    FewCircles,
    /// A multi-stage path: splitter circle, V-funnel, rotating paddle,
    /// flanking circles, narrowing chevron, second paddle, bottom deflector.
    /// Designed to be paired with `EmitScene::Stream` so particles enter
    /// from the top centre and visibly thread through every stage.
    Cascade,
}

/// Static and animated colliders.
///
/// - `Circle` — origin disc.
/// - `Rect` — oriented box. `axis` is the unit direction of the box's local
///   +x axis; `half_extents` is half the box width (along `axis`) and half
///   the box height (along the perpendicular).
#[derive(Copy, Clone, Debug)]
pub enum Obstacle {
    Circle {
        center: Vec2,
        radius: f32,
    },
    Rect {
        center: Vec2,
        axis: Vec2,
        half_extents: Vec2,
    },
}

/// Per-step angular velocity for the rotating bar variants (rad / sim-second).
/// Stays deterministic since it's tied to the physics step count, not
/// wall-clock time.
const BAR_ANGULAR_VEL: f32 = 5.0;

/// Per-substep push factor for the static obstacle (and the mouse cursor).
pub const OBSTACLE_PUSH: f32 = 0.1;

/// Per-scene obstacle layout. The `time` argument lets animated scenes (e.g.
/// rotating bars) parameterise their geometry deterministically against the
/// caller's `sim_time`. Static scenes ignore `time`.
pub fn build(scene: ObstacleScene, time: f32) -> Vec<Obstacle> {
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
            let ang = time * BAR_ANGULAR_VEL;
            let axis = vec2(ang.cos(), ang.sin());
            vec![Obstacle::Rect {
                center: vec2(cx, cy),
                axis,
                half_extents: vec2(280.0, 20.0),
            }]
        }
        ObstacleScene::RotatingCrosses => {
            // 5 crosses: one centre + 4 corners of an inner square.
            let offset = 280.0;
            let centers = [
                vec2(cx, cy),
                vec2(cx - offset, cy - offset),
                vec2(cx + offset, cy - offset),
                vec2(cx - offset, cy + offset),
                vec2(cx + offset, cy + offset),
            ];
            // Phases stagger the rotation so the crosses don't spin in
            // lockstep. Constant array → bit-deterministic across runs.
            let phases = [0.0, 0.7, 1.4, 2.1, 2.8];
            let arm_half = vec2(80.0, 6.0);
            let mut out = Vec::with_capacity(centers.len() * 2);
            for (c, &phase) in centers.iter().zip(phases.iter()) {
                let ang = time * BAR_ANGULAR_VEL + phase;
                let axis = vec2(ang.cos(), ang.sin());
                let perp = vec2(-axis.y, axis.x);
                out.push(Obstacle::Rect {
                    center: *c,
                    axis,
                    half_extents: arm_half,
                });
                out.push(Obstacle::Rect {
                    center: *c,
                    axis: perp,
                    half_extents: arm_half,
                });
            }
            out
        }
        ObstacleScene::Ring => {
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
        ObstacleScene::Cascade => {
            // Multi-stage cascade. Particles drop from the top, hit a
            // splitter, are funnelled inward, get whipped sideways by a
            // rotating paddle, cushioned by flanking circles, narrowed
            // again, hit a second (counter-rotating) paddle, then bounce
            // off a final big disc near the floor.
            let mut out = Vec::with_capacity(11);

            // Stage 1: splitter circle, just under the spawn line.
            out.push(Obstacle::Circle {
                center: vec2(cx, 220.0),
                radius: 55.0,
            });

            // Stage 2: V-funnel made of two static angled rects. Each arm
            // is tilted ~25° toward the centre so particles slide inward.
            let arm_angle: f32 = 0.45;
            let arm_axis_l = vec2(arm_angle.cos(), arm_angle.sin());
            let arm_axis_r = vec2(arm_angle.cos(), -arm_angle.sin());
            let arm_half = vec2(150.0, 10.0);
            out.push(Obstacle::Rect {
                center: vec2(cx - 200.0, 360.0),
                axis: arm_axis_l,
                half_extents: arm_half,
            });
            out.push(Obstacle::Rect {
                center: vec2(cx + 200.0, 360.0),
                axis: arm_axis_r,
                half_extents: arm_half,
            });

            // Stage 3: rotating paddle in the middle. Slower than the
            // historical RotatingBar so particles can interact properly.
            let ang1 = time * (BAR_ANGULAR_VEL * 0.7);
            out.push(Obstacle::Rect {
                center: vec2(cx, 520.0),
                axis: vec2(ang1.cos(), ang1.sin()),
                half_extents: vec2(170.0, 14.0),
            });

            // Stage 4: flanking circles that catch particles thrown
            // outward by the paddle and redirect them back inward.
            out.push(Obstacle::Circle {
                center: vec2(cx - 320.0, 660.0),
                radius: 60.0,
            });
            out.push(Obstacle::Circle {
                center: vec2(cx + 320.0, 660.0),
                radius: 60.0,
            });

            // Stage 5: chevron of two angled rects that narrow the flow
            // again toward the centre, this time pointing slightly
            // outward at the top so they form an inverted-V mouth.
            let chev_angle: f32 = -0.45;
            let chev_axis_l = vec2(chev_angle.cos(), chev_angle.sin());
            let chev_axis_r = vec2(chev_angle.cos(), -chev_angle.sin());
            let chev_half = vec2(140.0, 10.0);
            out.push(Obstacle::Rect {
                center: vec2(cx - 180.0, 800.0),
                axis: chev_axis_l,
                half_extents: chev_half,
            });
            out.push(Obstacle::Rect {
                center: vec2(cx + 180.0, 800.0),
                axis: chev_axis_r,
                half_extents: chev_half,
            });

            // Stage 6: second rotating paddle, counter-rotating against
            // the first (negative angular velocity, slight phase) for
            // visual contrast.
            let ang2 = -time * (BAR_ANGULAR_VEL * 0.55) + 0.9;
            out.push(Obstacle::Rect {
                center: vec2(cx, 940.0),
                axis: vec2(ang2.cos(), ang2.sin()),
                half_extents: vec2(160.0, 14.0),
            });

            // Stage 7: final big deflector near the bottom.
            out.push(Obstacle::Circle {
                center: vec2(cx, 1080.0),
                radius: 75.0,
            });

            out
        }
    }
}

/// Whether a scene needs its obstacle list rebuilt every step (rotating
/// bars, swept colliders) or whether the layout is static.
pub fn scene_is_animated(scene: ObstacleScene) -> bool {
    matches!(
        scene,
        ObstacleScene::RotatingBar | ObstacleScene::RotatingCrosses | ObstacleScene::Cascade
    )
}

/// World-space axis-aligned bounding box for an obstacle, expanded by `pad`.
/// We pad by `MAX_RADIUS` when registering into the per-bin buckets so a
/// particle's centre-bin lookup covers every obstacle it can possibly touch.
fn aabb(obs: &Obstacle, pad: f32) -> (Vec2, Vec2) {
    match *obs {
        Obstacle::Circle { center, radius } => {
            let r = radius + pad;
            (center - Vec2::splat(r), center + Vec2::splat(r))
        }
        Obstacle::Rect {
            center,
            axis,
            half_extents,
        } => {
            // World-axis projection of an OBB: `|axis_x| * hx + |perp_x| * hy`
            // gives the half-extent along world X, similarly for Y.
            let perp = vec2(-axis.y, axis.x);
            let extent = vec2(
                axis.x.abs() * half_extents.x + perp.x.abs() * half_extents.y,
                axis.y.abs() * half_extents.x + perp.y.abs() * half_extents.y,
            );
            let pad_v = Vec2::splat(pad);
            (center - extent - pad_v, center + extent + pad_v)
        }
    }
}

/// (Re)populate `bins` so each cell lists the obstacles whose padded AABB
/// covers it. Cheap: O(sum of bins covered by obstacles), which for our
/// scenes is a few thousand at most. Called once at construction (static
/// scenes) or once per step (animated scenes), well outside the hot loop.
pub fn rebuild_bins(obstacles: &[Obstacle], bins: &mut [Vec<u32>]) {
    debug_assert_eq!(bins.len(), NUM_BIN);
    for b in bins.iter_mut() {
        b.clear();
    }
    let inv_bin = 1.0 / BIN_SIZE as f32;
    let bin_w = BIN_W as i32;
    let bin_h = BIN_H as i32;
    for (i, obs) in obstacles.iter().enumerate() {
        let (lo, hi) = aabb(obs, MAX_RADIUS);
        let x0 = (lo.x * inv_bin).floor() as i32;
        let y0 = (lo.y * inv_bin).floor() as i32;
        let x1 = (hi.x * inv_bin).floor() as i32;
        let y1 = (hi.y * inv_bin).floor() as i32;
        let x0 = x0.clamp(0, bin_w - 1);
        let y0 = y0.clamp(0, bin_h - 1);
        let x1 = x1.clamp(0, bin_w - 1);
        let y1 = y1.clamp(0, bin_h - 1);
        for bx in x0..=x1 {
            let col = (bx as usize) * BIN_H;
            for by in y0..=y1 {
                bins[col + by as usize].push(i as u32);
            }
        }
    }
}

/// Resolve a single obstacle against a particle at `np` (radius `r`).
/// Returns the corrected position; called once per obstacle per particle.
pub fn resolve(np: Vec2, r: f32, obs: &Obstacle) -> Vec2 {
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
        Obstacle::Rect {
            center,
            axis,
            half_extents,
        } => {
            // OBB-vs-disc: transform the particle into rectangle-local space,
            // clamp to the box AABB to find the closest point, then push out
            // along the local-space contact normal (rotated back to world).
            let perp = vec2(-axis.y, axis.x);
            let d = np - center;
            let lx = d.dot(axis);
            let ly = d.dot(perp);
            let cx = lx.clamp(-half_extents.x, half_extents.x);
            let cy = ly.clamp(-half_extents.y, half_extents.y);
            let dx = lx - cx;
            let dy = ly - cy;
            let dist2_local = dx * dx + dy * dy;

            if dist2_local > 0.0 {
                // Particle centre is outside the box: standard push.
                if dist2_local >= r * r {
                    return np;
                }
                let dist = dist2_local.sqrt();
                let nx_local = dx / dist;
                let ny_local = dy / dist;
                let push = OBSTACLE_PUSH * (dist - r);
                let n_world = axis * nx_local + perp * ny_local;
                np - n_world * push
            } else {
                // Particle centre is inside the box: there's no contact normal
                // from the clamp, so push out along the nearest face.
                let pen_x = half_extents.x - lx.abs();
                let pen_y = half_extents.y - ly.abs();
                let (nx_local, ny_local, pen) = if pen_x < pen_y {
                    (lx.signum(), 0.0, pen_x)
                } else {
                    (0.0, ly.signum(), pen_y)
                };
                let push = OBSTACLE_PUSH * (pen + r);
                let n_world = axis * nx_local + perp * ny_local;
                np + n_world * push
            }
        }
    }
}

/// Per-instance attributes for the oriented-rect render pipeline. Layout
/// matches the vertex shader's three `vec2` attributes, packed sequentially
/// so the buffer can be uploaded as a slice of `RectInstance`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RectInstance {
    pub center: Vec2,
    pub axis: Vec2,
    pub half_extents: Vec2,
}

/// Renderable circle list. Each entry is `(center.xy, radius)` packed into
/// a `Vec3` for the existing instanced-circle pipeline. Rects are skipped —
/// they go through `rects_for` and a separate pipeline.
pub fn circles_for(obstacles: &[Obstacle]) -> Vec<Vec3> {
    obstacles
        .iter()
        .filter_map(|obs| match *obs {
            Obstacle::Circle { center, radius } => Some(center.extend(radius)),
            Obstacle::Rect { .. } => None,
        })
        .collect()
}

/// Renderable rect list, ready to upload as a `BufferSource::slice` into the
/// oriented-rect pipeline's per-instance buffer.
pub fn rects_for(obstacles: &[Obstacle]) -> Vec<RectInstance> {
    obstacles
        .iter()
        .filter_map(|obs| match *obs {
            Obstacle::Rect {
                center,
                axis,
                half_extents,
            } => Some(RectInstance {
                center,
                axis,
                half_extents,
            }),
            Obstacle::Circle { .. } => None,
        })
        .collect()
}
