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

/// Oriented-rect vertex shader. Takes a unit rectangle vertex (spine,
/// normal) ∈ [-1,1]² and maps it through the per-instance frame:
///   world = center + spine * half.x * axis + normal * half.y * perp
/// where `perp = (-axis.y, axis.x)`. One instance = one rotated rectangle.
pub const VERTEX_RECT: &str = r#"#version 100
  attribute vec2 pos;
  attribute vec2 inst_center;
  attribute vec2 inst_axis;
  attribute vec2 inst_half;
  attribute vec3 color0;

  varying lowp vec4 color;

  uniform mat4 mvp;

  void main() {
      vec2 perp = vec2(-inst_axis.y, inst_axis.x);
      vec2 world =
          inst_center
        + inst_axis * (pos.x * inst_half.x)
        + perp * (pos.y * inst_half.y);
      gl_Position = mvp * vec4(world, 0.0, 1.0);
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
