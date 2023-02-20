use miniquad::*;

pub const VERTEX: &str = r#"#version 100
  attribute vec2 geom;
  attribute vec2 pos;
  attribute vec3 color0;

  varying lowp vec4 color;

  uniform mat4 mvp;

  void main() {
      vec4 rpos = vec4(geom * 2.0 + pos, 0.0, 1.0);
      gl_Position = mvp * rpos;
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
