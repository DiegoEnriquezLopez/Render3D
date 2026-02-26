export type Vec3 = [number, number, number];
export type Mat4 = Float32Array; // column-major

export const vec3 = {
  add(a: Vec3, b: Vec3): Vec3 {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  },
  sub(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  },
  scale(v: Vec3, s: number): Vec3 {
    return [v[0] * s, v[1] * s, v[2] * s];
  },
  dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  },
  cross(a: Vec3, b: Vec3): Vec3 {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  },
  normalize(v: Vec3): Vec3 {
    const len = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / len, v[1] / len, v[2] / len];
  },
};

export const mat4 = {
  identity(): Mat4 {
    const m = new Float32Array(16);
    m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
    return m;
  },

  // out = a * b (column-major)
  multiply(a: Mat4, b: Mat4): Mat4 {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        out[c * 4 + r] =
          a[0 * 4 + r] * b[c * 4 + 0] +
          a[1 * 4 + r] * b[c * 4 + 1] +
          a[2 * 4 + r] * b[c * 4 + 2] +
          a[3 * 4 + r] * b[c * 4 + 3];
      }
    }
    return out;
  },

  translation(tx: number, ty: number, tz: number): Mat4 {
    const m = mat4.identity();
    m[12] = tx;
    m[13] = ty;
    m[14] = tz;
    return m;
  },

  scaling(sx: number, sy: number, sz: number): Mat4 {
    const m = mat4.identity();
    m[0] = sx;
    m[5] = sy;
    m[10] = sz;
    return m;
  },

  rotationX(rad: number): Mat4 {
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    const m = mat4.identity();

    m[5] = c;
    m[6] = s;
    m[9] = -s;
    m[10] = c;

    return m;
  },

  rotationY(rad: number): Mat4 {
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    const m = mat4.identity();

    m[0] = c;
    m[2] = -s;
    m[8] = s;
    m[10] = c;

    return m;
  },

  rotationZ(rad: number): Mat4 {
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    const m = mat4.identity();

    m[0] = c;
    m[1] = s;
    m[4] = -s;
    m[5] = c;

    return m;
  },

  perspective(fovyRad: number, aspect: number, near: number, far: number): Mat4 {
    const f = 1.0 / Math.tan(fovyRad / 2);
    const m = new Float32Array(16);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = far / (near - far);
    m[11] = -1;
    m[14] = (far * near) / (near - far);
    return m;
  },

  lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
    const z = vec3.normalize(vec3.sub(eye, target));
    const x = vec3.normalize(vec3.cross(up, z));
    const y = vec3.cross(z, x);

    const m = new Float32Array(16);
    m[0] = x[0]; m[4] = x[1]; m[8]  = x[2]; m[12] = -vec3.dot(x, eye);
    m[1] = y[0]; m[5] = y[1]; m[9]  = y[2]; m[13] = -vec3.dot(y, eye);
    m[2] = z[0]; m[6] = z[1]; m[10] = z[2]; m[14] = -vec3.dot(z, eye);
    m[3] = 0;    m[7] = 0;    m[11] = 0;    m[15] = 1;
    return m;
  },
};
