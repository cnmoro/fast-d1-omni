//! GGUF v2/v3 reader: metadata and tensor data (F32, F16, BF16, Q8_0 are converted on load).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U(v) => Some(*v),
            Value::I(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F(v) => Some(*v),
            Value::U(v) => Some(*v as f64),
            Value::I(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dims: Vec<u64>, // ggml order: dims[0] is the contiguous (innermost) dimension
    pub ty: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
}

pub struct Gguf {
    pub path: String,
    pub meta: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    data_start: u64,
}

fn rd<const N: usize>(r: &mut impl Read) -> std::io::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}
fn u32_(r: &mut impl Read) -> std::io::Result<u32> {
    Ok(u32::from_le_bytes(rd::<4>(r)?))
}
fn u64_(r: &mut impl Read) -> std::io::Result<u64> {
    Ok(u64::from_le_bytes(rd::<8>(r)?))
}
fn string(r: &mut impl Read) -> std::io::Result<String> {
    let n = u64_(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn value(r: &mut impl Read, t: u32) -> std::io::Result<Value> {
    Ok(match t {
        0 => Value::U(rd::<1>(r)?[0] as u64),
        1 => Value::I(rd::<1>(r)?[0] as i8 as i64),
        2 => Value::U(u16::from_le_bytes(rd::<2>(r)?) as u64),
        3 => Value::I(i16::from_le_bytes(rd::<2>(r)?) as i64),
        4 => Value::U(u32_(r)? as u64),
        5 => Value::I(u32_(r)? as i32 as i64),
        6 => Value::F(f32::from_le_bytes(rd::<4>(r)?) as f64),
        7 => Value::Bool(rd::<1>(r)?[0] != 0),
        8 => Value::Str(string(r)?),
        9 => {
            let et = u32_(r)?;
            let n = u64_(r)? as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(value(r, et)?);
            }
            Value::Arr(v)
        }
        10 => Value::U(u64_(r)?),
        11 => Value::I(u64_(r)? as i64),
        12 => Value::F(f64::from_le_bytes(rd::<8>(r)?)),
        _ => return Err(std::io::Error::other(format!("bad gguf value type {t}"))),
    })
}

impl Gguf {
    pub fn open(path: &str) -> Result<Gguf, String> {
        let f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut r = BufReader::with_capacity(1 << 20, f);
        let io = |e: std::io::Error| format!("{path}: {e}");
        let magic = rd::<4>(&mut r).map_err(io)?;
        if &magic != b"GGUF" {
            return Err(format!("{path}: not a GGUF file"));
        }
        let ver = u32_(&mut r).map_err(io)?;
        if ver < 2 {
            return Err(format!("{path}: GGUF v{ver} unsupported"));
        }
        let nt = u64_(&mut r).map_err(io)?;
        let nkv = u64_(&mut r).map_err(io)?;
        let mut meta = HashMap::new();
        for _ in 0..nkv {
            let k = string(&mut r).map_err(io)?;
            let t = u32_(&mut r).map_err(io)?;
            meta.insert(k, value(&mut r, t).map_err(io)?);
        }
        let mut tensors = HashMap::new();
        for _ in 0..nt {
            let name = string(&mut r).map_err(io)?;
            let nd = u32_(&mut r).map_err(io)?;
            let mut dims = Vec::new();
            for _ in 0..nd {
                dims.push(u64_(&mut r).map_err(io)?);
            }
            let ty = u32_(&mut r).map_err(io)?;
            let offset = u64_(&mut r).map_err(io)?;
            tensors.insert(name, TensorInfo { dims, ty, offset });
        }
        let pos = r.stream_position().map_err(io)?;
        let align = meta.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32);
        let data_start = pos.div_ceil(align) * align;
        Ok(Gguf { path: path.to_string(), meta, tensors, data_start })
    }

    pub fn get(&self, k: &str) -> Option<&Value> {
        self.meta.get(k)
    }
    pub fn u(&self, k: &str) -> Option<u64> {
        self.meta.get(k).and_then(|v| v.as_u64())
    }
    pub fn f(&self, k: &str) -> Option<f64> {
        self.meta.get(k).and_then(|v| v.as_f64())
    }
    pub fn s(&self, k: &str) -> Option<&str> {
        self.meta.get(k).and_then(|v| v.as_str())
    }
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors.get(name).ok_or_else(|| format!("{}: missing tensor {name}", self.path))
    }

    fn raw(&self, name: &str) -> Result<(TensorInfo, Vec<u8>), String> {
        let ti = self.info(name)?.clone();
        let n = ti.numel();
        let bytes = match ti.ty {
            0 => n * 4,
            1 | 30 => n * 2,
            8 => n / 32 * 34,
            t => return Err(format!("{name}: unsupported ggml type {t}")),
        };
        let mut f = File::open(&self.path).map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(self.data_start + ti.offset)).map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; bytes];
        f.read_exact(&mut buf).map_err(|e| format!("{name}: {e}"))?;
        Ok((ti, buf))
    }

    /// Tensor as f32, in row-major order with ggml dims reversed (PyTorch order).
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let (ti, b) = self.raw(name)?;
        Ok(match ti.ty {
            0 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            1 => b.chunks_exact(2).map(|c| crate::cuda::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            30 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            8 => {
                let mut out = Vec::with_capacity(ti.numel());
                for blk in b.chunks_exact(34) {
                    let d = crate::cuda::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    for &q in &blk[2..] {
                        out.push(q as i8 as f32 * d);
                    }
                }
                out
            }
            _ => unreachable!(),
        })
    }

    /// Tensor as fp16 bits (F16 is passed through untouched; other types are converted).
    pub fn f16(&self, name: &str) -> Result<Vec<u16>, String> {
        if self.info(name)?.ty == 1 {
            let (_, b) = self.raw(name)?;
            return Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect());
        }
        Ok(crate::cuda::to_f16_vec(&self.f32(name)?))
    }

    pub fn dims(&self, name: &str) -> Result<Vec<usize>, String> {
        Ok(self.info(name)?.dims.iter().rev().map(|&d| d as usize).collect())
    }
}
