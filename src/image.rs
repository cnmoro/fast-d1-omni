//! Image decoding (JPEG baseline/progressive, PNG, BMP, PPM) to RGB8, EXIF orientation, and torchvision-style
//! antialiased bilinear resizing. JPEG decoding follows libjpeg(-turbo) arithmetic (islow IDCT, fancy upsampling,
//! fixed-point YCbCr), which is what PIL uses.

pub struct Image {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>, // HWC
}

const MAX_PIXELS: usize = 100_000_000;

pub fn decode(data: &[u8]) -> Result<Image, String> {
    if data.len() > 3 && data[0] == 0xff && data[1] == 0xd8 {
        let (img, orient) = jpeg::decode(data)?;
        Ok(apply_orientation(img, orient))
    } else if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        png::decode(data)
    } else if data.starts_with(b"BM") {
        bmp(data)
    } else if data.len() > 2 && data[0] == b'P' && (data[1] == b'6' || data[1] == b'5') {
        ppm(data)
    } else if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WEBP" {
        Err("WebP images are not supported (use JPEG or PNG)".into())
    } else if data.starts_with(b"GIF8") {
        Err("GIF images are not supported (use JPEG or PNG)".into())
    } else {
        Err("unrecognised image format (supported: JPEG, PNG, BMP, PPM)".into())
    }
}

/// EXIF orientation 1..8 -> upright image (what PIL.ImageOps.exif_transpose does).
fn apply_orientation(img: Image, o: u16) -> Image {
    if !(2..=8).contains(&o) {
        return img;
    }
    let (w, h) = (img.w, img.h);
    let swap = o >= 5;
    let (nw, nh) = if swap { (h, w) } else { (w, h) };
    let mut out = vec![0u8; nw * nh * 3];
    for y in 0..nh {
        for x in 0..nw {
            let (sx, sy) = match o {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (y, h - 1 - x),
                7 => (w - 1 - y, h - 1 - x),
                8 => (w - 1 - y, x),
                _ => (x, y),
            };
            let s = (sy * w + sx) * 3;
            let d = (y * nw + x) * 3;
            out[d..d + 3].copy_from_slice(&img.rgb[s..s + 3]);
        }
    }
    Image { w: nw, h: nh, rgb: out }
}

fn ppm(data: &[u8]) -> Result<Image, String> {
    let mut fields = Vec::new();
    let mut i = 2;
    while fields.len() < 3 && i < data.len() {
        while i < data.len() && (data[i].is_ascii_whitespace() || data[i] == b'#') {
            if data[i] == b'#' {
                while i < data.len() && data[i] != b'\n' {
                    i += 1;
                }
            }
            i += 1;
        }
        let s = i;
        while i < data.len() && data[i].is_ascii_digit() {
            i += 1;
        }
        fields.push(std::str::from_utf8(&data[s..i]).ok().and_then(|x| x.parse::<usize>().ok()).ok_or("bad PPM header")?);
    }
    i += 1;
    let (w, h, maxv) = (fields[0], fields[1], fields[2]);
    if maxv != 255 || w * h > MAX_PIXELS {
        return Err("unsupported PPM".into());
    }
    let ch = if data[1] == b'6' { 3 } else { 1 };
    let px = data.get(i..i + w * h * ch).ok_or("truncated PPM")?;
    let rgb = if ch == 3 { px.to_vec() } else { px.iter().flat_map(|&g| [g, g, g]).collect() };
    Ok(Image { w, h, rgb })
}

fn bmp(d: &[u8]) -> Result<Image, String> {
    let u32_ = |o: usize| d.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or("truncated BMP");
    let u16_ = |o: usize| d.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]])).ok_or("truncated BMP");
    let off = u32_(10)? as usize;
    let w = u32_(18)? as i32;
    let hh = u32_(22)? as i32;
    let bpp = u16_(28)?;
    let comp = u32_(30)?;
    if !(bpp == 24 || bpp == 32) || !(comp == 0 || comp == 3) || w <= 0 {
        return Err("unsupported BMP (only uncompressed 24/32-bit)".into());
    }
    let (w, h, flip) = (w as usize, hh.unsigned_abs() as usize, hh > 0);
    if w * h > MAX_PIXELS {
        return Err("image too large".into());
    }
    let bpp = bpp as usize / 8;
    let stride = (w * bpp).div_ceil(4) * 4;
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        let sy = if flip { h - 1 - y } else { y };
        let row = d.get(off + sy * stride..off + sy * stride + w * bpp).ok_or("truncated BMP")?;
        for x in 0..w {
            rgb[(y * w + x) * 3] = row[x * bpp + 2];
            rgb[(y * w + x) * 3 + 1] = row[x * bpp + 1];
            rgb[(y * w + x) * 3 + 2] = row[x * bpp];
        }
    }
    Ok(Image { w, h, rgb })
}

// ================================================================================================ resize
// torch's uint8 antialiased bilinear (separable, horizontal then vertical, int16 fixed-point weights).

struct Taps {
    start: Vec<usize>,
    len: Vec<usize>,
    w: Vec<i32>, // [out][ksize]
    ksize: usize,
    prec: u32,
}

fn taps(inp: usize, out: usize) -> Taps {
    let scale = inp as f64 / out as f64;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let ksize = (support.ceil() as usize) * 2 + 1;
    let mut start = vec![0; out];
    let mut len = vec![0; out];
    let mut wf = vec![0f64; out * ksize];
    let mut wmax = 0f64;
    for i in 0..out {
        let center = scale * (i as f64 + 0.5);
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(inp as i64) as usize;
        let n = xmax - xmin;
        let mut total = 0.0;
        for j in 0..n {
            let x = (j as f64 + xmin as f64 - center + 0.5) * invscale;
            let v = if x.abs() < 1.0 { 1.0 - x.abs() } else { 0.0 };
            wf[i * ksize + j] = v;
            total += v;
        }
        for j in 0..n {
            if total != 0.0 {
                wf[i * ksize + j] /= total;
            }
            wmax = wmax.max(wf[i * ksize + j]);
        }
        start[i] = xmin;
        len[i] = n;
    }
    let mut prec = 0u32;
    while prec < 22 {
        let next = (0.5 + wmax * (1u64 << (prec + 1)) as f64) as i64;
        if next >= 1 << 15 {
            break;
        }
        prec += 1;
    }
    let w = wf
        .iter()
        .map(|&v| {
            let s = v * (1u64 << prec) as f64;
            if v < 0.0 { (-0.5 + s) as i32 } else { (0.5 + s) as i32 }
        })
        .collect();
    Taps { start, len, w, ksize, prec }
}

fn par_rows<F: Fn(usize, &mut [u8]) + Sync>(out: &mut [u8], row_len: usize, rows: usize, f: F) {
    let threads = if rows * row_len > 1 << 20 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8) } else { 1 };
    if threads <= 1 {
        for (y, r) in out.chunks_mut(row_len).enumerate() {
            f(y, r);
        }
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|sc| {
        for (i, chunk) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            sc.spawn(move || {
                for (j, r) in chunk.chunks_mut(row_len).enumerate() {
                    f(i * per + j, r);
                }
            });
        }
    });
}

/// Resize an RGB8 image to (oh, ow) like `torchvision.transforms.v2.functional.resize(antialias=True)` on uint8.
pub fn resize(img: &Image, oh: usize, ow: usize) -> Image {
    let mut cur_w = img.w;
    let cur_h = img.h;
    let mut buf: std::borrow::Cow<[u8]> = std::borrow::Cow::Borrowed(&img.rgb);
    if ow != img.w {
        let t = taps(img.w, ow);
        let mut out = vec![0u8; ow * cur_h * 3];
        let half = if t.prec > 0 { 1i32 << (t.prec - 1) } else { 0 };
        let src: &[u8] = &buf;
        par_rows(&mut out, ow * 3, cur_h, |y, orow| {
            let row = &src[y * cur_w * 3..(y + 1) * cur_w * 3];
            for x in 0..ow {
                let (s, n) = (t.start[x], t.len[x]);
                let ww = &t.w[x * t.ksize..x * t.ksize + n];
                let px = &row[s * 3..(s + n) * 3];
                let (mut r, mut g, mut b) = (half, half, half);
                for (w, p) in ww.iter().zip(px.chunks_exact(3)) {
                    r += w * p[0] as i32;
                    g += w * p[1] as i32;
                    b += w * p[2] as i32;
                }
                orow[x * 3] = (r >> t.prec).clamp(0, 255) as u8;
                orow[x * 3 + 1] = (g >> t.prec).clamp(0, 255) as u8;
                orow[x * 3 + 2] = (b >> t.prec).clamp(0, 255) as u8;
            }
        });
        buf = std::borrow::Cow::Owned(out);
        cur_w = ow;
    }
    if oh != cur_h {
        let t = taps(cur_h, oh);
        let rs = cur_w * 3;
        let mut out = vec![0u8; rs * oh];
        let half = if t.prec > 0 { 1i32 << (t.prec - 1) } else { 0 };
        let src: &[u8] = &buf;
        par_rows(&mut out, rs, oh, |y, orow| {
            let (s, n) = (t.start[y], t.len[y]);
            let ww = &t.w[y * t.ksize..y * t.ksize + n];
            let mut acc = vec![half; rs];
            for (j, &w) in ww.iter().enumerate() {
                let r = &src[(s + j) * rs..(s + j + 1) * rs];
                for (a, &v) in acc.iter_mut().zip(r) {
                    *a += w * v as i32;
                }
            }
            for (o, a) in orow.iter_mut().zip(&acc) {
                *o = (a >> t.prec).clamp(0, 255) as u8;
            }
        });
        buf = std::borrow::Cow::Owned(out);
    }
    Image { w: ow, h: oh, rgb: buf.into_owned() }
}

// ================================================================================================ JPEG
pub mod jpeg {
    use super::{Image, MAX_PIXELS};

    const ZZ: [usize; 64] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
        28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
        54, 47, 55, 62, 63,
    ];

    #[derive(Clone, Default)]
    struct Huff {
        maxcode: [i32; 18],
        valptr: [i32; 17],
        mincode: [i32; 17],
        vals: Vec<u8>,
        look: Vec<(u8, u8)>, // 9-bit lookahead: (len, value), len 0 = slow path
    }

    impl Huff {
        fn new(counts: &[u8; 16], vals: Vec<u8>) -> Huff {
            let mut h = Huff { vals, look: vec![(0, 0); 512], ..Default::default() };
            let mut code = 0i32;
            let mut k = 0i32;
            for l in 1..=16 {
                let n = counts[l - 1] as i32;
                if n > 0 {
                    h.valptr[l] = k;
                    h.mincode[l] = code;
                    code += n;
                    k += n;
                    h.maxcode[l] = code - 1;
                } else {
                    h.maxcode[l] = -1;
                }
                code <<= 1;
            }
            h.maxcode[17] = i32::MAX;
            // lookahead table
            let mut code = 0u32;
            let mut k = 0usize;
            for l in 1..=9u32 {
                for _ in 0..counts[l as usize - 1] {
                    let shift = 9 - l;
                    for f in 0..(1u32 << shift) {
                        let idx = ((code << shift) | f) as usize;
                        if k < h.vals.len() {
                            h.look[idx] = (l as u8, h.vals[k]);
                        }
                    }
                    code += 1;
                    k += 1;
                }
                code <<= 1;
            }
            h
        }
    }

    struct Bits<'a> {
        d: &'a [u8],
        pos: usize,
        acc: u64,
        n: u32,
        marker_hit: bool,
    }

    impl<'a> Bits<'a> {
        fn fill(&mut self) {
            while self.n <= 56 {
                let mut b = 0u8;
                if !self.marker_hit && self.pos < self.d.len() {
                    b = self.d[self.pos];
                    if b == 0xff {
                        let nx = self.d.get(self.pos + 1).copied().unwrap_or(0);
                        if nx == 0 {
                            self.pos += 2;
                        } else {
                            self.marker_hit = true;
                            b = 0;
                        }
                    } else {
                        self.pos += 1;
                    }
                }
                self.acc |= (b as u64) << (56 - self.n);
                self.n += 8;
            }
        }
        #[inline]
        fn bits(&mut self, k: u32) -> u32 {
            if k == 0 {
                return 0;
            }
            if self.n < k {
                self.fill();
            }
            let v = (self.acc >> (64 - k)) as u32;
            self.acc <<= k;
            self.n -= k;
            v
        }
        #[inline]
        fn bit(&mut self) -> u32 {
            self.bits(1)
        }
        fn peek9(&mut self) -> u32 {
            if self.n < 9 {
                self.fill();
            }
            (self.acc >> 55) as u32
        }
        fn decode(&mut self, h: &Huff) -> Result<u8, String> {
            let p = self.peek9();
            let (l, v) = h.look[p as usize];
            if l > 0 {
                self.acc <<= l;
                self.n -= l as u32;
                return Ok(v);
            }
            let mut code = self.bits(1) as i32;
            let mut l = 1;
            while code > h.maxcode[l] {
                code = (code << 1) | self.bits(1) as i32;
                l += 1;
                if l > 16 {
                    return Err("corrupt JPEG huffman data".into());
                }
            }
            let idx = h.valptr[l] + code - h.mincode[l];
            h.vals.get(idx as usize).copied().ok_or_else(|| "corrupt JPEG huffman data".into())
        }
        fn receive_extend(&mut self, s: u32) -> i32 {
            if s == 0 {
                return 0;
            }
            let v = self.bits(s) as i32;
            if v < (1 << (s - 1)) {
                v - (1 << s) + 1
            } else {
                v
            }
        }
        /// Skip to the restart marker, consume it.
        fn restart(&mut self) {
            self.acc = 0;
            self.n = 0;
            self.marker_hit = false;
            while self.pos + 1 < self.d.len() {
                if self.d[self.pos] == 0xff && (0xd0..=0xd7).contains(&self.d[self.pos + 1]) {
                    self.pos += 2;
                    return;
                }
                self.pos += 1;
            }
        }
    }

    #[derive(Clone, Default)]
    struct Comp {
        id: u8,
        h: usize,
        v: usize,
        tq: usize,
        bw: usize, // blocks per row (padded to MCU)
        bh: usize,
        cw: usize, // component width in samples (unpadded)
        ch: usize,
        coef: Vec<i16>,
        dc_pred: i32,
        td: usize,
        ta: usize,
    }

    pub fn decode(d: &[u8]) -> Result<(Image, u16), String> {
        let mut qt = [[0u16; 64]; 4];
        let mut dc_tabs: Vec<Option<Huff>> = vec![None; 4];
        let mut ac_tabs: Vec<Option<Huff>> = vec![None; 4];
        let mut comps: Vec<Comp> = Vec::new();
        let (mut w, mut h) = (0usize, 0usize);
        let (mut hmax, mut vmax) = (1, 1);
        let mut progressive = false;
        let mut restart_interval = 0usize;
        let mut adobe_transform: Option<u8> = None;
        let mut jfif = false;
        let mut orientation = 1u16;
        let mut eobrun: u32;
        let mut pos = 2;
        let mut frame_seen = false;
        loop {
            while pos < d.len() && d[pos] != 0xff {
                pos += 1;
            }
            while pos < d.len() && d[pos] == 0xff {
                pos += 1;
            }
            if pos >= d.len() {
                break;
            }
            let marker = d[pos];
            pos += 1;
            if marker == 0xd9 {
                break;
            }
            if (0xd0..=0xd7).contains(&marker) || marker == 0x01 {
                continue;
            }
            let len = d.get(pos..pos + 2).map(|b| u16::from_be_bytes([b[0], b[1]]) as usize).ok_or("truncated JPEG")?;
            let seg = d.get(pos + 2..pos + len).ok_or("truncated JPEG segment")?;
            pos += len;
            match marker {
                0xdb => {
                    let mut i = 0;
                    while i < seg.len() {
                        let pq = seg[i] >> 4;
                        let tq = (seg[i] & 15) as usize & 3;
                        i += 1;
                        for k in 0..64 {
                            let v = if pq == 0 {
                                let v = *seg.get(i).ok_or("bad DQT")? as u16;
                                i += 1;
                                v
                            } else {
                                let v = u16::from_be_bytes([*seg.get(i).ok_or("bad DQT")?, *seg.get(i + 1).ok_or("bad DQT")?]);
                                i += 2;
                                v
                            };
                            qt[tq][ZZ[k]] = v;
                        }
                    }
                }
                0xc4 => {
                    let mut i = 0;
                    while i + 17 <= seg.len() {
                        let tc = seg[i] >> 4;
                        let th = (seg[i] & 15) as usize & 3;
                        let mut counts = [0u8; 16];
                        counts.copy_from_slice(&seg[i + 1..i + 17]);
                        let n: usize = counts.iter().map(|&c| c as usize).sum();
                        let vals = seg.get(i + 17..i + 17 + n).ok_or("bad DHT")?.to_vec();
                        i += 17 + n;
                        let t = Huff::new(&counts, vals);
                        if tc == 0 {
                            dc_tabs[th] = Some(t);
                        } else {
                            ac_tabs[th] = Some(t);
                        }
                    }
                }
                0xc0 | 0xc1 | 0xc2 => {
                    if frame_seen {
                        return Err("multiple JPEG frames".into());
                    }
                    frame_seen = true;
                    progressive = marker == 0xc2;
                    if seg.len() < 6 || seg[0] != 8 {
                        return Err("only 8-bit JPEG is supported".into());
                    }
                    h = u16::from_be_bytes([seg[1], seg[2]]) as usize;
                    w = u16::from_be_bytes([seg[3], seg[4]]) as usize;
                    let nc = seg[5] as usize;
                    if w == 0 || h == 0 || w * h > MAX_PIXELS {
                        return Err(format!("bad JPEG size {w}x{h}"));
                    }
                    if !(nc == 1 || nc == 3) {
                        return Err(format!("JPEG with {nc} components is not supported"));
                    }
                    for c in 0..nc {
                        let b = seg.get(6 + c * 3..9 + c * 3).ok_or("bad SOF")?;
                        let (hh, vv) = ((b[1] >> 4) as usize, (b[1] & 15) as usize);
                        if !(1..=4).contains(&hh) || !(1..=4).contains(&vv) {
                            return Err("bad sampling factors".into());
                        }
                        comps.push(Comp { id: b[0], h: hh, v: vv, tq: (b[2] & 3) as usize, ..Default::default() });
                    }
                    hmax = comps.iter().map(|c| c.h).max().unwrap();
                    vmax = comps.iter().map(|c| c.v).max().unwrap();
                    let mcux = w.div_ceil(8 * hmax);
                    let mcuy = h.div_ceil(8 * vmax);
                    for c in comps.iter_mut() {
                        c.bw = mcux * c.h;
                        c.bh = mcuy * c.v;
                        c.cw = (w * c.h).div_ceil(hmax);
                        c.ch = (h * c.v).div_ceil(vmax);
                        c.coef = vec![0i16; c.bw * c.bh * 64];
                    }
                }
                0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf => {
                    return Err("lossless/hierarchical/arithmetic JPEG is not supported".into());
                }
                0xdd => {
                    restart_interval = u16::from_be_bytes([seg[0], seg[1]]) as usize;
                }
                0xe0 => {
                    if seg.starts_with(b"JFIF\0") {
                        jfif = true;
                    }
                }
                0xe1 => {
                    if seg.starts_with(b"Exif\0\0") {
                        orientation = exif_orientation(&seg[6..]).unwrap_or(1);
                    }
                }
                0xee => {
                    if seg.starts_with(b"Adobe") && seg.len() >= 12 {
                        adobe_transform = Some(seg[11]);
                    }
                }
                0xda => {
                    if !frame_seen {
                        return Err("JPEG scan before frame".into());
                    }
                    let ns = seg[0] as usize;
                    let mut sc = Vec::new();
                    for i in 0..ns {
                        let cid = seg[1 + i * 2];
                        let t = seg[2 + i * 2];
                        let ci = comps.iter().position(|c| c.id == cid).ok_or("bad scan component")?;
                        comps[ci].td = (t >> 4) as usize & 3;
                        comps[ci].ta = (t & 15) as usize & 3;
                        sc.push(ci);
                    }
                    let p = 1 + ns * 2;
                    let (ss, se, ah, al) = (seg[p] as usize, seg[p + 1] as usize, seg[p + 2] >> 4, seg[p + 2] & 15);
                    let mut br = Bits { d, pos, acc: 0, n: 0, marker_hit: false };
                    for &ci in &sc {
                        comps[ci].dc_pred = 0;
                    }
                    eobrun = 0;
                    let single = sc.len() == 1;
                    let (units_x, units_y) = if single {
                        let c = &comps[sc[0]];
                        (c.cw.div_ceil(8), c.ch.div_ceil(8))
                    } else {
                        (w.div_ceil(8 * hmax), h.div_ceil(8 * vmax))
                    };
                    let mut todo = restart_interval;
                    for uy in 0..units_y {
                        for ux in 0..units_x {
                            if restart_interval > 0 {
                                if todo == 0 {
                                    br.restart();
                                    for &ci in &sc {
                                        comps[ci].dc_pred = 0;
                                    }
                                    eobrun = 0;
                                    todo = restart_interval;
                                }
                                todo -= 1;
                            }
                            if single {
                                let c = &mut comps[sc[0]];
                                let off = (uy * c.bw + ux) * 64;
                                decode_block(&mut br, c, off, progressive, ss, se, ah, al, &dc_tabs, &ac_tabs, &mut eobrun)?;
                            } else {
                                for &ci in &sc {
                                    let c = &mut comps[ci];
                                    for by in 0..c.v {
                                        for bx in 0..c.h {
                                            let off = ((uy * c.v + by) * c.bw + ux * c.h + bx) * 64;
                                            decode_block(&mut br, c, off, progressive, ss, se, ah, al, &dc_tabs, &ac_tabs, &mut eobrun)?;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // continue after the entropy-coded data
                    pos = br.pos;
                    while pos + 1 < d.len() && !(d[pos] == 0xff && d[pos + 1] != 0 && !(0xd0..=0xd7).contains(&d[pos + 1])) {
                        pos += 1;
                    }
                }
                _ => {}
            }
        }
        if comps.is_empty() {
            return Err("JPEG has no frame".into());
        }
        let tm = std::env::var("D1_JPEG_TIMING").is_ok();
        let t_idct = std::time::Instant::now();
        // IDCT every component into a plane of bw*8 x bh*8 samples (block rows in parallel for large images)
        let mut planes: Vec<Vec<u8>> = Vec::new();
        for c in &comps {
            let pw = c.bw * 8;
            let mut plane = vec![0u8; pw * c.bh * 8];
            let q = &qt[c.tq];
            let row_bytes = pw * 8;
            let work = |by: usize, dst: &mut [u8]| {
                for bx in 0..c.bw {
                    let blk: &[i16; 64] = c.coef[(by * c.bw + bx) * 64..(by * c.bw + bx + 1) * 64].try_into().unwrap();
                    idct_islow(blk, q, &mut dst[bx * 8..], pw);
                }
            };
            let threads = if w * h > 1 << 20 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8) } else { 1 };
            if threads > 1 {
                let per = c.bh.div_ceil(threads);
                std::thread::scope(|sc| {
                    for (i, chunk) in plane.chunks_mut(per * row_bytes).enumerate() {
                        let work = &work;
                        sc.spawn(move || {
                            for (j, rows) in chunk.chunks_mut(row_bytes).enumerate() {
                                work(i * per + j, rows);
                            }
                        });
                    }
                });
            } else {
                for (by, rows) in plane.chunks_mut(row_bytes).enumerate() {
                    work(by, rows);
                }
            }
            planes.push(plane);
        }
        let t_up = std::time::Instant::now();
        // upsample to full resolution
        let mut full: Vec<Vec<u8>> = Vec::new();
        for (c, p) in comps.iter().zip(&planes) {
            let pw = c.bw * 8;
            let (fx, fy) = (hmax / c.h, vmax / c.v);
            if fx == 1 && fy == 1 {
                let mut o = vec![0u8; w * h];
                for y in 0..h {
                    o[y * w..(y + 1) * w].copy_from_slice(&p[y * pw..y * pw + w]);
                }
                full.push(o);
            } else if fx == 2 && fy == 1 {
                full.push(h2v1_fancy(p, pw, c.cw, c.ch, w, h));
            } else if fx == 2 && fy == 2 {
                full.push(h2v2_fancy(p, pw, c.cw, c.ch, w, h));
            } else {
                let mut o = vec![0u8; w * h];
                for y in 0..h {
                    for x in 0..w {
                        o[y * w + x] = p[(y / fy) * pw + x / fx];
                    }
                }
                full.push(o);
            }
        }
        let t_col = std::time::Instant::now();
        let mut rgb = vec![0u8; w * h * 3];
        if comps.len() == 1 {
            for i in 0..w * h {
                let g = full[0][i];
                rgb[i * 3] = g;
                rgb[i * 3 + 1] = g;
                rgb[i * 3 + 2] = g;
            }
        } else {
            let is_rgb = match adobe_transform {
                Some(t) => t == 0 && !jfif,
                None => !jfif && comps[0].id == b'R' && comps[1].id == b'G' && comps[2].id == b'B',
            };
            if is_rgb {
                for i in 0..w * h {
                    rgb[i * 3] = full[0][i];
                    rgb[i * 3 + 1] = full[1][i];
                    rgb[i * 3 + 2] = full[2][i];
                }
            } else {
                let t = ColorTables::new();
                let (f0, f1, f2) = (&full[0], &full[1], &full[2]);
                super::par_rows(&mut rgb, w * 3, h, |yy, row| {
                    let r = yy * w..(yy + 1) * w;
                    for (((o, &y), &cb), &cr) in row.chunks_exact_mut(3).zip(&f0[r.clone()]).zip(&f1[r.clone()]).zip(&f2[r]) {
                        let (y, cb, cr) = (y as i32, cb as usize, cr as usize);
                        o[0] = (y + t.cr_r[cr]).clamp(0, 255) as u8;
                        o[1] = (y + ((t.cb_g[cb] + t.cr_g[cr]) >> 16)).clamp(0, 255) as u8;
                        o[2] = (y + t.cb_b[cb]).clamp(0, 255) as u8;
                    }
                });
            }
        }
        if tm {
            eprintln!("jpeg: idct {:.2} up {:.2} color {:.2} ms", (t_up - t_idct).as_secs_f64() * 1e3, (t_col - t_up).as_secs_f64() * 1e3, t_col.elapsed().as_secs_f64() * 1e3);
        }
        Ok((Image { w, h, rgb }, orientation))
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_block(br: &mut Bits, c: &mut Comp, off: usize, progressive: bool, ss: usize, se: usize, ah: u8, al: u8,
                    dc_tabs: &[Option<Huff>], ac_tabs: &[Option<Huff>], eobrun: &mut u32) -> Result<(), String> {
        let blk = &mut c.coef[off..off + 64];
        if !progressive {
            let dct = dc_tabs[c.td].as_ref().ok_or("missing DC table")?;
            let act = ac_tabs[c.ta].as_ref().ok_or("missing AC table")?;
            let t = br.decode(dct)? as u32;
            let diff = br.receive_extend(t);
            c.dc_pred += diff;
            blk[0] = c.dc_pred as i16;
            let mut k = 1;
            while k < 64 {
                let rs = br.decode(act)?;
                let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
                if s != 0 {
                    k += r;
                    if k > 63 {
                        break;
                    }
                    blk[ZZ[k]] = br.receive_extend(s) as i16;
                    k += 1;
                } else if r == 15 {
                    k += 16;
                } else {
                    break;
                }
            }
            return Ok(());
        }
        if ss == 0 {
            // DC scan
            if ah == 0 {
                let dct = dc_tabs[c.td].as_ref().ok_or("missing DC table")?;
                let t = br.decode(dct)? as u32;
                let diff = br.receive_extend(t);
                c.dc_pred += diff;
                blk[0] = (c.dc_pred << al) as i16;
            } else if br.bit() != 0 {
                blk[0] |= (1 << al) as i16;
            }
            return Ok(());
        }
        let act = ac_tabs[c.ta].as_ref().ok_or("missing AC table")?;
        if ah == 0 {
            if *eobrun > 0 {
                *eobrun -= 1;
                return Ok(());
            }
            let mut k = ss;
            while k <= se {
                let rs = br.decode(act)?;
                let (r, s) = ((rs >> 4) as u32, (rs & 15) as u32);
                if s != 0 {
                    k += r as usize;
                    if k > 63 {
                        break;
                    }
                    blk[ZZ[k]] = (br.receive_extend(s) * (1 << al)) as i16;
                } else if r < 15 {
                    *eobrun = (1 << r) - 1;
                    if r > 0 {
                        *eobrun += br.bits(r);
                    }
                    break;
                } else {
                    k += 15;
                }
                k += 1;
            }
            return Ok(());
        }
        // AC refinement
        let p1 = 1i32 << al;
        let m1 = -1i32 << al;
        let mut k = ss;
        if *eobrun == 0 {
            while k <= se {
                let rs = br.decode(act)?;
                let mut r = (rs >> 4) as i32;
                let s0 = (rs & 15) as u32;
                let mut s = 0i32;
                if s0 != 0 {
                    s = if br.bit() != 0 { p1 } else { m1 };
                } else if r != 15 {
                    *eobrun = 1 << r;
                    if r > 0 {
                        *eobrun += br.bits(r as u32);
                    }
                    break;
                }
                while k <= se {
                    let co = &mut blk[ZZ[k]];
                    if *co != 0 {
                        if br.bit() != 0 && (*co as i32 & p1) == 0 {
                            *co = if *co >= 0 { (*co as i32 + p1) as i16 } else { (*co as i32 + m1) as i16 };
                        }
                    } else {
                        r -= 1;
                        if r < 0 {
                            break;
                        }
                    }
                    k += 1;
                }
                if s != 0 && k <= 63 {
                    blk[ZZ[k]] = s as i16;
                }
                k += 1;
            }
        }
        if *eobrun > 0 {
            while k <= se {
                let co = &mut blk[ZZ[k]];
                if *co != 0 && br.bit() != 0 && (*co as i32 & p1) == 0 {
                    *co = if *co >= 0 { (*co as i32 + p1) as i16 } else { (*co as i32 + m1) as i16 };
                }
                k += 1;
            }
            *eobrun -= 1;
        }
        Ok(())
    }

    // ---- libjpeg jidctint.c (islow) ----
    const CONST_BITS: i32 = 13;
    const PASS1_BITS: i32 = 2;
    const FIX_0_298631336: i32 = 2446;
    const FIX_0_390180644: i32 = 3196;
    const FIX_0_541196100: i32 = 4433;
    const FIX_0_765366865: i32 = 6270;
    const FIX_0_899976223: i32 = 7373;
    const FIX_1_175875602: i32 = 9633;
    const FIX_1_501321110: i32 = 12299;
    const FIX_1_847759065: i32 = 15137;
    const FIX_1_961570560: i32 = 16069;
    const FIX_2_053119869: i32 = 16819;
    const FIX_2_562915447: i32 = 20995;
    const FIX_3_072711026: i32 = 25172;

    #[inline]
    fn descale(x: i32, n: i32) -> i32 {
        (x + (1 << (n - 1))) >> n
    }
    #[inline]
    fn range_limit(x: i32) -> u8 {
        let i = (x & 1023) as usize;
        if i < 128 {
            (i + 128) as u8
        } else if i < 512 {
            255
        } else if i < 896 {
            0
        } else {
            (i - 896) as u8
        }
    }

    #[inline(always)]
    fn idct_1d(i0: i32, i1: i32, i2: i32, i3: i32, i4: i32, i5: i32, i6: i32, i7: i32) -> [i32; 8] {
        // returns un-descaled outputs 0..7 (scaled by 2^CONST_BITS)
        let z1 = (i2 + i6) * FIX_0_541196100;
        let tmp2 = z1 + i6 * -FIX_1_847759065;
        let tmp3 = z1 + i2 * FIX_0_765366865;
        let tmp0 = (i0 + i4) << CONST_BITS;
        let tmp1 = (i0 - i4) << CONST_BITS;
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        let (mut t0, mut t1, mut t2, mut t3) = (i7, i5, i3, i1);
        let z1 = t0 + t3;
        let z2 = t1 + t2;
        let z3 = t0 + t2;
        let z4 = t1 + t3;
        let z5 = (z3 + z4) * FIX_1_175875602;
        t0 *= FIX_0_298631336;
        t1 *= FIX_2_053119869;
        t2 *= FIX_3_072711026;
        t3 *= FIX_1_501321110;
        let z1 = z1 * -FIX_0_899976223;
        let z2 = z2 * -FIX_2_562915447;
        let z3 = z3 * -FIX_1_961570560 + z5;
        let z4 = z4 * -FIX_0_390180644 + z5;
        t0 += z1 + z3;
        t1 += z2 + z4;
        t2 += z2 + z3;
        t3 += z1 + z4;
        [tmp10 + t3, tmp11 + t2, tmp12 + t1, tmp13 + t0, tmp13 - t0, tmp12 - t1, tmp11 - t2, tmp10 - t3]
    }

    fn idct_islow(coef: &[i16; 64], q: &[u16; 64], out: &mut [u8], stride: usize) {
        let mut ws = [0i32; 64];
        for col in 0..8 {
            let c = |r: usize| coef[r * 8 + col] as i32 * q[r * 8 + col] as i32;
            if coef[8 + col] == 0 && coef[16 + col] == 0 && coef[24 + col] == 0 && coef[32 + col] == 0
                && coef[40 + col] == 0 && coef[48 + col] == 0 && coef[56 + col] == 0 {
                let dc = c(0) << PASS1_BITS;
                for r in 0..8 {
                    ws[r * 8 + col] = dc;
                }
                continue;
            }
            let o = idct_1d(c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7));
            let n = CONST_BITS - PASS1_BITS;
            for r in 0..8 {
                ws[r * 8 + col] = descale(o[r], n);
            }
        }
        let n = CONST_BITS + PASS1_BITS + 3;
        for row in 0..8 {
            let w: &[i32; 8] = ws[row * 8..row * 8 + 8].try_into().unwrap();
            let o = idct_1d(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
            let dst = &mut out[row * stride..row * stride + 8];
            for i in 0..8 {
                dst[i] = range_limit(descale(o[i], n));
            }
        }
    }

    /// libjpeg h2v1 fancy upsampling (triangle filter), per row.
    fn h2v1_fancy(p: &[u8], pw: usize, cw: usize, _ch: usize, w: usize, h: usize) -> Vec<u8> {
        let mut o = vec![0u8; w * h];
        let mut row = vec![0u8; cw * 2];
        for y in 0..h {
            let inp = &p[y * pw..y * pw + cw];
            up_row_h2(inp, &mut row);
            o[y * w..(y + 1) * w].copy_from_slice(&row[..w]);
        }
        o
    }

    fn up_row_h2(inp: &[u8], out: &mut [u8]) {
        let n = inp.len();
        if n == 1 {
            out[0] = inp[0];
            out[1] = inp[0];
            return;
        }
        out[0] = inp[0];
        out[1] = ((inp[0] as i32 * 3 + inp[1] as i32 + 2) >> 2) as u8;
        for i in 1..n - 1 {
            let v = inp[i] as i32 * 3;
            out[2 * i] = ((v + inp[i - 1] as i32 + 1) >> 2) as u8;
            out[2 * i + 1] = ((v + inp[i + 1] as i32 + 2) >> 2) as u8;
        }
        let v = inp[n - 1] as i32 * 3;
        out[2 * n - 2] = ((v + inp[n - 2] as i32 + 1) >> 2) as u8;
        out[2 * n - 1] = inp[n - 1];
    }

    /// libjpeg h2v2 fancy upsampling (triangle filter in both directions).
    fn h2v2_fancy(p: &[u8], pw: usize, cw: usize, ch: usize, w: usize, h: usize) -> Vec<u8> {
        let mut o = vec![0u8; w * h];
        let mut tmp = vec![0u8; cw * 2];
        for y in 0..h {
            let iy = y / 2;
            let ny = if y % 2 == 0 { iy.saturating_sub(1) } else { (iy + 1).min(ch - 1) };
            let r0 = &p[iy * pw..iy * pw + cw];
            let r1 = &p[ny * pw..ny * pw + cw];
            let cs = |i: usize| r0[i] as i32 * 3 + r1[i] as i32;
            if cw == 1 {
                let v = ((cs(0) * 4 + 8) >> 4) as u8;
                tmp[0] = v;
                tmp[1] = ((cs(0) * 4 + 7) >> 4) as u8;
            } else {
                let mut this = cs(0);
                let mut next = cs(1);
                tmp[0] = ((this * 4 + 8) >> 4) as u8;
                tmp[1] = ((this * 3 + next + 7) >> 4) as u8;
                let mut last = this;
                this = next;
                for i in 1..cw - 1 {
                    next = cs(i + 1);
                    tmp[2 * i] = ((this * 3 + last + 8) >> 4) as u8;
                    tmp[2 * i + 1] = ((this * 3 + next + 7) >> 4) as u8;
                    last = this;
                    this = next;
                }
                tmp[2 * cw - 2] = ((this * 3 + last + 8) >> 4) as u8;
                tmp[2 * cw - 1] = ((this * 4 + 7) >> 4) as u8;
            }
            o[y * w..(y + 1) * w].copy_from_slice(&tmp[..w]);
        }
        o
    }

    struct ColorTables {
        cr_r: [i32; 256],
        cb_b: [i32; 256],
        cr_g: [i32; 256],
        cb_g: [i32; 256],
    }
    impl ColorTables {
        fn new() -> ColorTables {
            const SB: i32 = 16;
            const HALF: i32 = 1 << (SB - 1);
            let fix = |x: f64| (x * (1i64 << SB) as f64 + 0.5) as i32;
            let mut t = ColorTables { cr_r: [0; 256], cb_b: [0; 256], cr_g: [0; 256], cb_g: [0; 256] };
            for i in 0..256 {
                let x = i as i32 - 128;
                t.cr_r[i] = (fix(1.40200) * x + HALF) >> SB;
                t.cb_b[i] = (fix(1.77200) * x + HALF) >> SB;
                t.cr_g[i] = -fix(0.71414) * x;
                t.cb_g[i] = -fix(0.34414) * x + HALF;
            }
            t
        }
    }

    fn exif_orientation(t: &[u8]) -> Option<u16> {
        if t.len() < 8 {
            return None;
        }
        let le = &t[0..2] == b"II";
        let u16_ = |o: usize| t.get(o..o + 2).map(|b| if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) });
        let u32_ = |o: usize| {
            t.get(o..o + 4).map(|b| if le { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) } else { u32::from_be_bytes([b[0], b[1], b[2], b[3]]) })
        };
        let ifd = u32_(4)? as usize;
        let n = u16_(ifd)? as usize;
        for i in 0..n {
            let e = ifd + 2 + i * 12;
            if u16_(e)? == 0x0112 {
                return u16_(e + 8);
            }
        }
        None
    }
}

// ================================================================================================ PNG
pub mod png {
    use super::{Image, MAX_PIXELS};

    pub fn decode(d: &[u8]) -> Result<Image, String> {
        let mut pos = 8;
        let (mut w, mut h, mut depth, mut ctype, mut interlace) = (0usize, 0usize, 0u8, 0u8, 0u8);
        let mut idat = Vec::new();
        let mut palette: Vec<u8> = Vec::new();
        while pos + 8 <= d.len() {
            let len = u32::from_be_bytes([d[pos], d[pos + 1], d[pos + 2], d[pos + 3]]) as usize;
            let ty = &d[pos + 4..pos + 8];
            let body = d.get(pos + 8..pos + 8 + len).ok_or("truncated PNG")?;
            pos += 12 + len;
            match ty {
                b"IHDR" => {
                    w = u32::from_be_bytes([body[0], body[1], body[2], body[3]]) as usize;
                    h = u32::from_be_bytes([body[4], body[5], body[6], body[7]]) as usize;
                    depth = body[8];
                    ctype = body[9];
                    interlace = body[12];
                    if w == 0 || h == 0 || w * h > MAX_PIXELS {
                        return Err(format!("bad PNG size {w}x{h}"));
                    }
                }
                b"PLTE" => palette = body.to_vec(),
                b"IDAT" => idat.extend_from_slice(body),
                b"IEND" => break,
                _ => {}
            }
        }
        let channels = match ctype {
            0 => 1,
            2 => 3,
            3 => 1,
            4 => 2,
            6 => 4,
            _ => return Err("bad PNG color type".into()),
        };
        let bpp_bits = channels * depth as usize;
        let raw = super::inflate::zlib(&idat, (w * bpp_bits).div_ceil(8) * h + h + 64)?;
        let mut samples = vec![0u16; w * h * channels]; // full-depth samples
        let passes: Vec<(usize, usize, usize, usize)> = if interlace == 1 {
            vec![(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
        } else {
            vec![(0, 0, 1, 1)]
        };
        let mut off = 0;
        let bpp = bpp_bits.div_ceil(8).max(1);
        for (x0, y0, dx, dy) in passes {
            if x0 >= w || y0 >= h {
                continue;
            }
            let pw = (w - x0).div_ceil(dx);
            let ph = (h - y0).div_ceil(dy);
            let stride = (pw * bpp_bits).div_ceil(8);
            let mut prev = vec![0u8; stride];
            let mut cur = vec![0u8; stride];
            for py in 0..ph {
                let ft = *raw.get(off).ok_or("truncated PNG data")?;
                cur.copy_from_slice(raw.get(off + 1..off + 1 + stride).ok_or("truncated PNG data")?);
                off += 1 + stride;
                unfilter(ft, &mut cur, &prev, bpp)?;
                for px in 0..pw {
                    for c in 0..channels {
                        let v = match depth {
                            8 => cur[px * channels + c] as u16,
                            16 => u16::from_be_bytes([cur[(px * channels + c) * 2], cur[(px * channels + c) * 2 + 1]]),
                            1 | 2 | 4 => {
                                let bit = (px * channels + c) * depth as usize;
                                let byte = cur[bit / 8];
                                let shift = 8 - depth as usize - bit % 8;
                                ((byte >> shift) & ((1u8 << depth) - 1)) as u16
                            }
                            _ => return Err("bad PNG bit depth".into()),
                        };
                        samples[((y0 + py * dy) * w + x0 + px * dx) * channels + c] = v;
                    }
                }
                std::mem::swap(&mut prev, &mut cur);
            }
        }
        let to8 = |v: u16| -> u8 {
            match depth {
                16 => (v >> 8) as u8,
                8 => v as u8,
                d => ((v as u32 * 255) / ((1u32 << d) - 1)) as u8,
            }
        };
        let mut rgb = vec![0u8; w * h * 3];
        for i in 0..w * h {
            let s = &samples[i * channels..(i + 1) * channels];
            let (r, g, b) = match ctype {
                0 | 4 => {
                    let v = to8(s[0]);
                    (v, v, v)
                }
                3 => {
                    let k = s[0] as usize * 3;
                    let p = palette.get(k..k + 3).ok_or("bad PNG palette index")?;
                    (p[0], p[1], p[2])
                }
                _ => (to8(s[0]), to8(s[1]), to8(s[2])),
            };
            rgb[i * 3] = r;
            rgb[i * 3 + 1] = g;
            rgb[i * 3 + 2] = b;
        }
        Ok(Image { w, h, rgb })
    }

    fn unfilter(ft: u8, cur: &mut [u8], prev: &[u8], bpp: usize) -> Result<(), String> {
        let n = cur.len();
        match ft {
            0 => {}
            1 => {
                for i in bpp..n {
                    cur[i] = cur[i].wrapping_add(cur[i - bpp]);
                }
            }
            2 => {
                for i in 0..n {
                    cur[i] = cur[i].wrapping_add(prev[i]);
                }
            }
            3 => {
                for i in 0..n {
                    let a = if i >= bpp { cur[i - bpp] as u16 } else { 0 };
                    cur[i] = cur[i].wrapping_add(((a + prev[i] as u16) / 2) as u8);
                }
            }
            4 => {
                for i in 0..n {
                    let a = if i >= bpp { cur[i - bpp] as i16 } else { 0 };
                    let b = prev[i] as i16;
                    let c = if i >= bpp { prev[i - bpp] as i16 } else { 0 };
                    let p = a + b - c;
                    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                    let pr = if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c };
                    cur[i] = cur[i].wrapping_add(pr as u8);
                }
            }
            _ => return Err("bad PNG filter".into()),
        }
        Ok(())
    }
}

// ================================================================================================ inflate
pub mod inflate {
    struct Bits<'a> {
        d: &'a [u8],
        pos: usize,
        acc: u64,
        n: u32,
    }
    impl<'a> Bits<'a> {
        fn need(&mut self, k: u32) -> Result<(), String> {
            while self.n < k {
                let b = *self.d.get(self.pos).ok_or("truncated deflate stream")?;
                self.pos += 1;
                self.acc |= (b as u64) << self.n;
                self.n += 8;
            }
            Ok(())
        }
        fn bits(&mut self, k: u32) -> Result<u32, String> {
            if k == 0 {
                return Ok(0);
            }
            self.need(k)?;
            let v = (self.acc & ((1u64 << k) - 1)) as u32;
            self.acc >>= k;
            self.n -= k;
            Ok(v)
        }
    }

    struct Huff {
        counts: [u16; 16],
        syms: Vec<u16>,
    }
    impl Huff {
        fn new(lens: &[u8]) -> Huff {
            let mut counts = [0u16; 16];
            for &l in lens {
                counts[l as usize] += 1;
            }
            counts[0] = 0;
            let mut offs = [0u16; 16];
            for i in 1..16 {
                offs[i] = offs[i - 1] + counts[i - 1];
            }
            let mut syms = vec![0u16; lens.len()];
            for (s, &l) in lens.iter().enumerate() {
                if l != 0 {
                    syms[offs[l as usize] as usize] = s as u16;
                    offs[l as usize] += 1;
                }
            }
            Huff { counts, syms }
        }
        fn decode(&self, b: &mut Bits) -> Result<u16, String> {
            let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
            for len in 1..16 {
                code |= b.bits(1)? as i32;
                let count = self.counts[len] as i32;
                if code - count < first {
                    return Ok(self.syms[(index + (code - first)) as usize]);
                }
                index += count;
                first += count;
                first <<= 1;
                code <<= 1;
            }
            Err("bad deflate code".into())
        }
    }

    const LBASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
    const LEXT: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
    const DBASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
    const DEXT: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

    pub fn zlib(d: &[u8], hint: usize) -> Result<Vec<u8>, String> {
        if d.len() < 2 {
            return Err("truncated zlib stream".into());
        }
        inflate(&d[2..], hint)
    }

    pub fn inflate(d: &[u8], hint: usize) -> Result<Vec<u8>, String> {
        let mut out = Vec::with_capacity(hint);
        let mut b = Bits { d, pos: 0, acc: 0, n: 0 };
        loop {
            let last = b.bits(1)?;
            let ty = b.bits(2)?;
            match ty {
                0 => {
                    b.acc = 0;
                    b.n = 0;
                    let p = b.pos;
                    let len = u16::from_le_bytes([*d.get(p).ok_or("trunc")?, *d.get(p + 1).ok_or("trunc")?]) as usize;
                    out.extend_from_slice(d.get(p + 4..p + 4 + len).ok_or("truncated stored block")?);
                    b.pos = p + 4 + len;
                }
                1 | 2 => {
                    let (lit, dist) = if ty == 1 {
                        let mut l = [0u8; 288];
                        for (i, v) in l.iter_mut().enumerate() {
                            *v = if i < 144 { 8 } else if i < 256 { 9 } else if i < 280 { 7 } else { 8 };
                        }
                        (Huff::new(&l), Huff::new(&[5u8; 30]))
                    } else {
                        let hlit = b.bits(5)? as usize + 257;
                        let hdist = b.bits(5)? as usize + 1;
                        let hclen = b.bits(4)? as usize + 4;
                        const ORD: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                        let mut cl = [0u8; 19];
                        for &o in ORD.iter().take(hclen) {
                            cl[o] = b.bits(3)? as u8;
                        }
                        let ch = Huff::new(&cl);
                        let mut lens = vec![0u8; hlit + hdist];
                        let mut i = 0;
                        while i < hlit + hdist {
                            let s = ch.decode(&mut b)?;
                            match s {
                                0..=15 => {
                                    lens[i] = s as u8;
                                    i += 1;
                                }
                                16 => {
                                    let prev = *lens.get(i.wrapping_sub(1)).ok_or("bad code lengths")?;
                                    for _ in 0..3 + b.bits(2)? {
                                        *lens.get_mut(i).ok_or("bad code lengths")? = prev;
                                        i += 1;
                                    }
                                }
                                17 => i += 3 + b.bits(3)? as usize,
                                _ => i += 11 + b.bits(7)? as usize,
                            }
                        }
                        if i > hlit + hdist {
                            return Err("bad code lengths".into());
                        }
                        (Huff::new(&lens[..hlit]), Huff::new(&lens[hlit..]))
                    };
                    loop {
                        let s = lit.decode(&mut b)? as usize;
                        if s < 256 {
                            out.push(s as u8);
                        } else if s == 256 {
                            break;
                        } else {
                            let s = s - 257;
                            if s >= 29 {
                                return Err("bad length code".into());
                            }
                            let len = LBASE[s] as usize + b.bits(LEXT[s] as u32)? as usize;
                            let ds = dist.decode(&mut b)? as usize;
                            if ds >= 30 {
                                return Err("bad distance code".into());
                            }
                            let dd = DBASE[ds] as usize + b.bits(DEXT[ds] as u32)? as usize;
                            if dd > out.len() {
                                return Err("bad distance".into());
                            }
                            let st = out.len() - dd;
                            for k in 0..len {
                                let v = out[st + k];
                                out.push(v);
                            }
                        }
                    }
                }
                _ => return Err("bad deflate block type".into()),
            }
            if last == 1 {
                break;
            }
        }
        Ok(out)
    }
}
