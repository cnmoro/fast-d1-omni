//! Small JSON parser/serializer. Serialization follows Python's `json.dumps(ensure_ascii=False)` (separators
//! ", " and ": ", key order kept, floats in `repr` form) because the model was trained on that rendering.

use std::fmt::Write;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// Number with its source lexeme (integers are re-emitted verbatim, as Python ints are).
    Num(f64, Option<String>),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

#[allow(dead_code)]
impl Json {
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().find(|(kk, _)| kk == k).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(f, _) => Some(*f),
            _ => None,
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
    pub fn num(f: f64) -> Json {
        Json::Num(f, None)
    }
    pub fn str(s: impl Into<String>) -> Json {
        Json::Str(s.into())
    }
}

// ------------------------------------------------------------------------------------------------ parse

pub fn parse(s: &str) -> Result<Json, String> {
    let mut p = Parser { b: s.as_bytes(), i: 0, depth: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(p.err("trailing characters"));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, m: &str) -> String {
        format!("invalid JSON at byte {}: {m}", self.i)
    }
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn lit(&mut self, w: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(w.as_bytes()) {
            self.i += w.len();
            Ok(v)
        } else {
            Err(self.err("unexpected token"))
        }
    }
    fn value(&mut self) -> Result<Json, String> {
        if self.i >= self.b.len() {
            return Err(self.err("unexpected end"));
        }
        match self.b[self.i] {
            b'{' => {
                self.depth += 1;
                if self.depth > 512 {
                    return Err(self.err("nesting too deep"));
                }
                self.i += 1;
                let mut obj: Vec<(String, Json)> = Vec::new();
                self.ws();
                if self.i < self.b.len() && self.b[self.i] == b'}' {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Json::Obj(obj));
                }
                loop {
                    self.ws();
                    if self.i >= self.b.len() || self.b[self.i] != b'"' {
                        return Err(self.err("expected key"));
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.i >= self.b.len() || self.b[self.i] != b':' {
                        return Err(self.err("expected ':'"));
                    }
                    self.i += 1;
                    self.ws();
                    let v = self.value()?;
                    match obj.iter_mut().find(|(kk, _)| *kk == k) {
                        Some(e) => e.1 = v, // like a Python dict: last value, first position
                        None => obj.push((k, v)),
                    }
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            self.depth -= 1;
                            return Ok(Json::Obj(obj));
                        }
                        _ => return Err(self.err("expected ',' or '}'")),
                    }
                }
            }
            b'[' => {
                self.depth += 1;
                if self.depth > 512 {
                    return Err(self.err("nesting too deep"));
                }
                self.i += 1;
                let mut arr = Vec::new();
                self.ws();
                if self.i < self.b.len() && self.b[self.i] == b']' {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Json::Arr(arr));
                }
                loop {
                    self.ws();
                    arr.push(self.value()?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            self.depth -= 1;
                            return Ok(Json::Arr(arr));
                        }
                        _ => return Err(self.err("expected ',' or ']'")),
                    }
                }
            }
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => self.lit("true", Json::Bool(true)),
            b'f' => self.lit("false", Json::Bool(false)),
            b'n' => self.lit("null", Json::Null),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(self.err("unexpected character")),
        }
    }
    fn number(&mut self) -> Result<Json, String> {
        let st = self.i;
        if self.b[self.i] == b'-' {
            self.i += 1;
        }
        let digits = |p: &mut Self| {
            let s = p.i;
            while p.i < p.b.len() && p.b[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i - s
        };
        if self.i < self.b.len() && self.b[self.i] == b'0' {
            self.i += 1;
        } else if digits(self) == 0 {
            return Err(self.err("bad number"));
        }
        let mut is_int = true;
        if self.i < self.b.len() && self.b[self.i] == b'.' {
            self.i += 1;
            is_int = false;
            if digits(self) == 0 {
                return Err(self.err("bad number"));
            }
        }
        if self.i < self.b.len() && (self.b[self.i] == b'e' || self.b[self.i] == b'E') {
            self.i += 1;
            is_int = false;
            if self.i < self.b.len() && (self.b[self.i] == b'+' || self.b[self.i] == b'-') {
                self.i += 1;
            }
            if digits(self) == 0 {
                return Err(self.err("bad number"));
            }
        }
        let lex = std::str::from_utf8(&self.b[st..self.i]).unwrap();
        let f: f64 = lex.parse().map_err(|_| self.err("bad number"))?;
        Ok(Json::Num(f, if is_int { Some(lex.to_string()) } else { None }))
    }
    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err(self.err("bad escape"));
        }
        let s = std::str::from_utf8(&self.b[self.i..self.i + 4]).map_err(|_| self.err("bad escape"))?;
        let v = u32::from_str_radix(s, 16).map_err(|_| self.err("bad escape"))?;
        self.i += 4;
        Ok(v)
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let st = self.i;
            while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' && self.b[self.i] >= 0x20 {
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.b[st..self.i]).map_err(|_| self.err("invalid UTF-8"))?);
            match self.b.get(self.i) {
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let c = *self.b.get(self.i).ok_or_else(|| self.err("bad escape"))?;
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xd800..0xdc00).contains(&cp) && self.b[self.i..].starts_with(b"\\u") {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) {
                                    cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                                } else {
                                    self.i = save;
                                }
                            }
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(self.err("bad escape")),
                    }
                }
                Some(_) => return Err(self.err("control character in string")),
                None => return Err(self.err("unterminated string")),
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ serialize

/// Python `repr(float)`.
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let s = format!("{:e}", x); // shortest round-trip digits: "-1.2345e-5"
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    let decpt = exp + 1;
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..(-decpt) {
            out.push('0');
        }
        out.push_str(&digits);
    } else {
        let d = decpt as usize;
        if digits.len() <= d {
            out.push_str(&digits);
            for _ in digits.len()..d {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..d]);
            out.push('.');
            out.push_str(&digits[d..]);
        }
    }
    out
}

pub fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `json.dumps(v, ensure_ascii=False)` (pretty = false) or a compact form for responses.
pub fn dumps(v: &Json, out: &mut String, compact: bool) {
    let (isep, ksep) = if compact { (",", ":") } else { (", ", ": ") };
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(f, lex) => match lex {
            Some(l) => {
                let l = l.trim_start_matches('-');
                let zero = l.chars().all(|c| c == '0');
                if f.is_sign_negative() && !zero {
                    out.push('-');
                }
                out.push_str(l);
            }
            None => out.push_str(&py_float(*f)),
        },
        Json::Str(s) => write_str(out, s),
        Json::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(isep);
                }
                dumps(x, out, compact);
            }
            out.push(']');
        }
        Json::Obj(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(isep);
                }
                write_str(out, k);
                out.push_str(ksep);
                dumps(x, out, compact);
            }
            out.push('}');
        }
    }
}

pub fn to_string(v: &Json) -> String {
    let mut s = String::new();
    dumps(v, &mut s, false);
    s
}

pub fn to_string_compact(v: &Json) -> String {
    let mut s = String::new();
    dumps(v, &mut s, true);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn floats() {
        for (x, s) in [(0.0001, "0.0001"), (0.00001, "1e-05"), (1e16, "1e+16"), (1e15, "1000000000000000.0"), (1.5, "1.5"),
                       (0.998469889163971, "0.998469889163971"), (123.0, "123.0"), (-2.5e-7, "-2.5e-07")] {
            assert_eq!(py_float(x), s);
        }
    }
    #[test]
    fn roundtrip() {
        let v = parse(r#"{"a": [1, 2.50, -0, "xé\n"], "b": {"c": null, "d": true}}"#).unwrap();
        assert_eq!(to_string(&v), r#"{"a": [1, 2.5, 0, "xé\n"], "b": {"c": null, "d": true}}"#);
    }
}
