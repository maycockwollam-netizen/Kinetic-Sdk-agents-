use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Debug)]
struct Hunk {
    old_start: usize,
    old_count: usize,
    new_start: usize,
    new_count: usize,
    lines: Vec<String>,
}

fn lines_keepends(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\n' || bytes[i] == b'\r' {
            let end = if bytes[i] == b'\r' && i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                i + 2
            } else {
                i + 1
            };
            result.push(text[start..end].to_string());
            start = end;
            i = end;
        } else {
            i += 1;
        }
    }
    if start < text.len() {
        result.push(text[start..].to_string());
    }
    result
}

fn strip_line_ending(line: &str) -> &str {
    line.strip_suffix("\r\n")
        .or_else(|| line.strip_suffix('\n'))
        .or_else(|| line.strip_suffix('\r'))
        .unwrap_or(line)
}

fn parse_path(header: &str, prefix: &str) -> Result<String, String> {
    let value = header[prefix.len()..]
        .split('\t')
        .next()
        .unwrap_or("")
        .trim();
    let value = if value.starts_with("a/") || value.starts_with("b/") {
        &value[2..]
    } else {
        value
    };
    if value.is_empty() || value == "/dev/null" {
        return Err("patches must modify existing workspace files".into());
    }
    Ok(value.to_string())
}

fn parse_hunk_header(line: &str) -> Option<(usize, usize, usize, usize)> {
    let rest = line.strip_prefix("@@ -")?;
    let (old, rest) = rest.split_once(" +")?;
    let (new, _) = rest.split_once(" @@")?;
    fn range(value: &str) -> Option<(usize, usize)> {
        let mut parts = value.split(',');
        let start = parts.next()?.parse().ok()?;
        let count = match parts.next() {
            Some(v) => v.parse().ok()?,
            None => 1,
        };
        if parts.next().is_some() {
            return None;
        }
        Some((start, count))
    }
    let (old_start, old_count) = range(old)?;
    let (new_start, new_count) = range(new)?;
    Some((old_start, old_count, new_start, new_count))
}

fn parse_patch_impl(patch: &str) -> Result<Vec<(String, Vec<Hunk>)>, String> {
    if patch.trim().is_empty() {
        return Err("no patch content".into());
    }
    let lines = lines_keepends(patch);
    let mut patches = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let line = &lines[index];
        if line.starts_with("diff --git ")
            || line.starts_with("index ")
            || line.starts_with("new file mode ")
            || line.starts_with("deleted file mode ")
        {
            index += 1;
            continue;
        }
        if !line.starts_with("--- ") {
            return Err(format!("expected --- file header at line {}", index + 1));
        }
        let old_path = parse_path(line, "--- ")?;
        index += 1;
        if index >= lines.len() || !lines[index].starts_with("+++ ") {
            return Err("missing +++ file header".into());
        }
        let new_path = parse_path(&lines[index], "+++ ")?;
        if old_path != new_path {
            return Err("renames are not supported".into());
        }
        let path = new_path;
        index += 1;
        let mut hunks = Vec::new();
        while index < lines.len() && !lines[index].starts_with("--- ") {
            let header = match parse_hunk_header(strip_line_ending(&lines[index])) {
                Some(value) => value,
                None if lines[index].starts_with("diff --git ") || lines[index].starts_with("index ") => break,
                None => return Err(format!("expected @@ hunk header at line {}", index + 1)),
            };
            let (old_start, old_count, new_start, new_count) = header;
            index += 1;
            let mut hunk_lines: Vec<String> = Vec::new();
            let (mut old_seen, mut new_seen) = (0, 0);
            while index < lines.len() {
                let hunk_line = &lines[index];
                if hunk_line.starts_with("--- ") || hunk_line.starts_with("diff --git ") {
                    break;
                }
                if hunk_line.starts_with("\\ No newline at end of file") {
                    if hunk_lines.is_empty() {
                        return Err("newline marker has no preceding hunk line".into());
                    }
                    let prior = hunk_lines.last_mut().unwrap();
                    *prior = strip_line_ending(prior).to_string();
                    index += 1;
                    continue;
                }
                let first = hunk_line.chars().next();
                if matches!(first, Some(' ' | '+' | '-')) {
                    hunk_lines.push(hunk_line.clone());
                    if matches!(first, Some(' ' | '-')) { old_seen += 1; }
                    if matches!(first, Some(' ' | '+')) { new_seen += 1; }
                    index += 1;
                    continue;
                }
                break;
            }
            if old_seen != old_count || new_seen != new_count {
                return Err("hunk line counts do not match its header".into());
            }
            hunks.push(Hunk { old_start, old_count, new_start, new_count, lines: hunk_lines });
        }
        if hunks.is_empty() {
            return Err(format!("no hunks for {path}"));
        }
        patches.push((path, hunks));
    }
    Ok(patches)
}

#[pyfunction]
fn parse_patch(patch: &str) -> PyResult<Vec<(String, Vec<(usize, usize, usize, usize, Vec<String>)>)>> {
    let parsed = parse_patch_impl(patch).map_err(PyValueError::new_err)?;
    Ok(parsed.into_iter().map(|(path, hunks)| {
        (path, hunks.into_iter().map(|h| (h.old_start, h.old_count, h.new_start, h.new_count, h.lines)).collect())
    }).collect())
}

fn apply_impl(original: &str, hunks: &[Hunk]) -> Result<String, String> {
    let mut lines = lines_keepends(original);
    let mut offset: isize = 0;
    let newline = if original.contains("\r\n") { "\r\n" } else { "\n" };
    for hunk in hunks {
        let old_index = if hunk.old_count == 0 { hunk.old_start as isize } else { hunk.old_start as isize - 1 };
        let mut start_signed = old_index + offset;
        if start_signed < 0 || start_signed as usize > lines.len() {
            return Err("hunk position is outside the file".into());
        }
        let file_newline = |line: &str| {
            let body = strip_line_ending(line);
            let has_ending = line.ends_with('\n') || line.ends_with('\r');
            if has_ending { format!("{body}{newline}") } else { body.to_string() }
        };
        let expected: Vec<String> = hunk.lines.iter().filter(|l| l.starts_with(' ') || l.starts_with('-')).map(|l| file_newline(&l[1..])).collect();
        let replacement: Vec<String> = hunk.lines.iter().filter(|l| l.starts_with(' ') || l.starts_with('+')).map(|l| file_newline(&l[1..])).collect();
        let mut start = start_signed as usize;
        let mut end = start + expected.len();
        if lines.get(start..end) != Some(expected.as_slice()) {
            let lower = start.saturating_sub(20);
            let upper = std::cmp::min(lines.len().saturating_sub(expected.len()), start.saturating_add(20));
            let candidates: Vec<usize> = (lower..=upper).filter(|candidate| lines.get(*candidate..candidate + expected.len()) == Some(expected.as_slice())).collect();
            if candidates.len() != 1 {
                return Err(if candidates.len() > 1 { "hunk context matches multiple nearby locations" } else { "hunk context does not match the file within 20 lines" }.into());
            }
            start = candidates[0];
            end = start + expected.len();
            start_signed = start as isize;
        }
        lines.splice(start..end, replacement.iter().cloned());
        offset += (start_signed - old_index - offset) + replacement.len() as isize - expected.len() as isize;
    }
    Ok(lines.concat())
}

#[pyfunction]
fn apply_file_patch(original: &str, raw_hunks: Vec<(usize, usize, usize, usize, Vec<String>)>) -> PyResult<String> {
    let hunks: Vec<Hunk> = raw_hunks.into_iter().map(|(old_start, old_count, new_start, new_count, lines)| Hunk { old_start, old_count, new_start, new_count, lines }).collect();
    apply_impl(original, &hunks).map_err(PyValueError::new_err)
}

#[pymodule]
fn _patch_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(parse_patch, module)?)?;
    module.add_function(wrap_pyfunction!(apply_file_patch, module)?)?;
    Ok(())
}
