//! Stream checkpoint bytes through macOS's libcurl, keeping HTTP headers out of the partial file.
use curl::easy::{Easy2, Handler, WriteError};
use eyre::{Result, WrapErr, eyre};
use std::ffi::c_long;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

/// Resume a partial checkpoint and sync it only after the pinned size has been verified.
pub fn fetch(url: &str, path: &Path, expected: u64) -> Result<()> {
    let metadata = fs::metadata(path);
    let offset = match metadata {
        Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
        Ok(metadata) => metadata.len(),
    };
    if offset == expected {
        return Ok(());
    }
    if offset > expected {
        eyre::bail!("partial download exceeds expected size: {}", path.display());
    }

    let mut request = Easy2::new(Download {
        path,
        expected,
        offset,
        head: ResponseHead::default(),
        file: None,
        error: None,
    });
    request.url(url)?;
    request.useragent(concat!("please/", env!("CARGO_PKG_VERSION")))?;
    request.follow_location(true)?;
    request.max_redirections(10)?;
    if offset > 0 {
        request.range(&format!("{offset}-"))?;
    }
    let protocols = c_long::from(curl_sys::CURLPROTO_HTTP | curl_sys::CURLPROTO_HTTPS);
    // The easy handle owns the live pointer, and this option takes a `long` bitmask.
    let configured = unsafe {
        curl_sys::curl_easy_setopt(request.raw(), curl_sys::CURLOPT_REDIR_PROTOCOLS, protocols)
    };
    if configured != curl_sys::CURLE_OK {
        return Err(curl::Error::new(configured).into());
    }

    let transferred = request.perform();
    let download = request.get_mut();
    if let Some(error) = download.error.take() {
        return Err(error);
    }
    transferred.wrap_err("checkpoint download failed")?;
    download.finish()
}

struct Download<'a> {
    path: &'a Path,
    expected: u64,
    offset: u64,
    head: ResponseHead,
    file: Option<File>,
    /// Why the body could not be written; the write callback only reports a byte count to libcurl.
    error: Option<eyre::Report>,
}

impl Handler for Download<'_> {
    fn header(&mut self, header: &[u8]) -> bool {
        if self.file.is_none() {
            self.head.read(header);
        }
        true
    }

    fn write(&mut self, chunk: &[u8]) -> std::result::Result<usize, WriteError> {
        if chunk.is_empty() {
            return Ok(0);
        }
        let written = self.write_body(chunk);
        match written {
            Err(error) => {
                self.error = Some(error);
                Ok(0)
            }
            Ok(()) => Ok(chunk.len()),
        }
    }
}

impl Download<'_> {
    fn write_body(&mut self, chunk: &[u8]) -> Result<()> {
        if self.file.is_none() {
            self.offset = self.head.validate(self.offset, self.expected)?;
        }
        if chunk.len() as u64 > self.expected - self.offset {
            eyre::bail!("download exceeds expected size");
        }
        let file = match &mut self.file {
            Some(file) => file,
            slot @ None => {
                let file = File::options()
                    .create(true)
                    .write(true)
                    .truncate(self.offset == 0)
                    .append(self.offset > 0)
                    .open(self.path)?;
                slot.insert(file)
            }
        };
        file.write_all(chunk)?;
        self.offset += chunk.len() as u64;
        eprint!(
            "\rplease load: {}/{} bytes ({:.1}%)",
            self.offset,
            self.expected,
            self.offset as f64 * 100.0 / self.expected as f64,
        );
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        let Some(file) = &self.file else {
            let offset = self.head.validate(self.offset, self.expected)?;
            eyre::bail!(
                "download size mismatch: expected {}, got {offset}",
                self.expected
            );
        };
        if self.offset != self.expected {
            eyre::bail!(
                "download size mismatch: expected {}, got {}",
                self.expected,
                self.offset
            );
        }
        file.sync_all()?;
        Ok(())
    }
}

#[derive(Default)]
struct ResponseHead {
    status: Option<u16>,
    length: Option<u64>,
    range: Option<String>,
}

impl ResponseHead {
    fn read(&mut self, line: &[u8]) {
        if line.starts_with(b"HTTP/") {
            // libcurl reports each response head, including redirects and proxy handshakes.
            let code = line.splitn(3, |&byte| byte == b' ').nth(1);
            let status = code
                .and_then(|code| std::str::from_utf8(code.trim_ascii()).ok())
                .and_then(|code| code.parse().ok());
            *self = Self {
                status,
                ..Self::default()
            };
            return;
        }
        let mut header = line.splitn(2, |&byte| byte == b':');
        let name = header.next().expect("splitn yields a first element");
        let Some(value) = header.next() else { return };
        let value = std::str::from_utf8(value.trim_ascii()).ok();
        if name.eq_ignore_ascii_case(b"Content-Length") {
            self.length = value.and_then(|value| value.parse().ok());
        } else if name.eq_ignore_ascii_case(b"Content-Range") {
            self.range = value.map(str::to_owned);
        }
    }

    /// Return where the server's body starts, after checking the exact resumed range and size.
    fn validate(&self, offset: u64, expected: u64) -> Result<u64> {
        let status = self
            .status
            .ok_or_else(|| eyre!("curl returned no HTTP status"))?;
        let offset = match status {
            200 => 0,
            206 => {
                let expected_range = format!("bytes {offset}-{}/{expected}", expected - 1);
                let range = self.range.as_deref();
                if range != Some(expected_range.as_str()) {
                    eyre::bail!("unexpected download Content-Range: {range:?}");
                }
                offset
            }
            status => eyre::bail!("download failed: HTTP {status}"),
        };
        if let Some(length) = self.length
            && length != expected - offset
        {
            eyre::bail!(
                "download size changed: expected {} bytes, got {length}",
                expected - offset
            );
        }
        Ok(offset)
    }
}
