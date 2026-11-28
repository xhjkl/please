use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
#[test]
fn verification_rejects_same_size_corruption() -> Result<()> {
    let directory = crate::test_support::TempDir::new()?;
    let path = directory.0.join("checkpoint.gguf.part");
    let hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    std::fs::write(&path, b"abc")?;
    verify_blocking(&path, 3, hash)?;
    std::fs::write(&path, b"abb")?;
    let result = verify_blocking(&path, 3, hash);
    assert!(result.unwrap_err().to_string().contains("has SHA-256"));
    Ok(())
}
async fn server(response: String) -> Result<(String, tokio::task::JoinHandle<String>)> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        loop {
            let byte = socket.read_u8().await.unwrap();
            request.push(byte);
            if request.ends_with(b"\r\n\r\n") {
                break;
            }
        }
        socket.write_all(response.as_bytes()).await.unwrap();
        String::from_utf8(request).unwrap()
    });
    Ok((format!("http://{address}/model.gguf"), task))
}

#[tokio::test]
async fn download_resumes_and_handles_ignored_ranges() -> Result<()> {
    let directory = crate::test_support::TempDir::new()?;
    let path = directory.0.join("model.gguf.part");
    for response in [
        "HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 3-5/6\r\nConnection: close\r\n\r\ndef",
        "HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabcdef",
    ] {
        std::fs::write(&path, b"abc")?;
        let (url, task) = server(response.to_string()).await?;
        fetch(&url, &path, 6).await?;
        let request = task.await?;
        assert!(request.to_lowercase().contains("range: bytes=3-"));
        let bytes = std::fs::read(&path)?;
        assert_eq!(bytes, b"abcdef");
    }
    // A complete partial file needs no network request before structural validation.
    fetch("http://127.0.0.1:1/unreachable", &path, 6).await?;
    Ok(())
}

#[tokio::test]
async fn download_follows_redirects_and_discards_trailers() -> Result<()> {
    let directory = crate::test_support::TempDir::new()?;
    let path = directory.0.join("model.gguf.part");
    std::fs::write(&path, b"abc")?;
    let (target, final_request) = server(
        concat!(
            "HTTP/1.1 103 Early Hints\r\nLink: </model.gguf>\r\n\r\n",
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 3-5/6\r\n",
            "Transfer-Encoding: chunked\r\nTrailer: X-Trailer\r\nConnection: close\r\n\r\n",
            "3\r\ndef\r\n0\r\nX-Trailer: metadata\r\n\r\n",
        )
        .to_string(),
    )
    .await?;
    let (url, redirect_request) = server(format!(
        "HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 4\r\nConnection: close\r\n\r\nskip",
    )).await?;

    fetch(&url, &path, 6).await?;

    for request in [redirect_request.await?, final_request.await?] {
        assert!(request.to_lowercase().contains("range: bytes=3-"));
    }
    let bytes = std::fs::read(&path)?;
    assert_eq!(bytes, b"abcdef");
    Ok(())
}

#[tokio::test]
async fn failed_or_misaligned_download_keeps_existing_bytes() -> Result<()> {
    let directory = crate::test_support::TempDir::new()?;
    let path = directory.0.join("model.gguf.part");
    for response in [
        "HTTP/1.1 403 Forbidden\r\nContent-Length: 3\r\nConnection: close\r\n\r\nerr",
        "HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 0-2/6\r\nConnection: close\r\n\r\nabc",
        "HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc",
    ] {
        std::fs::write(&path, b"abc")?;
        let (url, task) = server(response.to_string()).await?;
        let result = fetch(&url, &path, 6).await;
        assert!(result.is_err());
        task.await?;
        let bytes = std::fs::read(&path)?;
        assert_eq!(bytes, b"abc");
    }
    Ok(())
}
