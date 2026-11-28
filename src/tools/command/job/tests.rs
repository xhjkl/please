use super::*;
use crate::test_support::TempDir;

#[tokio::test]
async fn observing_a_running_job_ends_live_delivery_at_the_chunk_boundary() {
    let directory = TempDir::new().unwrap();
    let release = directory.0.join("more-output");
    let argv = [
        "sh".into(),
        "-c".into(),
        "printf first; while [ ! -e \"$1\" ]; do sleep 0.01; done; printf second".into(),
        "sh".into(),
        release.to_str().unwrap().into(),
    ];
    let (live, mut received) = tokio::sync::mpsc::unbounded_channel();
    let job = Job::spawn(&argv, Some(&live)).unwrap();
    drop(live);

    let first = tokio::time::timeout(Duration::from_secs(5), received.recv())
        .await
        .unwrap();
    assert_eq!(first.as_deref(), Some("first"));
    let first = job.observe();
    assert_eq!(first.stdout.text, "first");
    assert!(first.completion.is_none());
    assert_eq!(
        received.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
    );

    let (live, mut received) = tokio::sync::mpsc::unbounded_channel();
    job.stream_to(&live);
    drop(live);
    std::fs::write(release, []).unwrap();
    tokio::time::timeout(Duration::from_secs(5), job.wait())
        .await
        .unwrap();
    let finished = job.observe();
    assert_eq!(finished.stdout.text, "second");
    assert!(finished.completion.unwrap().exit.unwrap().status.success());
    assert_eq!(received.try_recv().unwrap(), "second");
    assert_eq!(
        received.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
    );
}
