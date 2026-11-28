use super::*;

fn words(line: &str) -> Vec<String> {
    line.split_whitespace().map(String::from).collect()
}

fn read<'a>(args: &'a [String], input: Option<&str>, terminal: bool) -> Result<Invocation<'a>> {
    invocation(
        args,
        specials::parse(args),
        input.map(String::from),
        terminal,
    )
}

#[test]
fn piped_data_makes_any_invocation_a_request() -> Result<()> {
    let (help, compact) = (words("help"), words("compact"));
    let Invocation::Ordinary(Mode::Request(request)) = read(&help, Some("data"), false)? else {
        panic!("piped data with a special's name is a request");
    };
    assert_eq!(
        request,
        [Message::Input("data".into()), Message::User("help".into())]
    );
    assert!(matches!(
        read(&compact, Some("data"), false)?,
        Invocation::Ordinary(Mode::Request(_))
    ));

    assert!(matches!(
        read(&help, None, false)?,
        Invocation::Special(Special::Help)
    ));
    assert!(matches!(
        read(&compact, None, false)?,
        Invocation::Ordinary(Mode::Command(Command::Compact))
    ));
    assert!(matches!(
        read(&[], None, true)?,
        Invocation::Ordinary(Mode::Interactive)
    ));
    assert!(read(&words("/frobnicate"), None, false).is_err());
    Ok(())
}
