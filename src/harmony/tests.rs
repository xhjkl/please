#![cfg(test)]

use crate::harmony::*;

fn header(role: &str, channel: &str, recipient: &str) -> HarmonyHeader {
    HarmonyHeader {
        role: role.to_string(),
        channel: channel.to_string(),
        recipient: recipient.to_string(),
    }
}

#[test]
fn test_header_parsing() {
    let cases = [
        ("assistant<|channel|>analysis", "assistant", "analysis", ""),
        (
            "assistant<|channel|>analysis to=functions.get_weather",
            "assistant",
            "analysis",
            "functions.get_weather",
        ),
        (
            "assistant to=functions.get_weather<|channel|>analysis",
            "assistant",
            "analysis",
            "functions.get_weather",
        ),
        (
            "to=functions.get_weather<|channel|>analysis",
            "tool",
            "analysis",
            "functions.get_weather",
        ),
        (
            "assistant to=functions.get_weather abc<|channel|>analysis",
            "assistant",
            "analysis",
            "functions.get_weather",
        ),
        (
            "assistant<|channel|>commentary to=functions.get_weather <|constrain|>json",
            "assistant",
            "commentary",
            "functions.get_weather",
        ),
        (
            "assistant to=functions.get_weather<|channel|>commentary <|constrain|>json",
            "assistant",
            "commentary",
            "functions.get_weather",
        ),
        (
            "assistant<|channel|>commentary to=functions.get_weather<|constrain|>json",
            "assistant",
            "commentary",
            "functions.get_weather",
        ),
        (
            "assistant to=functions.get_weather<|channel|>commentary<|constrain|>json",
            "assistant",
            "commentary",
            "functions.get_weather",
        ),
    ];

    for (i, (input, want_role, want_channel, want_recipient)) in cases.into_iter().enumerate() {
        let mut p = HarmonyParser::new();
        let got = HarmonyParser::parse_header(input);
        assert_eq!(got.role, want_role, "case {} role", i);
        assert_eq!(got.channel, want_channel, "case {} channel", i);
        assert_eq!(got.recipient, want_recipient, "case {} recipient", i);
        // Ensure p used to avoid warnings
        let _ = &mut p;
    }
}

#[test]
fn test_harmony_parser_header_event() {
    let cases = [
        (
            "<|start|>user<|message|>What is 2 + 2?<|end|>",
            "user",
            "",
            "",
            false,
        ),
        (
            "<|start|>assistant<|channel|>analysis<|message|>What is 2 + 2?<|end|>",
            "assistant",
            "analysis",
            "",
            false,
        ),
        (
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"San Francisco\"}<|call|><|start|>functions.get_weather to=assistant<|message|>{\"sunny\": true, \"temperature\": 20}<|end|>",
            "assistant",
            "commentary",
            "functions.get_weather",
            false,
        ),
        (
            "<|channel|>analysis<|message|>User asks weather in SF. We need location. Use get_current_weather with location \"San Francisco, CA\".<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{\"location\":\"San Francisco, CA\"}<|call|>",
            "assistant",
            "analysis",
            "",
            true,
        ),
    ];
    for (i, (input, want_role, want_channel, want_recipient, implicit)) in
        cases.into_iter().enumerate()
    {
        let mut p = HarmonyParser::new();
        if implicit {
            p.add_implicit_start();
        }
        let events = p.add_content(input);
        assert!(!events.is_empty(), "case {} no events", i);
        let mut found: Option<HarmonyHeader> = None;
        for ev in events {
            if let HarmonyEvent::HeaderComplete { header } = ev {
                found = Some(header);
                break;
            }
        }
        let got = found.expect("header event");
        assert_eq!(got.role, want_role, "case {} role", i);
        assert_eq!(got.channel, want_channel, "case {} channel", i);
        assert_eq!(got.recipient, want_recipient, "case {} recipient", i);
    }
}

#[test]
fn test_harmony_parser_non_streaming() {
    let cases: &[(&str, bool, Vec<HarmonyEvent>)] = &[
        (
            "<|start|>user<|message|>What is 2 + 2?<|end|>",
            false,
            vec![
                HarmonyEvent::MessageStart,
                HarmonyEvent::HeaderComplete {
                    header: header("user", "", ""),
                },
                HarmonyEvent::ContentEmitted {
                    content: "What is 2 + 2?".to_string(),
                },
                HarmonyEvent::MessageEnd,
            ],
        ),
        (
            "<|start|>assistant<|channel|>analysis<|message|>The answer is 4<|end|>",
            false,
            vec![
                HarmonyEvent::MessageStart,
                HarmonyEvent::HeaderComplete {
                    header: header("assistant", "analysis", ""),
                },
                HarmonyEvent::ContentEmitted {
                    content: "The answer is 4".to_string(),
                },
                HarmonyEvent::MessageEnd,
            ],
        ),
        (
            "<|start|>assistant<|channel|>commentary to=functions.calc<|message|>Computing...<|end|>",
            false,
            vec![
                HarmonyEvent::MessageStart,
                HarmonyEvent::HeaderComplete {
                    header: header("assistant", "commentary", "functions.calc"),
                },
                HarmonyEvent::ContentEmitted {
                    content: "Computing...".to_string(),
                },
                HarmonyEvent::MessageEnd,
            ],
        ),
        (
            "<|start|>user<|message|><|end|>",
            false,
            vec![
                HarmonyEvent::MessageStart,
                HarmonyEvent::HeaderComplete {
                    header: header("user", "", ""),
                },
                HarmonyEvent::MessageEnd,
            ],
        ),
        (
            "<|start|>user<|message|>Hello<|end|><|start|>assistant<|message|>Hi!<|end|>",
            false,
            vec![
                HarmonyEvent::MessageStart,
                HarmonyEvent::HeaderComplete {
                    header: header("user", "", ""),
                },
                HarmonyEvent::ContentEmitted {
                    content: "Hello".to_string(),
                },
                HarmonyEvent::MessageEnd,
                HarmonyEvent::MessageStart,
                HarmonyEvent::HeaderComplete {
                    header: header("assistant", "", ""),
                },
                HarmonyEvent::ContentEmitted {
                    content: "Hi!".to_string(),
                },
                HarmonyEvent::MessageEnd,
            ],
        ),
        (
            "<|channel|>analysis<|message|>Thinking about the request<|end|>",
            true,
            vec![
                HarmonyEvent::MessageStart,
                HarmonyEvent::HeaderComplete {
                    header: header("assistant", "analysis", ""),
                },
                HarmonyEvent::ContentEmitted {
                    content: "Thinking about the request".to_string(),
                },
                HarmonyEvent::MessageEnd,
            ],
        ),
    ];

    for (i, (input, implicit, want)) in cases.iter().enumerate() {
        let mut p = HarmonyParser::new();
        if *implicit {
            p.add_implicit_start();
        }
        let got = p.add_content(input);
        assert_eq!(&got, want, "case {}", i);
    }
}

#[test]
fn test_harmony_parser_streaming() {
    struct Step {
        input: &'static str,
        want: Vec<HarmonyEvent>,
    }
    let cases: &[(&str, bool, &[Step])] = &[
        (
            "simple message streamed character by character",
            false,
            &[
                Step {
                    input: "<",
                    want: vec![],
                },
                Step {
                    input: "|",
                    want: vec![],
                },
                Step {
                    input: "start|>u",
                    want: vec![HarmonyEvent::MessageStart],
                },
                Step {
                    input: "ser<|mess",
                    want: vec![],
                },
                Step {
                    input: "age|>Hi",
                    want: vec![
                        HarmonyEvent::HeaderComplete {
                            header: header("user", "", ""),
                        },
                        HarmonyEvent::ContentEmitted {
                            content: "Hi".to_string(),
                        },
                    ],
                },
                Step {
                    input: " there",
                    want: vec![HarmonyEvent::ContentEmitted {
                        content: " there".to_string(),
                    }],
                },
                Step {
                    input: "<|e",
                    want: vec![],
                },
                Step {
                    input: "nd|>",
                    want: vec![HarmonyEvent::MessageEnd],
                },
            ],
        ),
        (
            "message with channel streamed",
            false,
            &[
                Step {
                    input: "<|start|>assistant",
                    want: vec![HarmonyEvent::MessageStart],
                },
                Step {
                    input: "<|chan",
                    want: vec![],
                },
                Step {
                    input: "nel|>analysis",
                    want: vec![],
                },
                Step {
                    input: "<|message|>",
                    want: vec![HarmonyEvent::HeaderComplete {
                        header: header("assistant", "analysis", ""),
                    }],
                },
                Step {
                    input: "Thinking",
                    want: vec![HarmonyEvent::ContentEmitted {
                        content: "Thinking".to_string(),
                    }],
                },
                Step {
                    input: "...",
                    want: vec![HarmonyEvent::ContentEmitted {
                        content: "...".to_string(),
                    }],
                },
                Step {
                    input: "<|end|>",
                    want: vec![HarmonyEvent::MessageEnd],
                },
            ],
        ),
        (
            "message with channel and recipient",
            false,
            &[
                Step {
                    input: "<|start|>assistant<|channel|>commentary to=functions.calc<|message|>",
                    want: vec![
                        HarmonyEvent::MessageStart,
                        HarmonyEvent::HeaderComplete {
                            header: header("assistant", "commentary", "functions.calc"),
                        },
                    ],
                },
                Step {
                    input: "{\"x\": 5}",
                    want: vec![HarmonyEvent::ContentEmitted {
                        content: "{\"x\": 5}".to_string(),
                    }],
                },
                Step {
                    input: "<|end|>",
                    want: vec![HarmonyEvent::MessageEnd],
                },
            ],
        ),
        (
            "message with channel and recipient (receipient before channel)",
            false,
            &[
                Step {
                    input: "<|start|>assistant to=functions.calc<|channel|>commentary<|message|>",
                    want: vec![
                        HarmonyEvent::MessageStart,
                        HarmonyEvent::HeaderComplete {
                            header: header("assistant", "commentary", "functions.calc"),
                        },
                    ],
                },
                Step {
                    input: "{\"x\": 5}",
                    want: vec![HarmonyEvent::ContentEmitted {
                        content: "{\"x\": 5}".to_string(),
                    }],
                },
                Step {
                    input: "<|end|>",
                    want: vec![HarmonyEvent::MessageEnd],
                },
            ],
        ),
        (
            "implicit start with channel",
            true,
            &[
                Step {
                    input: "<|channel|>thinking",
                    want: vec![HarmonyEvent::MessageStart],
                },
                Step {
                    input: "<|message|>",
                    want: vec![HarmonyEvent::HeaderComplete {
                        header: header("assistant", "thinking", ""),
                    }],
                },
                Step {
                    input: "Processing request",
                    want: vec![HarmonyEvent::ContentEmitted {
                        content: "Processing request".to_string(),
                    }],
                },
                Step {
                    input: "<|end|>",
                    want: vec![HarmonyEvent::MessageEnd],
                },
            ],
        ),
        (
            "multiple messages streamed",
            false,
            &[
                Step {
                    input: "<|start|>user<|message|>Hello<|end|>",
                    want: vec![
                        HarmonyEvent::MessageStart,
                        HarmonyEvent::HeaderComplete {
                            header: header("user", "", ""),
                        },
                        HarmonyEvent::ContentEmitted {
                            content: "Hello".to_string(),
                        },
                        HarmonyEvent::MessageEnd,
                    ],
                },
                Step {
                    input: "<|start|>",
                    want: vec![HarmonyEvent::MessageStart],
                },
                Step {
                    input: "assistant<|message|>",
                    want: vec![HarmonyEvent::HeaderComplete {
                        header: header("assistant", "", ""),
                    }],
                },
                Step {
                    input: "Hi!",
                    want: vec![HarmonyEvent::ContentEmitted {
                        content: "Hi!".to_string(),
                    }],
                },
                Step {
                    input: "<|end|>",
                    want: vec![HarmonyEvent::MessageEnd],
                },
            ],
        ),
        (
            "empty message",
            false,
            &[Step {
                input: "<|start|>system<|message|><|end|>",
                want: vec![
                    HarmonyEvent::MessageStart,
                    HarmonyEvent::HeaderComplete {
                        header: header("system", "", ""),
                    },
                    HarmonyEvent::MessageEnd,
                ],
            }],
        ),
        (
            "partial tag that looks like end but isn't",
            false,
            &[
                Step {
                    input: "<|start|>user<|message|>test<|e",
                    want: vec![
                        HarmonyEvent::MessageStart,
                        HarmonyEvent::HeaderComplete {
                            header: header("user", "", ""),
                        },
                        HarmonyEvent::ContentEmitted {
                            content: "test".to_string(),
                        },
                    ],
                },
                Step {
                    input: "xample|>more",
                    want: vec![HarmonyEvent::ContentEmitted {
                        content: "<|example|>more".to_string(),
                    }],
                },
                Step {
                    input: "<|end|>",
                    want: vec![HarmonyEvent::MessageEnd],
                },
            ],
        ),
    ];

    for (desc, implicit, steps) in cases {
        let mut p = HarmonyParser::new();
        if *implicit {
            p.add_implicit_start();
        }
        for (i, s) in steps.iter().enumerate() {
            let got = p.add_content(s.input);
            assert_eq!(got, s.want, "{} step {}", desc, i);
        }
    }
}

#[test]
fn test_function_convert_to_valid_chars() {
    let cases = [
        ("get weather", "get_weather"),
        ("get-weather", "get_weather"),
        ("get.weather", "get_weather"),
        ("get weather!", "get_weather"),
        ("a🫠bc", "abc"),
        ("🫠", "unnamed"),
        ("123", "_123"),
        ("$", "$"),
        ("𝓸𝓵𝓵𝓪𝓶𝓪", "𝓸𝓵𝓵𝓪𝓶𝓪"),
        ("ⓞⓛⓛⓐⓜⓐ123", "_123"),
    ];
    for (name, want) in cases {
        let map = FunctionNameMap::new();
        let got = map.convert_to_valid_chars(name);
        assert_eq!(got, want);
    }
}

#[test]
fn test_function_convert_and_add() {
    let tests = vec![
        (
            "basic dupe handling",
            vec!["get weather", "get weather"],
            vec!["get_weather", "get_weather_2"],
        ),
        (
            "dupes from different user-specified names",
            vec!["get weather", "get_weather", "get-weather"],
            vec!["get_weather", "get_weather_2", "get_weather_3"],
        ),
        (
            "non dupes after dupes",
            vec![
                "get weather",
                "get_weather",
                "get-weather",
                "something-different",
            ],
            vec![
                "get_weather",
                "get_weather_2",
                "get_weather_3",
                "something_different",
            ],
        ),
        (
            "multiple sets of dupes",
            vec!["a", "a", "b", "a", "a", "b", "a"],
            vec!["a", "a_2", "b", "a_3", "a_4", "b_2", "a_5"],
        ),
        (
            "built-in functions should not be renamed",
            vec![
                "browser.open",
                "python",
                "not.a.built-in.function",
                "browser.not_a_real_built_in",
            ],
            vec![
                "browser.open",
                "python",
                "not_a_built_in_function",
                "browser_not_a_real_built_in",
            ],
        ),
    ];

    for (_name, input, want) in tests {
        let mut map = FunctionNameMap::new();
        for (i, in_name) in input.iter().enumerate() {
            let got = map.convert_and_add(in_name);
            assert_eq!(got, want[i]);
            assert_eq!(map.user_to_harmony[*in_name], want[i]);
            assert_eq!(map.harmony_to_user[want[i]], *in_name);
        }
    }
}
