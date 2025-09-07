use crate::Message;
use itertools::Itertools;

#[derive(thiserror::Error, Debug)]
#[error("Model not found")]
pub struct ModelNotFoundError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChatTemplate {
    /// chatml template
    Chatml,

    /// Official mistral 'v7' template
    MistralV7,

    /// Official mistral 'v7' template
    MistralV7Tekken,

    /*MistralV1,
    MistralV3,
    MistralV3Tekken,*/

    /*Llama2,
    Llama2Sys,
    Llama2SysBos,
    Llama2SysStrip,*/
    Phi3,
    Phi4,
    Falcon3,
    Zephyr,
    Monarch,
    Gemma,
    Orion,
    Openchat,
    Vicuna,
    VicunaOrca,
    Deepseek,
    CommandR,
    Llama3,
    Chatglm3,
    Chatglm4,
    Glmedge,
    Minicpm,
    Deepseek2,
    Deepseek3,
    Exaone3,
    Exaone4,
    RwkvWorld,
    Granite,
    //Gigachat,
    Megrez,
    Yandex,
    Bailing,
    Llama4,
    Smolvlm,
    Dots1,
    HunyuanMoe,
    OpenaiMoe,
    HunyuanDense,
    KimiK2,
    SeedOss,
}

use ChatTemplate::*;

pub fn detect(m: &str) -> anyhow::Result<ChatTemplate> {
    if m.contains("chatml") {
        Ok(Chatml)
    }
    /*else if m.contains("llama2-sys-bos")  {Ok(Llama2SysBos)}
    else if m.contains("llama2-sys-strip")  {Ok(Llama2SysStrip)}
    else if m.contains("llama2-sys")     {Ok(Llama2Sys)}
    else if m.contains("llama2")      {Ok(Llama2)}
    else if m.contains("mistral-v1")        {Ok(Mistralv1)}
    else if m.contains("mistral-v3")        {Ok(MistralV3)}
    else if m.contains("mistral-v3-tekken") {Ok(MistralV3Tekken)}*/
    else if m.contains("mistral-v7") {
        Ok(MistralV7)
    } else if m.contains("mistral-v7-tekken") {
        Ok(MistralV7Tekken)
    } else if m.contains("phi3") {
        Ok(Phi3)
    } else if m.contains("phi4") {
        Ok(Phi4)
    } else if m.contains("falcon3") {
        Ok(Falcon3)
    } else if m.contains("zephyr") {
        Ok(Zephyr)
    } else if m.contains("monarch") {
        Ok(Monarch)
    } else if m.contains("gemma") {
        Ok(Gemma)
    } else if m.contains("orion") {
        Ok(Orion)
    } else if m.contains("openchat") {
        Ok(Openchat)
    } else if m.contains("vicuna") {
        Ok(Vicuna)
    } else if m.contains("vicuna-orca") {
        Ok(VicunaOrca)
    } else if m.contains("deepseek") {
        Ok(Deepseek)
    } else if m.contains("deepseek2") {
        Ok(Deepseek2)
    } else if m.contains("deepseek3") {
        Ok(Deepseek3)
    } else if m.contains("command-r") {
        Ok(CommandR)
    } else if m.contains("llama3") {
        Ok(Llama3)
    } else if m.contains("chatglm3") {
        Ok(Chatglm3)
    } else if m.contains("chatglm4") {
        Ok(Chatglm4)
    } else if m.contains("Tulu") || m.contains("glmedge") {
        Ok(Glmedge)
    } else if m.contains("minicpm") {
        Ok(Minicpm)
    } else if m.contains("exaone3") {
        Ok(Exaone3)
    } else if m.contains("exaone4") {
        Ok(Exaone4)
    } else if m.contains("rwkv-world") {
        Ok(RwkvWorld)
    } else if m.contains("granite") {
        Ok(Granite)
    /*} else if m.contains("gigachat") {
    Ok(Gigachat)*/
    } else if m.contains("megrez") {
        Ok(Megrez)
    } else if m.contains("yandex") {
        Ok(Yandex)
    } else if m.contains("bailing") {
        Ok(Bailing)
    } else if m.contains("llama4") {
        Ok(Llama4)
    } else if m.contains("smolvlm") {
        Ok(Smolvlm)
    } else if m.contains("hunyuan-moe") {
        Ok(HunyuanMoe)
    } else if m.contains("gpt-oss") {
        Ok(OpenaiMoe)
    } else if m.contains("hunyuan-dense") {
        Ok(HunyuanDense)
    } else if m.contains("kimi-k2") {
        Ok(KimiK2)
    } else if m.contains("seed_oss") {
        Ok(SeedOss)
    } else {
        Err(ModelNotFoundError.into())
    }
}

/// Simple version of "llama_apply_chat_template" that only works with strings
/// This function uses heuristic checks to determine commonly used template. It is not a jinja parser.
pub fn apply(tmpl: ChatTemplate, chat: &[Message], add_ass: bool) -> String {
    // Taken from the research: https://github.com/ggerganov/llama.cpp/issues/5527
    match tmpl {
        Chatml => chat
            .iter()
            .map(|m| format!("<|im_start|>{}\n{}<|im_end|>", m.role(), m.content()))
            .chain(
                add_ass
                    .then_some(["<|im_start|>assistant".into()])
                    .into_iter()
                    .flatten(),
            )
            .join("\n"),

        MistralV7 | MistralV7Tekken => {
            // Official mistral 'v7' template
            // See: https://huggingface.co/mistralai/Mistral-Large-Instruct-2411#basic-instruct-template-v7
            //      https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503#basic-instruct-template-v7-tekken
            let trailing_space = match tmpl {
                MistralV7 => " ",
                _ => "",
            };
            chat.iter()
                .map(|m| {
                    let content = m.content();
                    let (prefix, suffix) = match m {
                        Message::System(_) => ("[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]"),
                        Message::User(_) => ("[INST]", "[/INST]"),
                        _ => ("", "</s>"),
                    };

                    format!("{prefix}{trailing_space}{content}{suffix}")
                })
                .join("")
        }
        /*MISTRAL_V1 | LLM_CHAT_TEMPLATE_MISTRAL_V3 | LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN => {
            // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/chat_templates.md
            // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/templates.md
            std::string leading_space = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V1 ? " " : "";
            std::string trailing_space = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN ? "" : " ";
            bool trim_assistant_message = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3;
            bool is_inside_turn = false;
            for (auto message : chat) {
                if (!is_inside_turn) {
                    ss << leading_space << "[INST]" << trailing_space;
                    is_inside_turn = true;
                }
                std::string role(message->role);
                std::string content(message->content);
                if (role == "system") {
                    ss << content << "\n\n";
                } else if (role == "user") {
                    ss << content << leading_space << "[/INST]";
                } else {
                    ss << trailing_space << (trim_assistant_message ? trim(content) : content) << "</s>";
                    is_inside_turn = false;
                }
            }
        }*/
        /*LLM_CHAT_TEMPLATE_LLAMA_2 | LLM_CHAT_TEMPLATE_LLAMA_2_SYS | LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS | LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP => {
            // llama2 template and its variants
            // [variant] support system message
            // See: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            bool support_system_message = tmpl != LLM_CHAT_TEMPLATE_LLAMA_2;
            // [variant] add BOS inside history
            bool add_bos_inside_history = tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS;
            // [variant] trim spaces from the input message
            bool strip_message = tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP;
            // construct the prompt
            bool is_inside_turn = true; // skip BOS at the beginning
            ss << "[INST] ";
            for (auto message : chat) {
                std::string content = strip_message ? trim(message->content) : message->content;
                std::string role(message->role);
                if (!is_inside_turn) {
                    is_inside_turn = true;
                    ss << (add_bos_inside_history ? "<s>[INST] " : "[INST] ");
                }
                if (role == "system") {
                    if (support_system_message) {
                        ss << "<<SYS>>\n" << content << "\n<</SYS>>\n\n";
                    } else {
                        // if the model does not support system message, we still include it in the first message, but without <<SYS>>
                        ss << content << "\n";
                    }
                } else if (role == "user") {
                    ss << content << " [/INST]";
                } else {
                    ss << content << "</s>";
                    is_inside_turn = false;
                }
            }
        }*/
        Phi3 => chat
            .iter()
            .map(|m| format!("<|{}|>\n{}<|end|>", m.role(), m.content()))
            .chain(
                add_ass
                    .then_some(["<|assistant|>".into()])
                    .into_iter()
                    .flatten(),
            )
            .join("\n"),

        Phi4 => chat
            .iter()
            .map(|m| {
                format!(
                    "<|im_start|>{}<|im_sep|>{}<|im_end|>",
                    m.role(),
                    m.content()
                )
            })
            .chain(
                add_ass
                    .then_some(["<|im_start|>assistant<|im_sep|>".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        Falcon3 => chat
            .iter()
            .map(|m| format!("<|{}|>\n{}", m.role(), m.content()))
            .chain(
                add_ass
                    .then_some(["<|assistant|>".into()])
                    .into_iter()
                    .flatten(),
            )
            .join("\n"),

        Zephyr => chat
            .iter()
            .map(|m| format!("<|{}|>\n{}<|endoftext|>", m.role(), m.content()))
            .chain(
                add_ass
                    .then_some(["<|assistant|>".into()])
                    .into_iter()
                    .flatten(),
            )
            .join("\n"),

        // mlabonne/AlphaMonarch-7B template (the <s> is included inside history)
        Monarch => chat
            .iter()
            .enumerate()
            .map(|(idx, m)| {
                format!(
                    "{}{}\n{}</s>",
                    if idx == 0 { "" } else { "<s>" },
                    m.role(),
                    m.content()
                )
            })
            .chain(
                add_ass
                    .then_some(["<s>assistant".into()])
                    .into_iter()
                    .flatten(),
            )
            .join("\n"),

        // TODO: there is no system message for gemma... merge it with user message rather than having a separate user message for the system message
        Gemma => chat
            .iter()
            .map(|m| {
                format!(
                    "<start_of_turn>{}\n{}<end_of_turn>",
                    // in gemma, "assistant" is "model"
                    match m {
                        Message::Assistant(_) => "model",
                        Message::System(_) => "user",
                        _ => m.role(),
                    },
                    m.content().trim(),
                )
            })
            .chain(
                add_ass
                    .then_some(["<start_of_turn>model".into()])
                    .into_iter()
                    .flatten(),
            )
            .join("\n"),

        // OrionStarAI/Orion-14B-Chat
        // TODO: there is no system message... merge it with user message rather than having a separate user message for the system message
        Orion => chat
            .iter()
            .map(|m| match m {
                Message::System(s) | Message::User(s) => format!("Human: {s}\n\nAssistant: </s>"),
                Message::Assistant(s) => format!("{s}</s>"),
            })
            .join(""),

        // openchat/openchat-3.5-0106,
        Openchat => chat
            .iter()
            .map(|m| match m {
                Message::System(s) => format!("{s}<|end_of_turn|>"),
                Message::User(s) => format!("GPT4 Correct User: {s}<|end_of_turn|>"),
                Message::Assistant(s) => format!("GPT4 Correct Assistant: {s}<|end_of_turn|>"),
            })
            .chain(
                add_ass
                    .then_some(["GPT4 Correct Assistant:".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // eachadea/vicuna-13b-1.1 (and Orca variant)
        Vicuna | VicunaOrca => chat
            .iter()
            .map(|m| {
                match m {
                    Message::System(s) => {
                        // Orca-Vicuna variant uses a system prefix
                        match tmpl {
                            VicunaOrca => format!("SYSTEM: {s}"),
                            _ => s.to_string(),
                        }
                    }
                    Message::User(s) => format!("USER: {s}"),
                    Message::Assistant(s) => format!("ASSISTANT: {s}</s>"),
                }
            })
            .chain(
                add_ass
                    .then_some(["ASSISTANT:".into()])
                    .into_iter()
                    .flatten(),
            )
            .join("\n"),

        // deepseek-ai/deepseek-coder-33b-instruct
        Deepseek => chat
            .iter()
            .map(|m| {
                match m {
                    Message::System(s) => s.to_string(), // hmm no newline?
                    Message::User(s) => format!("### Instruction:\n{s}"),
                    Message::Assistant(s) => format!("### Response:\n{s}n<|EOT|>"),
                }
            })
            .chain(
                add_ass
                    .then_some(["### Response:".into()])
                    .into_iter()
                    .flatten(),
            )
            .join("\n"),

        // CohereForAI/c4ai-command-r-plus
        CommandR => chat
            .iter()
            .map(|m| match m {
                Message::System(s) => format!(
                    "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{}<|END_OF_TURN_TOKEN|>",
                    s.trim()
                ),
                Message::User(s) => format!(
                    "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{}<|END_OF_TURN_TOKEN|>",
                    s.trim()
                ),
                Message::Assistant(s) => format!(
                    "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{}<|END_OF_TURN_TOKEN|>",
                    s.trim()
                ),
            })
            .chain(
                add_ass
                    .then_some(["<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // Llama 3
        Llama3 => chat
            .iter()
            .map(|m| {
                format!(
                    "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                    m.role(),
                    m.content().trim()
                )
            })
            .chain(
                add_ass
                    .then_some(["<|start_header_id|>assistant<|end_header_id|>\n\n".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // chatglm3-6b
        Chatglm3 => ["[gMASK]sop".to_string()]
            .into_iter()
            .chain(
                chat.iter()
                    .map(|m| format!("<|{}|>\n{}", m.role(), m.content())),
            )
            .chain(
                add_ass
                    .then_some(["<|assistant|>".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        Chatglm4 => ["[gMASK]sop".to_string()]
            .into_iter()
            .chain(
                chat.iter()
                    .map(|m| format!("<|{}|>\n{}", m.role(), m.content())),
            )
            .chain(
                add_ass
                    .then_some(["<|assistant|>\n".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        Glmedge => chat
            .iter()
            .map(|m| format!("\n<|{}|>\n{}", m.role(), m.content()))
            .chain(
                add_ass
                    .then_some(["\n<|assistant|>\n".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        Minicpm => chat
            .iter()
            .map(|m| match m {
                Message::User(s) => {
                    format!("<{}>{}<AI>", String::from("<用户>"), s.trim())
                }
                _ => m.content().trim().to_string(),
            })
            .join(""),

        // DeepSeek-V2
        Deepseek2 => chat
            .iter()
            .map(|m| match m {
                Message::System(s) => format!("{s}\n\n"),
                Message::User(s) => format!("User: {s}\n\n"),
                Message::Assistant(s) => {
                    format!("Assistant: {s}{}", String::from("<｜end▁of▁sentence｜>"))
                }
            })
            .chain(
                add_ass
                    .then_some(["Assistant:".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // DeepSeek-V3
        Deepseek3 => chat
            .iter()
            .map(|m| match m {
                Message::System(s) => format!("{s}\n\n"),
                Message::User(s) => format!("{}User: {s}\n\n", String::from("<｜User｜>")),
                Message::Assistant(s) => format!(
                    "{}{s}{}",
                    String::from("<｜Assistant｜>"),
                    String::from("<｜end▁of▁sentence｜>")
                ),
            })
            .chain(
                add_ass
                    .then_some(["<｜Assistant｜>".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/discussions/8#66bae61b1893d14ee8ed85bb
        // EXAONE-3.0-7.8B-Instruct
        Exaone3 | Exaone4 => chat
            .iter()
            .map(|m| match m {
                Message::User(s) => format!("[|user|]{}\n", s.trim()),
                _ => format!("[|{}|]{}[|endofturn|]\n", m.role(), m.content().trim()),
            })
            .chain(
                add_ass
                    .then_some(["[|assistant|]".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // this template requires the model to have "\n\n" as EOT token
        RwkvWorld => chat
            .iter()
            .enumerate()
            .map(|(i, m)| match m {
                Message::System(s) => format!("System: {}\n\n", s.trim()),
                Message::User(s) => format!(
                    "User: {}\n\n{}",
                    s.trim(),
                    if i == chat.len() - 1 {
                        "Assistant:"
                    } else {
                        ""
                    }
                ),
                Message::Assistant(s) => format!("Assistant: {}\n\n", s.trim()),
            })
            .join(""),

        // IBM Granite template
        // TODO assistant_tool_call role. add <|tool_call|> after <|end_of_role|>
        Granite => chat
            .iter()
            .map(|m| {
                format!(
                    "<|start_of_role|>{}<|end_of_role|>{}<|end_of_text|>",
                    m.role(),
                    m.content()
                )
            })
            .chain(
                add_ass
                    .then_some(["<|start_of_role|>assistant<|end_of_role|>".into()])
                    .into_iter()
                    .flatten(),
            )
            .join("\n"),

        // GigaChat template
        /*Gigachat =>
        bool has_system = !chat.empty() && std::string(chat[0]->role) == "system";

        // Handle system message if present
        if (has_system) {
            ss << "<s>" << chat[0]->content << "<|message_sep|>";
        } else {
            ss << "<s>";
        }

        // Process remaining messages
        for (size_t i = has_system ? 1 : 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (role == "user") {
                ss << "user<|role_sep|>" << chat[i]->content << "<|message_sep|>"
                << "available functions<|role_sep|>[]<|message_sep|>";
            } else if (role == "assistant") {
                ss << "assistant<|role_sep|>" << chat[i]->content << "<|message_sep|>";
            }
        }

        // Add generation prompt if needed
        if (add_ass) {
            ss << "assistant<|role_sep|>";
        }*/
        // Megrez template
        Megrez => chat
            .iter()
            .map(|m| {
                format!(
                    "<|role_start|>{}<|role_end|>{}<|turn_end|>",
                    m.role(),
                    m.content()
                )
            })
            .chain(
                add_ass
                    .then_some(["<|role_start|>assistant<|role_end|>".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // Yandex template ("\n\n" is defined as EOT token)
        // TODO system role?
        Yandex => chat
            .iter()
            .map(|m| match m {
                Message::System(s) | Message::User(s) => format!(" Пользователь: {s}\n\n"),
                Message::Assistant(s) => format!(" Ассистент: {s}\n\n"),
            })
            .chain(
                add_ass
                    .then_some([" Ассистент:[SEP]".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // Bailing (Ling) template
        Bailing => chat
            .iter()
            .map(|m| match m {
                Message::System(s) => format!("<role>SYSTEM</role>{s}"),
                Message::User(s) => format!("<role>HUMAN</role>{s}"),
                Message::Assistant(s) => format!("<role>ASSISTANT</role>{s}"),
            })
            .chain(
                add_ass
                    .then_some(["<role>ASSISTANT</role>".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // Llama 4
        Llama4 => chat
            .iter()
            .map(|m| {
                {
                    format!(
                        "<|header_start|>{}<|header_end|>\n\n{}<|eot|>",
                        m.role(),
                        m.content().trim()
                    )
                }
            })
            .chain(
                add_ass
                    .then_some(["<|header_start|>assistant<|header_end|>\n\n".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // SmolVLM
        Smolvlm => {
            ["<|im_start|>".to_string()] // uses <|im_start|> as BOS, but the actual content is NOT chatml
                .into_iter()
                .chain(chat.iter().map(|m| match m {
                    Message::System(s) => format!("{s}\n\n"),
                    Message::User(s) => format!("User: {s}<end_of_utterance>\n"),
                    Message::Assistant(s) => format!("Assistant: {s}<end_of_utterance>\n"),
                }))
                .chain(
                    add_ass
                        .then_some(["Assistant:".into()])
                        .into_iter()
                        .flatten(),
                )
                .join("")
        }

        // dots.llm1.inst (DOTS1)
        Dots1 => chat
            .iter()
            .map(|m| match m {
                Message::System(s) => format!("<|system|>{s}<|endofsystem|>"),
                Message::User(s) => format!("<|userprompt|>{s}<|endofuserprompt|>"),
                Message::Assistant(s) => format!("<|response|>{s}<|endofresponse|>"),
            })
            .chain(
                add_ass
                    .then_some(["<|response|>".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // tencent/Hunyuan-A13B-Instruct
        HunyuanMoe => chat
            .iter()
            .map(|m| match m {
                Message::System(s) => format!("<|startoftext|>{s}<|extra_4|>"),
                Message::Assistant(s) => format!("{s}<|eos|>"),
                Message::User(s) => format!("<|startoftext|>{s}<|extra_0|>"),
            })
            .join(""),

        // OpenAI MoE (based on Harmony chat template)
        OpenaiMoe => chat
            .iter()
            .map(|m| {
                format!(
                    "<|start|>{}<|message|>{}{}",
                    m.role(),
                    m.content(),
                    match m {
                        Message::Assistant(_) => "<|return|>",
                        _ => "<|end|>",
                    }
                )
            })
            .chain(
                add_ass
                    .then_some(["<|start|>assistant".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        // tencent/Hunyuan-4B-Instruct
        HunyuanDense => chat
            .iter()
            .enumerate()
            .map(|(i, m)| {
                match (i, m) {
                    (0, Message::System(s)) => format!("{s}<｜hy_place▁holder▁no▁3｜>"),
                    (_, Message::System(_)) => String::new(), // TODO issue warning? error?
                    (_, Message::Assistant(s)) => {
                        format!("<｜hy_Assistant｜>{s}<｜hy_place▁holder▁no▁2｜>")
                    }
                    (_, Message::User(s)) => format!(
                        "<｜hy_User｜>{s}{}",
                        if i == chat.len() - 1 {
                            "<｜hy_Assistant｜>"
                        } else {
                            ""
                        }
                    ),
                }
            })
            .join(""),

        // moonshotai/Kimi-K2-Instruct
        // TOOL tool role
        KimiK2 => chat
            .iter()
            .map(|m| {
                format!(
                    "<|im_{}|>{}<|im_middle|>{}<|im_end|>",
                    m.role(),
                    m.role(),
                    m.content()
                )
            })
            .chain(
                add_ass
                    .then_some(["<|im_assistant|>assistant<|im_middle|>".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),

        SeedOss => chat
            .iter()
            .map(|m| {
                format!(
                    "<seed:bos>{}\n{}<seed:eos>",
                    m.role(),
                    match m {
                        Message::Assistant(s) => s.trim().to_string(),
                        _ => m.content(),
                    }
                )
            })
            .chain(
                add_ass
                    .then_some(["<seed:bos>assistant\n".into()])
                    .into_iter()
                    .flatten(),
            )
            .join(""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_template() -> anyhow::Result<()> {
        assert_eq!(detect("llama3")?, Llama3);
        assert_eq!(detect("google/gemma-3-270m")?, Gemma);

        assert_eq!(detect("granite")?, Granite);
        assert_eq!(detect("ibm-granite/granite-3.3-8b-instruct")?, Granite);
        Ok(())
    }

    #[test]
    fn apply_template() {
        let chat = &[Message::User("hi".into())];
        assert_eq!(
            apply(Granite, chat, false),
            "<|start_of_role|>user<|end_of_role|>hi<|end_of_text|>"
        );
        assert_eq!(
            apply(Granite, chat, true),
            "<|start_of_role|>user<|end_of_role|>hi<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        );
        assert_eq!(
            apply(Glmedge, chat, true),
            "\n<|user|>\nhi\n<|assistant|>\n"
        );
    }
}
