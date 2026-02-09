use crate::ir::{Message, Query};
use crate::optimizer::llo::chat_template;

/// Tokenize input with chat template applied
pub fn tokenize_with_chat_template(
    input: &Query,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
) -> Result<Vec<u32>, anyhow::Error> {
    // Detect the chat template for this model
    let template = chat_template::detect(model_name)?;

    // Extract messages from the input
    let messages = extract_messages(input);

    // Apply chat template to get the formatted prompt
    // add_ass=true adds the assistant prompt token at the end
    let formatted_prompt = chat_template::apply(template, &messages, true) + "/no_think";

    // Tokenize the formatted prompt
    let encoding = tokenizer
        .encode(formatted_prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    Ok(encoding.get_ids().to_vec())
}

/// Extract messages from Query
fn extract_messages(input: &Query) -> Vec<Message> {
    match input {
        Query::Message(msg) => vec![msg.clone()],
        Query::Seq(inputs) | Query::Par(inputs) | Query::Cross(inputs) | Query::Plus(inputs) => {
            inputs.iter().flat_map(extract_messages).collect()
        }
        Query::Generate(_) | Query::Bulk(_) | Query::Monad(_) | Query::Zip(_) => vec![],
        #[cfg(feature = "rag")]
        Query::Augment(_) => vec![],
        #[cfg(feature = "print")]
        Query::Print(_) => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_messages() {
        let msg = Message::User("Hello".to_string());
        let input = Query::Message(msg.clone());
        let messages = extract_messages(&input);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], msg);
    }
}

// Made with Bob
