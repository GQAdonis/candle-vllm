//! Hardcoded fallback chat templates for known model families.
//!
//! These are used when the GGUF/HF template fails validation or
//! is absent. Keyed on architecture string (not model name) so
//! they apply to all variants in a family.

/// Get a known-good fallback chat template for a model architecture.
///
/// Returns `None` if the architecture is not recognized.
pub fn get_fallback_template(arch: &str) -> Option<&'static str> {
    let arch_lower = arch.to_lowercase();

    // ChatML family (Qwen, Yi, DeepSeek, generic ChatML)
    if arch_lower.contains("qwen")
        || arch_lower.contains("yi")
        || arch_lower.contains("deepseek")
        || arch_lower.contains("chatml")
    {
        return Some(CHATML_TEMPLATE);
    }

    // Llama 3 family
    if arch_lower.contains("llama3") || arch_lower == "llama" {
        return Some(LLAMA3_TEMPLATE);
    }

    // Mistral family
    if arch_lower.contains("mistral") || arch_lower.contains("ministral") {
        return Some(MISTRAL_TEMPLATE);
    }

    // Phi family
    if arch_lower.contains("phi") {
        return Some(PHI_TEMPLATE);
    }

    // Gemma family
    if arch_lower.contains("gemma") {
        return Some(GEMMA_TEMPLATE);
    }

    // GLM family
    if arch_lower.contains("glm") {
        return Some(GLM_TEMPLATE);
    }

    None
}

/// Get the expected role marker tokens for a model family.
/// Used for template validation — if the rendered output doesn't
/// contain these markers, the template is broken.
pub fn get_expected_markers(arch: &str) -> &'static [&'static str] {
    let arch_lower = arch.to_lowercase();

    if arch_lower.contains("qwen")
        || arch_lower.contains("yi")
        || arch_lower.contains("deepseek")
        || arch_lower.contains("chatml")
    {
        return &["<|im_start|>", "<|im_end|>"];
    }

    if arch_lower.contains("llama3") || arch_lower == "llama" {
        return &["<|start_header_id|>", "<|end_header_id|>"];
    }

    if arch_lower.contains("mistral") || arch_lower.contains("ministral") {
        return &["[INST]", "[/INST]"];
    }

    if arch_lower.contains("phi") {
        return &["<|user|>", "<|end|>"];
    }

    if arch_lower.contains("gemma") {
        return &["<start_of_turn>", "<end_of_turn>"];
    }

    if arch_lower.contains("glm") {
        return &["<|user|>", "<|assistant|>"];
    }

    // Unknown family — no expected markers
    &[]
}

// ============================================================================
// Template Definitions
// ============================================================================

/// ChatML template (Qwen, Yi, DeepSeek, generic).
/// Supports thinking mode via `enable_thinking` variable.
const CHATML_TEMPLATE: &str = r#"{%- for message in messages %}
{%- if message.role == "system" %}
<|im_start|>system
{{ message.content }}<|im_end|>
{% elif message.role == "user" %}
<|im_start|>user
{{ message.content }}<|im_end|>
{% elif message.role == "assistant" %}
<|im_start|>assistant
{{ message.content }}<|im_end|>
{% elif message.role == "tool" %}
<|im_start|>tool
{{ message.content }}<|im_end|>
{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{% if enable_thinking is defined and enable_thinking %}<think>
{% endif %}
{%- endif %}"#;

/// Llama 3 template.
const LLAMA3_TEMPLATE: &str = r#"<|begin_of_text|>{%- for message in messages %}
{%- if message.role == "system" %}
<|start_header_id|>system<|end_header_id|>

{{ message.content }}<|eot_id|>
{%- elif message.role == "user" %}
<|start_header_id|>user<|end_header_id|>

{{ message.content }}<|eot_id|>
{%- elif message.role == "assistant" %}
<|start_header_id|>assistant<|end_header_id|>

{{ message.content }}<|eot_id|>
{%- elif message.role == "tool" %}
<|start_header_id|>ipython<|end_header_id|>

{{ message.content }}<|eot_id|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}"#;

/// Mistral template.
const MISTRAL_TEMPLATE: &str = r#"{%- for message in messages %}
{%- if message.role == "user" %}
[INST] {{ message.content }} [/INST]
{%- elif message.role == "assistant" %}
{{ message.content }}</s>
{%- elif message.role == "system" %}
[INST] {{ message.content }}

{% endif %}
{%- endfor %}"#;

/// Phi template.
const PHI_TEMPLATE: &str = r#"{%- for message in messages %}
{%- if message.role == "system" %}
<|system|>
{{ message.content }}<|end|>
{%- elif message.role == "user" %}
<|user|>
{{ message.content }}<|end|>
{%- elif message.role == "assistant" %}
<|assistant|>
{{ message.content }}<|end|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|assistant|>
{% endif %}"#;

/// Gemma template.
const GEMMA_TEMPLATE: &str = r#"{%- for message in messages %}
{%- if message.role == "user" %}
<start_of_turn>user
{{ message.content }}<end_of_turn>
{%- elif message.role == "assistant" %}
<start_of_turn>model
{{ message.content }}<end_of_turn>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<start_of_turn>model
{% endif %}"#;

/// GLM template.
const GLM_TEMPLATE: &str = r#"[gMASK]<sop>{%- for message in messages %}
{%- if message.role == "system" %}
<|system|>
{{ message.content }}
{%- elif message.role == "user" %}
<|user|>
{{ message.content }}
{%- elif message.role == "assistant" %}
<|assistant|>
{{ message.content }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|assistant|>
{% endif %}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_family_lookup() {
        assert!(get_fallback_template("qwen35").is_some());
        assert!(get_fallback_template("qwen36").is_some());
        assert!(get_fallback_template("Qwen3_5ForCausalLM").is_some());
        assert!(get_fallback_template("Qwen3_6ForCausalLM").is_some());
        assert!(get_fallback_template("qwen2").is_some());
        assert!(get_fallback_template("qwen3").is_some());
    }

    #[test]
    fn test_llama_family_lookup() {
        assert!(get_fallback_template("llama").is_some());
        assert!(get_fallback_template("llama3").is_some());
    }

    #[test]
    fn test_mistral_family_lookup() {
        assert!(get_fallback_template("mistral").is_some());
    }

    #[test]
    fn test_unknown_family() {
        assert!(get_fallback_template("unknown_model_xyz").is_none());
    }

    #[test]
    fn test_expected_markers_chatml() {
        let markers = get_expected_markers("qwen35");
        assert!(markers.contains(&"<|im_start|>"));
        assert!(markers.contains(&"<|im_end|>"));
    }

    #[test]
    fn test_expected_markers_llama() {
        let markers = get_expected_markers("llama3");
        assert!(markers.contains(&"<|start_header_id|>"));
    }

    #[test]
    fn test_chatml_template_has_markers() {
        assert!(CHATML_TEMPLATE.contains("<|im_start|>"));
        assert!(CHATML_TEMPLATE.contains("<|im_end|>"));
        assert!(CHATML_TEMPLATE.contains("enable_thinking"));
    }
}
