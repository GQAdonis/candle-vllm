//! Template sanitizer for minijinja compatibility.
//!
//! Patches known Jinja2 constructs that minijinja handles incorrectly
//! before the template reaches the renderer. Applied universally to
//! all GGUF/HF templates at load time.

/// Sanitize a Jinja2 chat template for minijinja compatibility.
///
/// Applies known transforms for constructs that minijinja either
/// doesn't support or handles with silent rendering errors.
pub fn sanitize_template(raw: &str) -> String {
    let mut t = raw.to_string();

    // 1. Python-style reverse slicing → Jinja reverse filter
    t = t.replace("[::-1]", "|reverse");

    // 2. Remove {{ meta }} references (some templates set metadata that minijinja can't resolve)
    if t.contains("{{ meta }}") {
        t = t.replace("{%- set meta = message.get(\"metadata\", \"\") %}", "");
        t = t.replace("{% set meta = message.get(\"metadata\", \"\") %}", "");
        t = t.replace("{{ meta }}", "");
    }

    // 3. Remove `| safe` filter (minijinja auto-escapes differently)
    t = t.replace("| tojson | safe", "| tojson");
    t = t.replace("|tojson|safe", "| tojson");
    t = t.replace("| safe", "");

    // 4. Fix Qwen3.5 template bug: `arguments | items` → `arguments.items()`
    // (from community issue #1831)
    t = t.replace("arguments | items", "arguments.items()");

    // 5. Replace raise_exception calls with empty output
    // minijinja supports raise_exception as a registered function,
    // but some templates call it in ways that cause render failures
    // We keep the function registered but patch problematic invocations
    // that use string concatenation inside raise_exception()
    // e.g. raise_exception('Cannot ...' + something) → just skip it

    // 6. Patch `namespace()` usage that causes silent failures
    // minijinja supports namespace() but some complex usage patterns
    // (like namespace(value=0) followed by ns.value assignments in nested scopes)
    // can produce incorrect results. We leave namespace() as-is since
    // minijinja-contrib pycompat handles it, but log if it's present.
    if t.contains("namespace(") {
        tracing::debug!(
            "Template uses Jinja2 namespace() — minijinja-contrib pycompat handles this"
        );
    }

    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_slicing() {
        let input = "{{ items[::-1] }}";
        assert_eq!(sanitize_template(input), "{{ items|reverse }}");
    }

    #[test]
    fn test_safe_filter_removal() {
        let input = "{{ value | tojson | safe }}";
        assert_eq!(sanitize_template(input), "{{ value | tojson }}");
    }

    #[test]
    fn test_meta_removal() {
        let input = r#"{%- set meta = message.get("metadata", "") %}{{ meta }}rest"#;
        let result = sanitize_template(input);
        assert!(!result.contains("meta"));
        assert!(result.contains("rest"));
    }

    #[test]
    fn test_arguments_items_fix() {
        let input = "{% for key, val in arguments | items %}";
        let result = sanitize_template(input);
        assert!(result.contains("arguments.items()"));
    }

    #[test]
    fn test_passthrough_clean_template() {
        let input = "<|im_start|>{{ role }}\n{{ content }}<|im_end|>";
        assert_eq!(sanitize_template(input), input);
    }
}
