#!/usr/bin/env bash
# post-generate.sh
# Validates generated models.yaml and prints a summary.
# Run after the configuration wizard produces output.

set -euo pipefail

CONFIG_FILE="${1:-models.yaml}"

echo "================================================"
echo "  candle-vllm Configuration Post-Generate Check"
echo "================================================"
echo ""

# Check that the file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Basic YAML syntax validation
# Try python if available, otherwise fall back to a simple check
if command -v python3 &>/dev/null; then
    if python3 -c "
import yaml, sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        data = yaml.safe_load(f)
    if data is None:
        print('WARNING: Configuration file is empty')
        sys.exit(1)
    print('YAML syntax: VALID')
except yaml.YAMLError as e:
    print(f'YAML syntax: INVALID')
    print(f'Error: {e}')
    sys.exit(1)
" 2>/dev/null; then
        YAML_VALID=true
    else
        YAML_VALID=false
    fi
elif command -v ruby &>/dev/null; then
    if ruby -ryaml -e "YAML.safe_load(File.read('$CONFIG_FILE'))" 2>/dev/null; then
        echo "YAML syntax: VALID"
        YAML_VALID=true
    else
        echo "YAML syntax: INVALID"
        YAML_VALID=false
    fi
else
    # Fallback: basic structural checks
    if grep -q "^models:" "$CONFIG_FILE" || grep -q "^model_id:" "$CONFIG_FILE"; then
        echo "YAML syntax: BASIC CHECK PASSED (install PyYAML for full validation)"
        YAML_VALID=true
    else
        echo "YAML syntax: BASIC CHECK FAILED - missing expected top-level keys"
        YAML_VALID=false
    fi
fi

echo ""

# Print summary of configured models
echo "Configured Models:"
echo "------------------"
if command -v python3 &>/dev/null; then
    python3 -c "
import yaml, sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        data = yaml.safe_load(f)
    if data and 'models' in data:
        for i, model in enumerate(data['models'], 1):
            model_id = model.get('model_id', model.get('source', {}).get('path', 'unknown'))
            device = model.get('device', {}).get('type', 'unknown')
            quant = model.get('quantization', {}).get('level', model.get('quantization', {}).get('method', 'none'))
            print(f'  {i}. {model_id}')
            print(f'     Device: {device} | Quantization: {quant}')
    else:
        print('  No models found in configuration')
except Exception as e:
    print(f'  Could not parse models: {e}')
" 2>/dev/null || echo "  (install PyYAML to see model summary)"
else
    # Fallback: grep for model identifiers
    grep -E "model_id:|path:" "$CONFIG_FILE" 2>/dev/null | sed 's/^/  /' || echo "  (unable to extract model details)"
fi

echo ""
echo "================================================"
echo "  REMINDER: Review the configuration before"
echo "  deploying to production. Verify that:"
echo ""
echo "  - Model paths/IDs are correct"
echo "  - VRAM allocation fits your hardware"
echo "  - Quantization level meets quality needs"
echo "  - KV cache memory is appropriately sized"
echo "================================================"

if [ "$YAML_VALID" = false ]; then
    exit 1
fi

exit 0
