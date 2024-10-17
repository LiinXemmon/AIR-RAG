CONFIG_DIR="config/gasketrag"
for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
    echo "Executing configuration: $CONFIG_FILE"
    pixi run python ./main-evaluation.py --config "$CONFIG_FILE"
done