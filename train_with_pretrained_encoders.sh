for cfg in appearance bbox_only keypoints default; do
  uv run tools/train.py --config-name="$cfg" train.truncate=true
done
