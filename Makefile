.PHONY: test lint audit smoke-camera

test:
	python -m pytest

lint:
	python -m ruff check .

audit:
	python tools/audit_public_release.py

smoke-camera:
	recap_smoke_dir=$$(mktemp -d); \
	python pipeline/10_summarize_camera_activity.py \
		--image-manifest examples/synthetic/image_manifest.csv \
		--detections examples/synthetic/detections.csv \
		--output-dir $$recap_smoke_dir --mode recommended
