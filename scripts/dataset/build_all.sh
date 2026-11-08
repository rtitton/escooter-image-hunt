. scripts/.env

python scripts/dataset/build_union_dataset.py 

python scripts/dataset/build_union_dataset.py \
    --candidates-file $DATA_ROOT/$FLAGGED_RIDER_FILENAME \
    --out-dir $DATA_ROOT/$RIDER_CONTAMINATED_DIRNAME

python scripts/dataset/build_union_dataset.py \
    --candidates-file $DATA_ROOT/$FLAGGED_AREA_FILENAME \
    --out-dir $DATA_ROOT/$AREA_THRESHOLD_VIOLATIONS_DIRNAME

python scripts/dataset/build_visual_check_sample.py 

python scripts/dataset/build_visual_check_sample.py \
    --data-root $DATA_ROOT/$RIDER_CONTAMINATED_DIRNAME \
    --out-dir $DATA_ROOT/$RIDER_CONTAMINATED_REVIEW_SAMPLE_DIRNAME

python scripts/dataset/build_visual_check_sample.py \
    --data-root $DATA_ROOT/$AREA_THRESHOLD_VIOLATIONS_DIRNAME \
    --out-dir $DATA_ROOT/$AREA_THRESHOLD_VIOLATIONS_REVIEW_SAMPLE_DIRNAME