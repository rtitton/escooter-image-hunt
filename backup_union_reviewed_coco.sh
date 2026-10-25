. scripts/.env

BACKUP_TARGET="/mnt/x/data/processed/union_reviewed_coco"
mkdir -p $BACKUP_TARGET

# cp -r $DATA_ROOT/processed/union_reviewed_coco/images $BACKUP_TARGET
cp -r $DATA_ROOT/processed/union_reviewed_coco/labels $BACKUP_TARGET
cp $DATA_ROOT/processed/union_reviewed_coco/review_decisions.json $BACKUP_TARGET
cp -r $DATA_ROOT/processed/union_reviewed_coco/review_label_backups $BACKUP_TARGET
# cache pHash (v. review_app.py "ordina per pHash"): potrebbe non esistere ancora se non l'hai mai usata
[ -f $DATA_ROOT/processed/union_reviewed_coco/review_phash_cache.json ] && \
    cp $DATA_ROOT/processed/union_reviewed_coco/review_phash_cache.json $BACKUP_TARGET
    