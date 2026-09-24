. scripts/.env

BACKUP_BASE="/mnt/x/backup"
mkdir -p $BACKUP_BASE

##----------------------------------------- backup processed/union
RELATIVE_PATH=processed/union
mkdir -p $BACKUP_BASE/$DATA_ROOT/$RELATIVE_PATH

cp -r $DATA_ROOT/$RELATIVE_PATH/labels $BACKUP_BASE/$DATA_ROOT/$RELATIVE_PATH/labels
# cp -r $DATA_ROOT/$RELATIVE_PATH/images $BACKUP_BASE/$DATA_ROOT/$RELATIVE_PATH/images
cp $DATA_ROOT/$RELATIVE_PATH/review_decisions.json $BACKUP_BASE/$DATA_ROOT/$RELATIVE_PATH/review_decisions.json
cp $DATA_ROOT/$RELATIVE_PATH/review_phash_cache.json $BACKUP_BASE/$DATA_ROOT/$RELATIVE_PATH/review_phash_cache.json

##----------------------------------------- backup video test set
RELATIVE_PATH=processed/video_testset
# RELATIVE_PATH=processed/video_testset_final
mkdir -p $BACKUP_BASE/$DATA_ROOT/$RELATIVE_PATH
cp -r $DATA_ROOT/$RELATIVE_PATH $BACKUP_BASE/$DATA_ROOT/$RELATIVE_PATH

##----------------------------------------- test set da dataset escluso per annotazioni persona-scooter unica
RELATIVE_PATH=processed/holdout_kickboard_a75qx
# RELATIVE_PATH=processed/holdout_kickboard_a75qx_final
mkdir -p $BACKUP_BASE/$DATA_ROOT/$RELATIVE_PATH
cp -r $DATA_ROOT/$RELATIVE_PATH $BACKUP_BASE/$DATA_ROOT/$RELATIVE_PATH


##----------------------------------------- runs
mkdir -p $BACKUP_BASE/runs
cp -r runs $BACKUP_BASE/runs


##-----------------------------------------  lista dei folder da backuppare 
BACKUP_FOLDERS=(
    cache
)
BASE_DIR=data
BACKUP_BASE="/mnt/x/backup/data"
for folder in "${BACKUP_FOLDERS[@]}"; do
    mkdir -p "$BACKUP_BASE/$folder"
    cp -r "$BASE_DIR/$folder" "$BACKUP_BASE"
done