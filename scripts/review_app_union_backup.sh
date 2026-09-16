. scripts/.env

# $1 is the first argument passed to the script, which can be used to determine if we should copy images or not. If $1 is "copy_images", then we will copy the images directory to the backup location. Otherwise, we will only copy the labels, review_label_backups, review_decisions.json, and review_phash_cache.json files to the backup location.
# $2 is the second argument passed to the script, which can be used to specify a different backup location. If $2 is provided, we will use that as the backup location instead of the default /mnt/x/backup.

SOURCE_DIR=$DATA_ROOT/$UNION_DIRNAME

if [ -n "$2" ]; then
    BACKUP_DIR="$2"
else
    BACKUP_DIR="/mnt/x/backup/$UNION_DIRNAME"
fi
# if $1 is copy_images, then copy images to backup
if [ "$1" == "copy_images" ]; then
    cp -r $SOURCE_DIR/images $BACKUP_DIR
fi

mkdir -p $BACKUP_DIR

cp -r $SOURCE_DIR/labels $BACKUP_DIR
cp -r $SOURCE_DIR/review_label_backups $BACKUP_DIR
cp $SOURCE_DIR/review_decisions.json $BACKUP_DIR
cp $SOURCE_DIR/review_phash_cache.json $BACKUP_DIR

