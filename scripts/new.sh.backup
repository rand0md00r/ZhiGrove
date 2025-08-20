#!/usr/bin/env bash
set -e

TYPE="$1"         # paper | idea | experiment | report
TITLE="$2"        # "VTLA preference learning"
DATE=$(date +"%Y-%m-%d")
YEAR=$(date +"%Y")
MONTH=$(date +"%m")
SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g;s/^-|-$//g')

case "$TYPE" in
  paper)
    DIR="20-papers/$YEAR"
    FILE="$DIR/$DATE-$SLUG.md"
    TPL="templates/template-paper.md"
    ;;
  idea)
    DIR="30-ideas/$YEAR"
    FILE="$DIR/$DATE-$SLUG.md"
    TPL="templates/template-idea.md"
    ;;
  experiment)
    DIR="40-experiments/exp-$DATE-$SLUG"
    mkdir -p "$DIR/results" "$DIR/configs"
    FILE="$DIR/log.md"
    TPL="templates/template-experiment.md"
    ;;
  report)
    DIR="50-reports/weekly/$YEAR"
    FILE="$DIR/week-$(date +%V).md"
    TPL="templates/template-report.md"
    ;;
  *)
    echo "Usage: scripts/new.sh {paper|idea|experiment|report} \"Title\""; exit 1;
esac

mkdir -p "$DIR"
sed "s/{{DATE}}/$DATE/g" "$TPL" > "$FILE"
echo "Created: $FILE"
