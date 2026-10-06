#!/usr/bin/env bash
# set_item_status.sh <project-item-id> <field-value-id>
# Field value IDs (Status): Backlog=f75ad846 Ready=61e4505c In progress=47fc9ee4 In review=df73e18b Done=98236657
set -euo pipefail
ITEM_ID="$1"; VAL="$2"
Q="mutation { updateProjectV2ItemFieldValue(input: {itemId: \"$ITEM_ID\", field: {id: \"PVTSSF_lADOEyEOMs4Bl95-zhkosgM\"}, value: {singleSelectOptionId: \"$VAL\"}}) { item { id } } }"
echo "set $ITEM_ID -> $VAL"
