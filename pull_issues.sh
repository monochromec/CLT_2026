#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 owner/repo"
  exit 1
fi

REPO="$1"
OUTPUT="${OUTPUT:-issues.json}"
STATE="${STATE:-closed}"
STATE_UPPER=$(echo "$STATE" | tr '[:lower:]' '[:upper:]')

# Check dependencies
command -v gh >/dev/null || { echo "gh CLI required. Please install and authenticate."; exit 1; }
command -v jq >/dev/null || { echo "jq required. Please install jq."; exit 1; }

OWNER="${REPO%%/*}"
NAME="${REPO##*/}"

# The query MUST use $endCursor for 'gh api --paginate' to work properly
# Enclosing STATE_UPPER in brackets makes it a valid GraphQL Enum array: [CLOSED]
QUERY="
query(\$owner:String!, \$name:String!, \$endCursor:String) {
  repository(owner:\$owner, name:\$name) {
    issues(first:100, after:\$endCursor, states:[$STATE_UPPER]) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        createdAt
        closedAt
        url
      }
    }
  }
}"

echo "Fetching $STATE issues for $REPO..."

# 1. 'gh api' paginates automatically and outputs JSON for each page
# 2. '--jq' safely extracts the issue nodes from each page (ignoring nulls via '?')
# 3. 'jq -s' collects the stream of objects into a single array and maps the keys
gh api graphql --paginate \
  -f query="$QUERY" \
  -F owner="$OWNER" \
  -F name="$NAME" \
  --jq '.data.repository.issues.nodes[]?' | \
jq -s 'map({
  number: .number,
  title: .title,
  html_url: .url,
  created_at: .createdAt,
  closed_at: .closedAt
})' > "$OUTPUT"

# Verify the final output
if [ -s "$OUTPUT" ]; then
  TOTAL_ISSUES=$(jq 'length' "$OUTPUT")
  echo "Finished. Total unique issues downloaded: $TOTAL_ISSUES"
  
  gzip -9 -f "$OUTPUT"
  echo "Compressed successfully to $OUTPUT.gz"
else
  echo "Error: Output file is empty or the API returned no results."
  exit 1
fi
