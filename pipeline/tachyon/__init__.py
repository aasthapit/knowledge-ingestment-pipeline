# Tachyon integration modules — populated from the Tachyon pipeline repo.
#
# Expected modules (copy from your Tachyon repo or implement per
# TACHYON_INTEGRATION_CONTEXT.md):
#
#   auth.py      — get_access_token()   Apigee OAuth token exchange
#   search.py    — search_documents()   Tachyon semantic search
#   delete.py    — delete_file()        Remove S3 + vector doc by file IDs
#   upload.py    — upload_to_s3()       Upload JSONL to S3 (ingestion plan)
#   vectorize.py — vectorize_file()     Trigger Tachyon vectorization (ingestion plan)
