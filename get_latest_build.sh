gcloud container builds list | head -n 2 | tail -n 1 | awk '{ print $1 }' | xargs gcloud container builds describe $1 | grep "commitSha" | awk '{ print $2 }'


