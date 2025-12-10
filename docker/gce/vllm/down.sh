#!/bin/sh

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -n "$GCP_CREDENTIALS" ]
then
    GOOGLE_APPLICATION_CREDENTIALS=$(mktemp)
    echo "$GCP_CREDENTIALS" > $GOOGLE_APPLICATION_CREDENTIALS
elif [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]
then
    echo "Please provide GOOGLE_APPLICATION_CREDENTIALS, which is the path to your credentials file"
    exit 1
fi

terraform destroy \
          -var="gcp_project=$GCP_PROJECT" \
          -var="gcp_service_account=${GCP_SERVICE_ACCOUNT}" \
          -var="hf_token=$HF_TOKEN" \
          -var="vllm_org=${VLLM_ORG:-starpit}" \
          -var="vllm_repo=${VLLM_REPO:-vllm-ibm}" \
          -var="vllm_branch=${VLLM_BRANCH:-spnl-ibm}" \
          -auto-approve
