#!/usr/bin/env bash

set -eo pipefail
SCRIPTDIR=$(cd $(dirname "$0") && pwd)

cleanup() {
    "$SCRIPTDIR/down.sh"
}
trap cleanup SIGINT

bucket=${GCS_BUCKET:-spnl-test}
run_id=${RUN_ID:-$(uuidgen | tr [A-Z] [a-z])}

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -n "$GCP_CREDENTIALS" ]
then
    GOOGLE_APPLICATION_CREDENTIALS=$(mktemp)
    echo "$GCP_CREDENTIALS" > $GOOGLE_APPLICATION_CREDENTIALS
elif [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]
then
    echo "Please provide GOOGLE_APPLICATION_CREDENTIALS, which is the path to your credentials file"
    exit 1
fi

for region in us-west1 us-central1
do for zone in a b c
   do
       for i in $(seq 1 3)
       do
           echo "Trying region=$region zone=$zone"
           if terraform apply \
                        -var="gcp_project=$GCP_PROJECT" \
                        -var="gcp_service_account=${GCP_SERVICE_ACCOUNT}" \
                        -var="gcs_bucket=$bucket" \
                        -var="gce_region=$region" \
                        -var="gce_zone=$region-$zone" \
                        -var="run_id=$run_id" \
                        -var="hf_token=$HF_TOKEN" \
                        -var="model=${MODEL:-ibm-granite/granite-3.3-2b-instruct}" \
                        -var="spnl_github=${SPNL_GITHUB:-https://github.com/IBM/spnl.git}" \
                        -var="spnl_github_sha=${GITHUB_SHA}" \
                        -var="spnl_github_ref=${GITHUB_REF}" \
                        -var="vllm_org=${VLLM_ORG:-neuralmagic}" \
                        -var="vllm_repo=${VLLM_REPO:-vllm}" \
                        -var="vllm_branch=${VLLM_BRANCH:-llm-d-release-0.4}" \
                        -auto-approve
           then
               good=1
               break
           else "$SCRIPTDIR"/down.sh
           fi
       done
       if [[ $good = 1 ]]; then break; fi
   done
   if [[ $good = 1 ]]; then break; fi
done

if [[ $good != 1 ]]
then
    echo "Error: Could not allocate GCE VM" 2>&1
    exit 1
fi

zone="$region-$zone"

echo "Waiting for VM readiness in $zone"
until gcloud compute instances list --zones=$zone --filter labels.gh-run-id=$run_id; do sleep 1; done
vm=$(gcloud compute instances list --zones=$zone --filter labels.gh-run-id=$run_id | tail -1 | awk '{print $1}')

echo "Waiting for ssh readiness $vm"
until gcloud compute ssh $vm --zone=$zone --command "echo Ready"; do sleep 1; done

gcloud compute ssh $vm --zone=$zone --command 'tail -f /var/log/cloud-init-output.log' || \
    "$SCRIPTDIR"/down.sh

# fetch the exit code from GCS, then delete that GCS file, and then exit this shell with that code
exit_code=$(gsutil cat gs://$bucket/runs/$run_id/status/exit_code)
gsutil rm gs://$bucket/runs/$run_id/status/exit_code
exit $exit_code
