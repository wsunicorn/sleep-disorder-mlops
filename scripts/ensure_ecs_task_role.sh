#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-sleep-portal}"
TASK_ROLE_NAME="${ECS_TASK_ROLE_NAME:-${PROJECT_NAME}-ecs-task-role}"
TASK_POLICY_NAME="${ECS_TASK_POLICY_NAME:-${PROJECT_NAME}-ecs-s3-mlops}"
MLOPS_BUCKET="${MLOPS_BUCKET:-sleep-mlops-651709}"

ASSUME_ROLE_POLICY='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}'

ROLE_ARN=$(aws iam get-role \
  --role-name "$TASK_ROLE_NAME" \
  --query "Role.Arn" \
  --output text 2>/dev/null || true)

if [[ -z "${ROLE_ARN:-}" || "$ROLE_ARN" == "None" ]]; then
  echo "Creating ECS task role: $TASK_ROLE_NAME"
  if ROLE_ARN=$(aws iam create-role \
    --role-name "$TASK_ROLE_NAME" \
    --assume-role-policy-document "$ASSUME_ROLE_POLICY" \
    --query "Role.Arn" \
    --output text 2>/tmp/create-ecs-task-role.err); then
    echo "Created ECS task role: $TASK_ROLE_NAME"
  else
    echo "Could not create ECS task role. Falling back to AWS credentials from workflow environment."
    cat /tmp/create-ecs-task-role.err >&2 || true
    ROLE_ARN=""
  fi
else
  echo "ECS task role exists: $TASK_ROLE_NAME"
fi

if [[ -n "${ROLE_ARN:-}" && "$ROLE_ARN" != "None" ]]; then
  POLICY_DOCUMENT=$(jq -n \
    --arg bucket "$MLOPS_BUCKET" \
    '{
      Version: "2012-10-17",
      Statement: [
        {
          Effect: "Allow",
          Action: [
            "s3:GetObject",
            "s3:PutObject",
            "s3:ListBucket",
            "s3:DeleteObject"
          ],
          Resource: [
            "arn:aws:s3:::" + $bucket,
            "arn:aws:s3:::" + $bucket + "/*"
          ]
        },
        {
          Effect: "Allow",
          Action: [
            "cloudwatch:PutMetricData",
            "logs:CreateLogGroup",
            "logs:CreateLogStream",
            "logs:PutLogEvents"
          ],
          Resource: "*"
        }
      ]
    }')

  if ! aws iam put-role-policy \
    --role-name "$TASK_ROLE_NAME" \
    --policy-name "$TASK_POLICY_NAME" \
    --policy-document "$POLICY_DOCUMENT" 2>/tmp/put-ecs-task-role-policy.err; then
    echo "Could not attach inline policy to $TASK_ROLE_NAME. Falling back to AWS credentials from workflow environment."
    cat /tmp/put-ecs-task-role-policy.err >&2 || true
    ROLE_ARN=""
  fi
fi

if [[ -n "${ROLE_ARN:-}" && "$ROLE_ARN" != "None" ]]; then
  echo "ECS task role ARN: $ROLE_ARN"
else
  echo "ECS task role unavailable; deployments will inject AWS credentials from GitHub Secrets."
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "ECS_TASK_ROLE_ARN=$ROLE_ARN" >> "$GITHUB_ENV"
fi
