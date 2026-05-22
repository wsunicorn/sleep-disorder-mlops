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
  ROLE_ARN=$(aws iam create-role \
    --role-name "$TASK_ROLE_NAME" \
    --assume-role-policy-document "$ASSUME_ROLE_POLICY" \
    --query "Role.Arn" \
    --output text)
else
  echo "ECS task role exists: $TASK_ROLE_NAME"
fi

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

aws iam put-role-policy \
  --role-name "$TASK_ROLE_NAME" \
  --policy-name "$TASK_POLICY_NAME" \
  --policy-document "$POLICY_DOCUMENT"

echo "ECS task role ARN: $ROLE_ARN"
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "ECS_TASK_ROLE_ARN=$ROLE_ARN" >> "$GITHUB_ENV"
fi
