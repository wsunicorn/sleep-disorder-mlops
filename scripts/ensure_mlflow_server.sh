#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-sleep-portal}"
ECS_CLUSTER="${ECS_CLUSTER:?ECS_CLUSTER is required}"
APP_ECS_SERVICE="${APP_ECS_SERVICE:-${ECS_SERVICE:-sleep-portal-service}}"
MLFLOW_ECS_SERVICE="${MLFLOW_ECS_SERVICE:-${PROJECT_NAME}-mlflow-service}"
MLFLOW_TASK_FAMILY="${MLFLOW_TASK_FAMILY:-${PROJECT_NAME}-mlflow-task}"
MLFLOW_CONTAINER_NAME="${MLFLOW_CONTAINER_NAME:-mlflow-server}"
MLFLOW_CONTAINER_PORT="${MLFLOW_CONTAINER_PORT:-5000}"
MLFLOW_TARGET_GROUP_NAME="${MLFLOW_TARGET_GROUP_NAME:-${PROJECT_NAME}-mlflow-tg}"
MLFLOW_IMAGE_URI="${MLFLOW_IMAGE_URI:?MLFLOW_IMAGE_URI is required}"
MLFLOW_ARTIFACTS_DESTINATION="${MLFLOW_ARTIFACTS_DESTINATION:-s3://sleep-mlops-651709/mlflow-artifacts}"
LOG_GROUP="${LOG_GROUP:-/ecs/${PROJECT_NAME}}"
ALB_NAME="${ALB_NAME:-${PROJECT_NAME}-alb}"

not_found() {
  [[ -z "${1:-}" || "$1" == "None" || "$1" == "null" ]]
}

join_by() {
  local IFS="$1"
  shift
  echo "$*"
}

echo "Inspecting app ECS service $ECS_CLUSTER/$APP_ECS_SERVICE"
APP_SERVICE_JSON=$(aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services "$APP_ECS_SERVICE" \
  --output json)

SERVICE_SUBNET_IDS=$(jq -r '.services[0].networkConfiguration.awsvpcConfiguration.subnets[]?' <<< "$APP_SERVICE_JSON")
readarray -t SERVICE_SUBNETS <<< "$SERVICE_SUBNET_IDS"
SERVICE_SG_IDS=$(jq -r '.services[0].networkConfiguration.awsvpcConfiguration.securityGroups[]?' <<< "$APP_SERVICE_JSON")
readarray -t SERVICE_SGS <<< "$SERVICE_SG_IDS"
ASSIGN_PUBLIC_IP=$(jq -r '.services[0].networkConfiguration.awsvpcConfiguration.assignPublicIp // "DISABLED"' <<< "$APP_SERVICE_JSON")
APP_TASK_DEF=$(jq -r '.services[0].taskDefinition' <<< "$APP_SERVICE_JSON")

if [[ ${#SERVICE_SUBNETS[@]} -eq 0 || "${SERVICE_SUBNETS[0]}" == "" ]]; then
  echo "Could not determine subnets from $APP_ECS_SERVICE." >&2
  exit 1
fi
if [[ ${#SERVICE_SGS[@]} -eq 0 || "${SERVICE_SGS[0]}" == "" ]]; then
  echo "Could not determine security groups from $APP_ECS_SERVICE." >&2
  exit 1
fi

VPC_ID=$(aws ec2 describe-subnets \
  --subnet-ids "${SERVICE_SUBNETS[0]}" \
  --query "Subnets[0].VpcId" \
  --output text)

APP_TASK_JSON=$(aws ecs describe-task-definition \
  --task-definition "$APP_TASK_DEF" \
  --query "taskDefinition" \
  --output json)
DATABASE_URL="${MLFLOW_BACKEND_STORE_URI:-$(jq -r '.containerDefinitions[] | select(.name=="sleep-portal") | .environment[]? | select(.name=="DATABASE_URL") | .value' <<< "$APP_TASK_JSON" | head -n 1)}"
if not_found "$DATABASE_URL"; then
  echo "DATABASE_URL was not found in the app task definition; set MLFLOW_BACKEND_STORE_URI explicitly." >&2
  exit 1
fi

EXECUTION_ROLE_ARN=$(jq -r '.executionRoleArn' <<< "$APP_TASK_JSON")
TASK_ROLE_ARN=$(jq -r '.taskRoleArn' <<< "$APP_TASK_JSON")
CPU="${MLFLOW_TASK_CPU:-512}"
MEMORY="${MLFLOW_TASK_MEMORY:-1024}"

ALB_ARN=$(aws elbv2 describe-load-balancers \
  --names "$ALB_NAME" \
  --query "LoadBalancers[0].LoadBalancerArn" \
  --output text 2>/dev/null || true)
if not_found "$ALB_ARN"; then
  echo "ALB $ALB_NAME was not found. Run scripts/ensure_aws_alb.sh first." >&2
  exit 1
fi

ALB_JSON=$(aws elbv2 describe-load-balancers \
  --load-balancer-arns "$ALB_ARN" \
  --output json)
ALB_DNS=$(jq -r '.LoadBalancers[0].DNSName' <<< "$ALB_JSON")
ALB_SG_IDS=$(jq -r '.LoadBalancers[0].SecurityGroups[]?' <<< "$ALB_JSON")
readarray -t ALB_SGS <<< "$ALB_SG_IDS"
if [[ ${#ALB_SGS[@]} -eq 0 || "${ALB_SGS[0]}" == "" ]]; then
  echo "Could not determine ALB security group." >&2
  exit 1
fi
ALB_SG_ID="${ALB_SGS[0]}"

echo "Ensuring security groups allow MLflow traffic on port $MLFLOW_CONTAINER_PORT"
aws ec2 authorize-security-group-ingress \
  --group-id "$ALB_SG_ID" \
  --protocol tcp \
  --port "$MLFLOW_CONTAINER_PORT" \
  --cidr "0.0.0.0/0" \
  >/dev/null 2>&1 || true

for sg_id in "${SERVICE_SGS[@]}"; do
  aws ec2 authorize-security-group-ingress \
    --group-id "$sg_id" \
    --protocol tcp \
    --port "$MLFLOW_CONTAINER_PORT" \
    --source-group "$ALB_SG_ID" \
    >/dev/null 2>&1 || true
done

TG_ARN=$(aws elbv2 describe-target-groups \
  --names "$MLFLOW_TARGET_GROUP_NAME" \
  --query "TargetGroups[0].TargetGroupArn" \
  --output text 2>/dev/null || true)

if not_found "$TG_ARN"; then
  echo "Creating MLflow target group: $MLFLOW_TARGET_GROUP_NAME"
  TG_ARN=$(aws elbv2 create-target-group \
    --name "$MLFLOW_TARGET_GROUP_NAME" \
    --protocol HTTP \
    --port "$MLFLOW_CONTAINER_PORT" \
    --vpc-id "$VPC_ID" \
    --target-type ip \
    --health-check-protocol HTTP \
    --health-check-path "/" \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 10 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --matcher "HttpCode=200-399" \
    --query "TargetGroups[0].TargetGroupArn" \
    --output text)
  aws elbv2 add-tags \
    --resource-arns "$TG_ARN" \
    --tags "Key=Name,Value=$MLFLOW_TARGET_GROUP_NAME" \
    >/dev/null 2>&1 || true
fi
echo "MLflow target group: $TG_ARN"

LISTENERS_JSON=$(aws elbv2 describe-listeners \
  --load-balancer-arn "$ALB_ARN" \
  --output json)
LISTENER_ARN=$(jq -r --argjson port "$MLFLOW_CONTAINER_PORT" '.Listeners[]? | select(.Port == $port) | .ListenerArn' <<< "$LISTENERS_JSON" | head -n 1)

if not_found "$LISTENER_ARN"; then
  echo "Creating ALB listener on port $MLFLOW_CONTAINER_PORT"
  aws elbv2 create-listener \
    --load-balancer-arn "$ALB_ARN" \
    --protocol HTTP \
    --port "$MLFLOW_CONTAINER_PORT" \
    --default-actions "Type=forward,TargetGroupArn=$TG_ARN" \
    >/dev/null
else
  CURRENT_TG=$(aws elbv2 describe-listeners \
    --listener-arns "$LISTENER_ARN" \
    --query "Listeners[0].DefaultActions[0].TargetGroupArn" \
    --output text 2>/dev/null || true)
  if [[ "$CURRENT_TG" != "$TG_ARN" ]]; then
    echo "Updating MLflow listener default action"
    aws elbv2 modify-listener \
      --listener-arn "$LISTENER_ARN" \
      --default-actions "Type=forward,TargetGroupArn=$TG_ARN" \
      >/dev/null
  fi
fi

aws logs create-log-group --log-group-name "$LOG_GROUP" >/dev/null 2>&1 || true

CONTAINER_DEFINITIONS=$(jq -n \
  --arg name "$MLFLOW_CONTAINER_NAME" \
  --arg image "$MLFLOW_IMAGE_URI" \
  --arg port "$MLFLOW_CONTAINER_PORT" \
  --arg backend "$DATABASE_URL" \
  --arg artifacts "$MLFLOW_ARTIFACTS_DESTINATION" \
  --arg region "${AWS_REGION:-${AWS_DEFAULT_REGION:-ap-southeast-1}}" \
  --arg log_group "$LOG_GROUP" \
  '[
    {
      name: $name,
      image: $image,
      essential: true,
      portMappings: [{containerPort: ($port | tonumber), protocol: "tcp"}],
      environment: [
        {name: "MLFLOW_BACKEND_STORE_URI", value: $backend},
        {name: "MLFLOW_ARTIFACTS_DESTINATION", value: $artifacts},
        {name: "AWS_DEFAULT_REGION", value: $region}
      ],
      logConfiguration: {
        logDriver: "awslogs",
        options: {
          "awslogs-group": $log_group,
          "awslogs-region": $region,
          "awslogs-stream-prefix": "mlflow"
        }
      },
      healthCheck: {
        command: ["CMD-SHELL", "curl -f http://localhost:" + $port + "/ || exit 1"],
        interval: 30,
        timeout: 10,
        retries: 3,
        startPeriod: 60
      }
    }
  ]')

TASK_DEF_ARGS=(
  --family "$MLFLOW_TASK_FAMILY"
  --requires-compatibilities FARGATE
  --network-mode awsvpc
  --cpu "$CPU"
  --memory "$MEMORY"
  --execution-role-arn "$EXECUTION_ROLE_ARN"
  --container-definitions "$CONTAINER_DEFINITIONS"
)
if ! not_found "$TASK_ROLE_ARN"; then
  TASK_DEF_ARGS+=(--task-role-arn "$TASK_ROLE_ARN")
fi

NEW_TASK_DEF=$(aws ecs register-task-definition \
  "${TASK_DEF_ARGS[@]}" \
  --query "taskDefinition.taskDefinitionArn" \
  --output text)
echo "Registered MLflow task definition: $NEW_TASK_DEF"

SUBNETS_CSV=$(join_by "," "${SERVICE_SUBNETS[@]}")
SGS_CSV=$(join_by "," "${SERVICE_SGS[@]}")
NETWORK_CONFIG="awsvpcConfiguration={subnets=[$SUBNETS_CSV],securityGroups=[$SGS_CSV],assignPublicIp=$ASSIGN_PUBLIC_IP}"
LOAD_BALANCER="targetGroupArn=$TG_ARN,containerName=$MLFLOW_CONTAINER_NAME,containerPort=$MLFLOW_CONTAINER_PORT"

SERVICE_EXISTS=$(aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services "$MLFLOW_ECS_SERVICE" \
  --query "services[0].status" \
  --output text 2>/dev/null || true)

if [[ "$SERVICE_EXISTS" == "ACTIVE" ]]; then
  echo "Updating MLflow ECS service $MLFLOW_ECS_SERVICE"
  aws ecs update-service \
    --cluster "$ECS_CLUSTER" \
    --service "$MLFLOW_ECS_SERVICE" \
    --task-definition "$NEW_TASK_DEF" \
    --desired-count "${MLFLOW_DESIRED_COUNT:-1}" \
    --load-balancers "$LOAD_BALANCER" \
    --force-new-deployment \
    >/dev/null
else
  echo "Creating MLflow ECS service $MLFLOW_ECS_SERVICE"
  aws ecs create-service \
    --cluster "$ECS_CLUSTER" \
    --service-name "$MLFLOW_ECS_SERVICE" \
    --task-definition "$NEW_TASK_DEF" \
    --desired-count "${MLFLOW_DESIRED_COUNT:-1}" \
    --launch-type FARGATE \
    --network-configuration "$NETWORK_CONFIG" \
    --load-balancers "$LOAD_BALANCER" \
    --deployment-configuration "minimumHealthyPercent=100,maximumPercent=200" \
    >/dev/null
fi

aws ecs wait services-stable \
  --cluster "$ECS_CLUSTER" \
  --services "$MLFLOW_ECS_SERVICE"

MLFLOW_URL="http://$ALB_DNS:$MLFLOW_CONTAINER_PORT"
echo "MLflow URL: $MLFLOW_URL"
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "MLFLOW_TRACKING_URI=$MLFLOW_URL" >> "$GITHUB_ENV"
  echo "MLFLOW_URL=$MLFLOW_URL" >> "$GITHUB_ENV"
fi
