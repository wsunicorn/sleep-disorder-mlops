#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-sleep-portal}"
ECS_CLUSTER="${ECS_CLUSTER:?ECS_CLUSTER is required}"
ECS_SERVICE="${ECS_SERVICE:?ECS_SERVICE is required}"
CONTAINER_NAME="${CONTAINER_NAME:-sleep-portal}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"

ALB_NAME="${PROJECT_NAME}-alb"
ALB_SG_NAME="${PROJECT_NAME}-alb-sg"
TARGET_GROUP_NAME="${PROJECT_NAME}-tg"

not_found() {
  [[ -z "${1:-}" || "$1" == "None" ]]
}

echo "Inspecting ECS service $ECS_CLUSTER/$ECS_SERVICE"
SERVICE_SUBNET_IDS=$(aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE" \
  --query "services[0].networkConfiguration.awsvpcConfiguration.subnets" \
  --output text)
read -r -a SERVICE_SUBNETS <<< "$SERVICE_SUBNET_IDS"

if [[ ${#SERVICE_SUBNETS[@]} -eq 0 || "${SERVICE_SUBNETS[0]}" == "None" ]]; then
  echo "Could not determine ECS service subnets." >&2
  exit 1
fi

VPC_ID=$(aws ec2 describe-subnets \
  --subnet-ids "${SERVICE_SUBNETS[0]}" \
  --query "Subnets[0].VpcId" \
  --output text)
echo "VPC: $VPC_ID"

PUBLIC_SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=tag:Name,Values=${PROJECT_NAME}-public-*" \
  --query "Subnets[].SubnetId" \
  --output text)
read -r -a PUBLIC_SUBNETS <<< "$PUBLIC_SUBNET_IDS"

if [[ ${#PUBLIC_SUBNETS[@]} -lt 2 || "${PUBLIC_SUBNETS[0]:-None}" == "None" ]]; then
  echo "Could not find at least two public subnets tagged ${PROJECT_NAME}-public-*." >&2
  echo "Recreate/import the Terraform network stack before restoring the ALB." >&2
  exit 1
fi
echo "Public subnets: ${PUBLIC_SUBNETS[*]}"

ALB_SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=$ALB_SG_NAME" \
  --query "SecurityGroups[0].GroupId" \
  --output text 2>/dev/null || true)

if not_found "$ALB_SG_ID"; then
  echo "Creating ALB security group: $ALB_SG_NAME"
  ALB_SG_ID=$(aws ec2 create-security-group \
    --group-name "$ALB_SG_NAME" \
    --description "Allow HTTP/HTTPS inbound to ALB" \
    --vpc-id "$VPC_ID" \
    --query "GroupId" \
    --output text)
  aws ec2 create-tags --resources "$ALB_SG_ID" --tags "Key=Name,Value=$ALB_SG_NAME"
fi

aws ec2 authorize-security-group-ingress \
  --group-id "$ALB_SG_ID" \
  --ip-permissions \
    "IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0,Description=HTTP from anywhere}]" \
  >/dev/null 2>&1 || true
aws ec2 authorize-security-group-ingress \
  --group-id "$ALB_SG_ID" \
  --ip-permissions \
    "IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0,Description=HTTPS from anywhere}]" \
  >/dev/null 2>&1 || true

ECS_SECURITY_GROUPS=$(aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE" \
  --query "services[0].networkConfiguration.awsvpcConfiguration.securityGroups" \
  --output text)
read -r -a ECS_SGS <<< "$ECS_SECURITY_GROUPS"
if [[ ${#ECS_SGS[@]} -gt 0 && "${ECS_SGS[0]}" != "None" ]]; then
  echo "Ensuring ECS SG ${ECS_SGS[0]} accepts app traffic from ALB SG $ALB_SG_ID"
  aws ec2 authorize-security-group-ingress \
    --group-id "${ECS_SGS[0]}" \
    --protocol tcp \
    --port "$CONTAINER_PORT" \
    --source-group "$ALB_SG_ID" \
    >/dev/null 2>&1 || true
fi

TG_ARN=$(aws elbv2 describe-target-groups \
  --names "$TARGET_GROUP_NAME" \
  --query "TargetGroups[0].TargetGroupArn" \
  --output text 2>/dev/null || true)

if not_found "$TG_ARN"; then
  echo "Creating target group: $TARGET_GROUP_NAME"
  TG_ARN=$(aws elbv2 create-target-group \
    --name "$TARGET_GROUP_NAME" \
    --protocol HTTP \
    --port "$CONTAINER_PORT" \
    --vpc-id "$VPC_ID" \
    --target-type ip \
    --health-check-protocol HTTP \
    --health-check-path "/api/v1/health/" \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 10 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --matcher "HttpCode=200" \
    --query "TargetGroups[0].TargetGroupArn" \
    --output text)
  aws elbv2 add-tags --resource-arns "$TG_ARN" --tags "Key=Name,Value=$TARGET_GROUP_NAME"
fi
echo "Target group: $TG_ARN"

ALB_ARN=$(aws elbv2 describe-load-balancers \
  --names "$ALB_NAME" \
  --query "LoadBalancers[0].LoadBalancerArn" \
  --output text 2>/dev/null || true)

if not_found "$ALB_ARN"; then
  echo "Creating ALB: $ALB_NAME"
  ALB_ARN=$(aws elbv2 create-load-balancer \
    --name "$ALB_NAME" \
    --type application \
    --scheme internet-facing \
    --security-groups "$ALB_SG_ID" \
    --subnets "${PUBLIC_SUBNETS[@]}" \
    --query "LoadBalancers[0].LoadBalancerArn" \
    --output text)
  aws elbv2 add-tags --resource-arns "$ALB_ARN" --tags "Key=Name,Value=$ALB_NAME"
fi

aws elbv2 wait load-balancer-available --load-balancer-arns "$ALB_ARN"
ALB_DNS=$(aws elbv2 describe-load-balancers \
  --load-balancer-arns "$ALB_ARN" \
  --query "LoadBalancers[0].DNSName" \
  --output text)
echo "ALB DNS: $ALB_DNS"

LISTENER_ARN=$(aws elbv2 describe-listeners \
  --load-balancer-arn "$ALB_ARN" \
  --query 'Listeners[?Port==`80`].ListenerArn | [0]' \
  --output text 2>/dev/null || true)

if not_found "$LISTENER_ARN"; then
  echo "Creating HTTP listener on port 80"
  aws elbv2 create-listener \
    --load-balancer-arn "$ALB_ARN" \
    --protocol HTTP \
    --port 80 \
    --default-actions "Type=forward,TargetGroupArn=$TG_ARN" \
    >/dev/null
else
  echo "Updating HTTP listener default action"
  aws elbv2 modify-listener \
    --listener-arn "$LISTENER_ARN" \
    --default-actions "Type=forward,TargetGroupArn=$TG_ARN" \
    >/dev/null
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "TG_ARN=$TG_ARN"
    echo "APP_URL=http://$ALB_DNS"
  } >> "$GITHUB_ENV"
fi

