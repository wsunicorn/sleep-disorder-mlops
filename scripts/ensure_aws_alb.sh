#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-sleep-portal}"
ECS_CLUSTER="${ECS_CLUSTER:?ECS_CLUSTER is required}"
ECS_SERVICE="${ECS_SERVICE:?ECS_SERVICE is required}"
CONTAINER_NAME="${CONTAINER_NAME:-sleep-portal}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
ALB_IDLE_TIMEOUT_SECONDS="${ALB_IDLE_TIMEOUT_SECONDS:-300}"

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
  echo "Could not find two tagged public subnets; checking route tables with an internet gateway."
  PUBLIC_SUBNET_IDS=$(aws ec2 describe-route-tables \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query "RouteTables[?Routes[?DestinationCidrBlock=='0.0.0.0/0' && GatewayId!=null && starts_with(GatewayId, 'igw-')]].Associations[?SubnetId!=null].SubnetId[]" \
    --output text)
  read -r -a PUBLIC_SUBNETS <<< "$PUBLIC_SUBNET_IDS"
fi

if [[ ${#PUBLIC_SUBNETS[@]} -lt 2 || "${PUBLIC_SUBNETS[0]:-None}" == "None" ]]; then
  echo "Could not find public subnets in route tables; checking map-public-ip-on-launch subnets."
  MAP_PUBLIC_SUBNET_IDS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[?MapPublicIpOnLaunch==`true`].SubnetId' \
    --output text)
  read -r -a PUBLIC_SUBNETS <<< "$MAP_PUBLIC_SUBNET_IDS"
  if [[ "${PUBLIC_SUBNETS[0]:-None}" == "None" ]]; then
    PUBLIC_SUBNETS=()
  fi
fi

if [[ ${#PUBLIC_SUBNETS[@]} -lt 2 ]]; then
  echo "Creating replacement public subnets until two are available."
  VPC_CIDR=$(aws ec2 describe-vpcs \
    --vpc-ids "$VPC_ID" \
    --query "Vpcs[0].CidrBlock" \
    --output text)
  EXISTING_CIDRS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query "Subnets[].CidrBlock" \
    --output text)
  CANDIDATE_CIDRS=$(python3 - "$VPC_CIDR" "$EXISTING_CIDRS" <<'PY'
import ipaddress
import sys

vpc = ipaddress.ip_network(sys.argv[1])
existing = [
    ipaddress.ip_network(item)
    for item in sys.argv[2].split()
    if item.strip()
]

chosen = []
for subnet in vpc.subnets(new_prefix=24):
    if any(subnet.overlaps(current) for current in existing):
        continue
    chosen.append(str(subnet))
    if len(chosen) == 2:
        break

if len(chosen) < 2:
    raise SystemExit("Could not find two free /24 CIDR blocks in the VPC.")

print(" ".join(chosen))
PY
)
  read -r -a CANDIDATE_CIDRS_ARRAY <<< "$CANDIDATE_CIDRS"
  AZ_NAMES=$(aws ec2 describe-availability-zones \
    --query "AvailabilityZones[0:2].ZoneName" \
    --output text)
  read -r -a AZS <<< "$AZ_NAMES"

  candidate_index=0
  while [[ ${#PUBLIC_SUBNETS[@]} -lt 2 ]]; do
    if [[ $candidate_index -ge ${#CANDIDATE_CIDRS_ARRAY[@]} ]]; then
      echo "Could not find enough free CIDR blocks to create public subnets." >&2
      exit 1
    fi
    index="${#PUBLIC_SUBNETS[@]}"
    CIDR="${CANDIDATE_CIDRS_ARRAY[$candidate_index]}"
    AZ="${AZS[$(( index % ${#AZS[@]} ))]}"
    echo "Creating public subnet $CIDR in $AZ"
    SUBNET_ID=$(aws ec2 create-subnet \
      --vpc-id "$VPC_ID" \
      --cidr-block "$CIDR" \
      --availability-zone "$AZ" \
      --query "Subnet.SubnetId" \
      --output text)
    aws ec2 modify-subnet-attribute \
      --subnet-id "$SUBNET_ID" \
      --map-public-ip-on-launch
    aws ec2 create-tags \
      --resources "$SUBNET_ID" \
      --tags "Key=Name,Value=${PROJECT_NAME}-public-${index}" \
      >/dev/null 2>&1 || true
    PUBLIC_SUBNETS+=("$SUBNET_ID")
    candidate_index=$((candidate_index + 1))
  done

  IGW_ID=$(aws ec2 describe-internet-gateways \
    --filters "Name=attachment.vpc-id,Values=$VPC_ID" \
    --query "InternetGateways[0].InternetGatewayId" \
    --output text 2>/dev/null || true)
  if not_found "$IGW_ID"; then
    echo "Creating and attaching internet gateway"
    IGW_ID=$(aws ec2 create-internet-gateway \
      --query "InternetGateway.InternetGatewayId" \
      --output text)
    aws ec2 create-tags \
      --resources "$IGW_ID" \
      --tags "Key=Name,Value=${PROJECT_NAME}-igw" \
      >/dev/null 2>&1 || true
    aws ec2 attach-internet-gateway --internet-gateway-id "$IGW_ID" --vpc-id "$VPC_ID"
  fi

  ROUTE_TABLE_ID=$(aws ec2 create-route-table \
    --vpc-id "$VPC_ID" \
    --query "RouteTable.RouteTableId" \
    --output text)
  aws ec2 create-tags \
    --resources "$ROUTE_TABLE_ID" \
    --tags "Key=Name,Value=${PROJECT_NAME}-public-rt" \
    >/dev/null 2>&1 || true
  aws ec2 create-route \
    --route-table-id "$ROUTE_TABLE_ID" \
    --destination-cidr-block "0.0.0.0/0" \
    --gateway-id "$IGW_ID" \
    >/dev/null 2>&1 || true
  for subnet_id in "${PUBLIC_SUBNETS[@]}"; do
    aws ec2 associate-route-table \
      --route-table-id "$ROUTE_TABLE_ID" \
      --subnet-id "$subnet_id" \
      >/dev/null
  done
fi

declare -A SEEN_AZS=()
UNIQUE_PUBLIC_SUBNETS=()
for subnet_id in "${PUBLIC_SUBNETS[@]}"; do
  AZ=$(aws ec2 describe-subnets \
    --subnet-ids "$subnet_id" \
    --query "Subnets[0].AvailabilityZone" \
    --output text)
  if [[ -z "${SEEN_AZS[$AZ]:-}" ]]; then
    UNIQUE_PUBLIC_SUBNETS+=("$subnet_id")
    SEEN_AZS[$AZ]=1
  fi
done
PUBLIC_SUBNETS=("${UNIQUE_PUBLIC_SUBNETS[@]}")
if [[ ${#PUBLIC_SUBNETS[@]} -lt 2 ]]; then
  echo "ALB needs public subnets in at least two availability zones." >&2
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
    --description "Allow HTTP/HTTPS and MLflow inbound to ALB" \
    --vpc-id "$VPC_ID" \
    --query "GroupId" \
    --output text)
  aws ec2 create-tags \
    --resources "$ALB_SG_ID" \
    --tags "Key=Name,Value=$ALB_SG_NAME" \
    >/dev/null 2>&1 || true
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
  aws elbv2 add-tags \
    --resource-arns "$TG_ARN" \
    --tags "Key=Name,Value=$TARGET_GROUP_NAME" \
    >/dev/null 2>&1 || true
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
  aws elbv2 add-tags \
    --resource-arns "$ALB_ARN" \
    --tags "Key=Name,Value=$ALB_NAME" \
    >/dev/null 2>&1 || true
fi

aws elbv2 wait load-balancer-available --load-balancer-arns "$ALB_ARN"
echo "Ensuring ALB idle timeout is ${ALB_IDLE_TIMEOUT_SECONDS}s"
if aws elbv2 modify-load-balancer-attributes \
  --load-balancer-arn "$ALB_ARN" \
  --attributes "Key=idle_timeout.timeout_seconds,Value=$ALB_IDLE_TIMEOUT_SECONDS" \
  >/dev/null 2>&1; then
  echo "ALB idle timeout is ${ALB_IDLE_TIMEOUT_SECONDS}s"
else
  echo "Warning: could not update ALB idle timeout; continuing deploy. Grant elasticloadbalancing:ModifyLoadBalancerAttributes to allow long EDF uploads." >&2
fi
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
  CURRENT_LISTENER_TG_ARN=$(aws elbv2 describe-listeners \
    --listener-arns "$LISTENER_ARN" \
    --query "Listeners[0].DefaultActions[0].TargetGroupArn" \
    --output text 2>/dev/null || true)
  if [[ "$CURRENT_LISTENER_TG_ARN" == "$TG_ARN" ]]; then
    echo "HTTP listener already forwards to target group"
  else
    echo "Updating HTTP listener default action"
    aws elbv2 modify-listener \
      --listener-arn "$LISTENER_ARN" \
      --default-actions "Type=forward,TargetGroupArn=$TG_ARN" \
      >/dev/null
  fi
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "TG_ARN=$TG_ARN"
    echo "APP_URL=http://$ALB_DNS"
  } >> "$GITHUB_ENV"
fi
