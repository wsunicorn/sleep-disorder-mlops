variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "ap-southeast-1"
}

variable "project" {
  description = "Project name prefix used in all resource names"
  type        = string
  default     = "sleep-portal"
}

variable "aws_account_id" {
  description = "AWS account ID"
  type        = string
  default     = "651709558967"
}

# ── Network ──────────────────────────────────────────────────────────────────
variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets (ALB)"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets (ECS tasks, RDS)"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24"]
}

# ── ECS / App ─────────────────────────────────────────────────────────────────
variable "container_port" {
  description = "Port the Django container listens on"
  type        = number
  default     = 8000
}

variable "task_cpu" {
  description = "Fargate task CPU units (256 = 0.25 vCPU)"
  type        = string
  default     = "512"
}

variable "task_memory" {
  description = "Fargate task memory in MB"
  type        = string
  default     = "1024"
}

variable "desired_count" {
  description = "Number of ECS task replicas"
  type        = number
  default     = 1
}

variable "ecr_image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

# ── RDS ───────────────────────────────────────────────────────────────────────
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "RDS storage in GB"
  type        = number
  default     = 20
}

variable "db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "sleep_portal"
}

variable "db_username" {
  description = "RDS master username"
  type        = string
  default     = "sleepAdmin"
  sensitive   = true
}

variable "db_password" {
  description = "RDS master password (set via TF_VAR_db_password)"
  type        = string
  sensitive   = true
}

# ── App secrets ──────────────────────────────────────────────────────────────
variable "django_secret_key" {
  description = "Django SECRET_KEY (set via TF_VAR_django_secret_key)"
  type        = string
  sensitive   = true
}

variable "django_allowed_hosts" {
  description = "Comma-separated list of Django ALLOWED_HOSTS"
  type        = string
  default     = "*"
}

# ── CloudWatch ────────────────────────────────────────────────────────────────
variable "alarm_sns_arn" {
  description = "SNS topic ARN for CloudWatch alarm notifications (optional)"
  type        = string
  default     = ""
}
