# ── Core identifiers ──────────────────────────────────────────────────────────
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

# ── Load balancer ─────────────────────────────────────────────────────────────
output "alb_dns_name" {
  description = "Public DNS name of the ALB"
  value       = aws_lb.main.dns_name
}

output "app_url" {
  description = "Base URL of the deployed application"
  value       = "http://${aws_lb.main.dns_name}"
}

# ── ECR ───────────────────────────────────────────────────────────────────────
output "ecr_repository_url" {
  description = "ECR repository URL (use as Docker registry)"
  value       = aws_ecr_repository.app.repository_url
}

# ── ECS ───────────────────────────────────────────────────────────────────────
output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.app.name
}

# ── RDS ───────────────────────────────────────────────────────────────────────
output "rds_endpoint" {
  description = "RDS instance connection endpoint"
  value       = aws_db_instance.postgres.endpoint
  sensitive   = true
}

output "database_url" {
  description = "Full DATABASE_URL connection string for Django"
  value       = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.postgres.address}:5432/${var.db_name}"
  sensitive   = true
}
