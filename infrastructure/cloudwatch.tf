# ── CloudWatch alarms ─────────────────────────────────────────────────────────
# Notifications go to var.alarm_sns_arn if provided.
locals {
  alarm_actions = var.alarm_sns_arn != "" ? [var.alarm_sns_arn] : []
}

# 1. ECS CPU utilisation > 80%
resource "aws_cloudwatch_metric_alarm" "ecs_cpu_high" {
  alarm_name          = "${var.project}-ecs-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = 60
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "ECS CPU utilization > 80%"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.app.name
  }

  alarm_actions = local.alarm_actions
  ok_actions    = local.alarm_actions
  tags          = { Name = "${var.project}-cpu-alarm" }
}

# 2. ALB 5xx error rate > 5%
resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "${var.project}-alb-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 5
  alarm_description   = "ALB HTTP 5xx error rate > 5%"

  metric_query {
    id          = "error_rate"
    expression  = "(m2 / m1) * 100"
    label       = "5xx Error Rate (%)"
    return_data = true
  }

  metric_query {
    id = "m1"
    metric {
      metric_name = "RequestCount"
      namespace   = "AWS/ApplicationELB"
      period      = 60
      stat        = "Sum"
      dimensions = {
        LoadBalancer = aws_lb.main.arn_suffix
      }
    }
  }

  metric_query {
    id = "m2"
    metric {
      metric_name = "HTTPCode_ELB_5XX_Count"
      namespace   = "AWS/ApplicationELB"
      period      = 60
      stat        = "Sum"
      dimensions = {
        LoadBalancer = aws_lb.main.arn_suffix
      }
    }
  }

  alarm_actions = local.alarm_actions
  ok_actions    = local.alarm_actions
  tags          = { Name = "${var.project}-5xx-alarm" }
}

# 3. ALB P99 response latency > 3 seconds
resource "aws_cloudwatch_metric_alarm" "alb_latency" {
  alarm_name          = "${var.project}-alb-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  extended_statistic  = "p99"
  threshold           = 3
  alarm_description   = "ALB P99 latency > 3 seconds"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
  }

  alarm_actions = local.alarm_actions
  ok_actions    = local.alarm_actions
  tags          = { Name = "${var.project}-latency-alarm" }
}

# 4. RDS CPU utilisation > 80%
resource "aws_cloudwatch_metric_alarm" "rds_cpu_high" {
  alarm_name          = "${var.project}-rds-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "RDS CPU utilization > 80%"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgres.identifier
  }

  alarm_actions = local.alarm_actions
  ok_actions    = local.alarm_actions
  tags          = { Name = "${var.project}-rds-cpu-alarm" }
}
