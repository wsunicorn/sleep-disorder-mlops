# ── ECS task execution role (pull ECR, write CloudWatch logs) ─────────────────
data "aws_iam_policy_document" "ecs_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_task_execution" {
  name               = "${var.project}-ecs-execution-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role.json
  tags               = { Name = "${var.project}-ecs-execution-role" }
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_policy" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ── ECS task role (runtime permissions for the container) ─────────────────────
resource "aws_iam_role" "ecs_task" {
  name               = "${var.project}-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role.json
  tags               = { Name = "${var.project}-ecs-task-role" }
}

# Allow task to write CloudWatch metrics and logs
resource "aws_iam_role_policy" "ecs_task_cloudwatch" {
  name = "${var.project}-ecs-cloudwatch"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "*"
      }
    ]
  })
}
