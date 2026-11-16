output "ecr_repository_url" {
  description = "Push the inference image here."
  value       = aws_ecr_repository.inference.repository_url
}

output "artifact_bucket" {
  description = "S3 bucket for MLflow artifacts and model checkpoints."
  value       = aws_s3_bucket.artifacts.bucket
}

output "service_url" {
  description = "Public HTTPS URL of the inference API (once deploy_service = true)."
  value       = var.deploy_service ? "https://${aws_apprunner_service.inference[0].service_url}" : null
}
