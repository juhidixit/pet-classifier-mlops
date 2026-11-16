variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Name prefix for all resources."
  type        = string
  default     = "pet-classifier"
}

variable "image_tag" {
  description = "Tag of the inference image in ECR to deploy (e.g. the git SHA pushed by CI)."
  type        = string
  default     = "latest"
}

variable "deploy_service" {
  description = "Create the App Runner service. Set false for a first apply, push an image, then true."
  type        = bool
  default     = false
}

variable "cpu" {
  description = "App Runner vCPU (1024 = 1 vCPU)."
  type        = string
  default     = "1024"
}

variable "memory" {
  description = "App Runner memory in MB."
  type        = string
  default     = "2048"
}

variable "max_instances" {
  description = "Autoscaling ceiling; keeps worst-case cost bounded."
  type        = number
  default     = 2

  validation {
    condition     = var.max_instances >= 1 && var.max_instances <= 10
    error_message = "max_instances must be between 1 and 10."
  }
}
