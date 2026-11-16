# Infrastructure (AWS, Terraform)

Provisions a minimal, cost-bounded serving stack for the classifier:

| Resource | Purpose |
|---|---|
| ECR repository (scan-on-push, keep last 10 images) | stores the Docker inference image |
| S3 bucket (versioned, encrypted, no public access) | MLflow artifacts and model checkpoints |
| IAM role | lets App Runner pull from ECR |
| App Runner service + autoscaling (1 to `max_instances`) | HTTPS endpoint with `/health` checks |

```bash
cd infra/terraform
terraform init
terraform plan                                  # review; nothing is created yet
terraform apply                                 # 1) registry + bucket
# push the image to the printed ECR URL, then:
terraform apply -var deploy_service=true -var image_tag=<git-sha>
terraform destroy                               # tear everything down
```

> **Cost note:** App Runner bills while the service is provisioned (~$0.064/vCPU-hour active).
> The repo's CI only runs `terraform fmt -check` and `terraform validate`; it never applies.
