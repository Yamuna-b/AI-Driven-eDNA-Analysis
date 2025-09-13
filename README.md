# AI Driven eDNA Analysis

Welcome!  
This project provides code for analyzing environmental DNA (eDNA) data using AI and transformer models.  
**Large data and results files are hosted on S3 and not included in this GitHub repository.**

---

## 1. Prerequisites

- Python 3.8 or later
- [Git](https://git-scm.com/)
- [VS Code](https://code.visualstudio.com/) (recommended)
- Internet connection (to download files from S3)

---

## 2. Clone This Repository

git clone https://github.com/your-username/AI_Driven_eDNA_Analysis.git cd AI_Driven_eDNA_Analysis

---

## 3. Download Required Data and Results

**Download each file via the S3 links below. Place files in the appropriate folders inside your project.  
If a folder doesn’t exist, create it.  
Each link is active for 7 days—contact the maintainer for fresh links if expired.**

### Download Data Files (`data/`):

| File Name           | Download Link               |
| ------------------- | --------------------------- |
| SRR11851935_1.fastq | [Download](https://edna-pipeline-data-yamuna.s3.us-east-1.amazonaws.com/data/SRR11851935_1.fastq?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIF%2FRwjq8HOXHGHIl2ZUCwiEdUVAgmXW6dRg%2BxnoDyvjeAiA4dqA24ygBtcs2ip1zRppqS87EGJi0%2Fj0nvomEIk%2FNOCq5Awg%2BEAAaDDg0MDg2NDYxNDM2OCIMseQuSWQceK%2BraJavKpYDDsVvBxXa9dxogVPPMmtD4BPFiYXU5a6cCKyvyifAZaSk8zIz%2FSqzgcQ%2BPA5M41FEpnuCPLXT8xW2kXQsL52FhCRKsaOcgnapbFwYW6lY8hGOAM8pjXxTmcFvp52jMLKmzEyyn8BdNDm%2Fhfc4iojA1qz6ymKZHWZB4jJ2M728uIVqQXwZ%2FP1gZU39aAaQTdw8JhqJckej%2FwIksHYf0iOLPjCDlPwQdZjD4Q4m0yapOt%2FZT2n4eLREgCJWsLu4awAbEarP7YFtme%2BMH6LJwjosNdO%2ByK5JXLYPVjkY62%2FlUOGHmx78%2FKMJWca7pJAqVnOycqUIPs7XXnFIggAtwTXomD2q1AeUkZqZdf0O5LWb7%2B7M58iWne%2BMGlsVarKLMLpQPPEG%2BP%2B%2F0HsOllgsWEXxczfpk%2Fspl8u%2F3LlMjnrzqqcpZgr0LsmG7HFkt3rnwwBVSQW7TfmrVq4BHTCDQ5FRMHLSEHDOgNOwHkodhFg0diK%2BfOwr9p86hXjNcEHgk52whSHnPzQ0nEL3GoKwJiBTzrD83XeYuDClv5PGBjrfAp2QOpweN4S3PjsHUNWUbW25cWQhhvBv2GF0xHn%2FMLcIbJ%2Fl5epgmigpbgT%2FVq0qGl1lmCht%2BAK55%2F5rJXAbu6Zg6GqO5qhhi7m0ssnD4n3aI1kH4mFOUAYLA%2Bec42LlsGhwyqW5jimdbCX%2FR7zANqPFOkLPAw6hYDADtZhpKt6Z%2BsVIbzPa4LFQBt1qMiD%2BkC3KEb5U5L7iBvDRekf%2FiWZxe3Z%2BSOsV2dTHAzgkH7fby85iIpvJGMu7k0XyqH3GVGEtKgsqNY%2FQcBKaqUCfXvw7LRIz3svafVGCKxq%2Fl8KgfNnyML3dfsXftDunrQrqSqUmGoqPOgaiM%2FdLXE3g68%2F3dF1l8a%2Bcz1BLaC2kRT%2Bvc7FqiUvRnFwA8Y1CfhfcnqeH3Gk9Xd94B0HCEiZTYBzl9bcznSKRuSyP3hAAYuG%2FUpfdX0Zn0thIMN%2FXi54Jh9n0uv6siskHcbWVtkxSTg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA4HR3PY7QMEH6C7YY%2F20250913%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250913T050337Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=d6168ea9b74362022ce6779046d6581d66efa9c00468b6293c6b7c456c7b9215)|
| SRR11851935_2.fastq | [Download](https://edna-pipeline-data-yamuna.s3.us-east-1.amazonaws.com/data/SRR11851935_2.fastq?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIGsEd5p2ZHQHqj6LrQ%2BLdgr26vpKaHSe9pIWWbHb30WvAiAbGKbx1d1%2FwnIJhLPfJbRD%2F9pYfNGXGrJXx9xOlSo82Sq5Awg%2BEAAaDDg0MDg2NDYxNDM2OCIM9lA%2FlM3CeH5Q1Xh6KpYD%2B194egqpzt0dLtnPCUFdIhJ2br%2FqtV8gS8n7PL6zHagSL6CVcj57qra4bIj4OQgcF0OrgXkUGRUOq6a8ec6YKy7ZLtVgw9WGiB0HrasE9K8jOl4lLQD%2BFx2MjAN2Nx%2FdL1fnx%2Ff5KWJuE3hJTqBOrs%2BVAMuwxcyUGaszVDHMyR203OlUf%2Fj1l8PIwSFWy5y52ZK4Iyh2z2OlQp9VieLpaLxpDzU9JKpHWQfGDa8ezpJ77P2J7e3iTOWL6fbxKcrEe0Zalq1oW4%2B22QioK5P%2BTXHyxl%2B%2BtI0%2FBS70%2Bi3DN899hRMtNn9Y2N6ZnaqMxxj2lKfENM0lLE%2B0L0jyduGHhzQXYvvHO0OCVouHvZVs%2FKYhg6Hi1QXke7QX%2BcrT9koBSDasQaqW86cp%2FPJJn5zItCgMmi0in4AHrhfhD6A7LNuHYhoW1TajfbrmJ5dpeT1APVo251SvLA0N5VwOyNayQG7n1NgvyZD7%2FWudfoWfmtkw%2FlxPs8avZVjzkh7pfnAdhk8p31yHPkQ7o68vBlSaWJA09E5QlDClv5PGBjrfAvpxLDcnoQ6DY77LUaLNLxk244jYmPBNRRvuWn63DuYJ3uAbCKd%2FMDfeMKo5OtdrSWDtiUNRdgWm4x2hnwGqbz8GmbBfVvvEMjGXtodcnLsLtWNtmVJ4BR50cZCkxLBTvhxc0JsZ4n6t4NKKGzZNXOCquyHmCIRVFxuRlpNhk20Emi8vbJVWlyLikEUtkwpKVvhliYWfGe%2BnQ0YxYkZlMazAvm9CWlqHAEoQy%2F27Rd4URt0frxEHaCOFZL0pqrdbWJ8gNfc4vyg47EqtO3aZvxeLP8QfL8FunQbbECI0RoEGwwLDCk6jCqYFDo2CWMbz0KNRtqsckBMX6XC1uLRtjcE0x%2FOF5xhRvC1s7ndz7Ct8vNwuoamJAb4LErMOcz0dzJOY0I1f9GYOcxbrYqH4hXSf2J9n%2FZJTKGeoeYQpsiF0I3fYOaITyCkVN6RshESt2LYKV2vs49edW1%2FAFap%2FAA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA4HR3PY7QOJQ6XHII%2F20250913%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250913T044253Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=0720dc52b01aac27260274b77072ea88ae5383c3cc101362d3cbfc049e34ab06) |
| SRR12076396_1.fastq | [Download](https://edna-pipeline-data-yamuna.s3.us-east-1.amazonaws.com/data/SRR12076396_1.fastq?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIF%2FRwjq8HOXHGHIl2ZUCwiEdUVAgmXW6dRg%2BxnoDyvjeAiA4dqA24ygBtcs2ip1zRppqS87EGJi0%2Fj0nvomEIk%2FNOCq5Awg%2BEAAaDDg0MDg2NDYxNDM2OCIMseQuSWQceK%2BraJavKpYDDsVvBxXa9dxogVPPMmtD4BPFiYXU5a6cCKyvyifAZaSk8zIz%2FSqzgcQ%2BPA5M41FEpnuCPLXT8xW2kXQsL52FhCRKsaOcgnapbFwYW6lY8hGOAM8pjXxTmcFvp52jMLKmzEyyn8BdNDm%2Fhfc4iojA1qz6ymKZHWZB4jJ2M728uIVqQXwZ%2FP1gZU39aAaQTdw8JhqJckej%2FwIksHYf0iOLPjCDlPwQdZjD4Q4m0yapOt%2FZT2n4eLREgCJWsLu4awAbEarP7YFtme%2BMH6LJwjosNdO%2ByK5JXLYPVjkY62%2FlUOGHmx78%2FKMJWca7pJAqVnOycqUIPs7XXnFIggAtwTXomD2q1AeUkZqZdf0O5LWb7%2B7M58iWne%2BMGlsVarKLMLpQPPEG%2BP%2B%2F0HsOllgsWEXxczfpk%2Fspl8u%2F3LlMjnrzqqcpZgr0LsmG7HFkt3rnwwBVSQW7TfmrVq4BHTCDQ5FRMHLSEHDOgNOwHkodhFg0diK%2BfOwr9p86hXjNcEHgk52whSHnPzQ0nEL3GoKwJiBTzrD83XeYuDClv5PGBjrfAp2QOpweN4S3PjsHUNWUbW25cWQhhvBv2GF0xHn%2FMLcIbJ%2Fl5epgmigpbgT%2FVq0qGl1lmCht%2BAK55%2F5rJXAbu6Zg6GqO5qhhi7m0ssnD4n3aI1kH4mFOUAYLA%2Bec42LlsGhwyqW5jimdbCX%2FR7zANqPFOkLPAw6hYDADtZhpKt6Z%2BsVIbzPa4LFQBt1qMiD%2BkC3KEb5U5L7iBvDRekf%2FiWZxe3Z%2BSOsV2dTHAzgkH7fby85iIpvJGMu7k0XyqH3GVGEtKgsqNY%2FQcBKaqUCfXvw7LRIz3svafVGCKxq%2Fl8KgfNnyML3dfsXftDunrQrqSqUmGoqPOgaiM%2FdLXE3g68%2F3dF1l8a%2Bcz1BLaC2kRT%2Bvc7FqiUvRnFwA8Y1CfhfcnqeH3Gk9Xd94B0HCEiZTYBzl9bcznSKRuSyP3hAAYuG%2FUpfdX0Zn0thIMN%2FXi54Jh9n0uv6siskHcbWVtkxSTg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA4HR3PY7QMEH6C7YY%2F20250913%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250913T050552Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=4b491b9be98118433cafef1d7a55da56a04ee1540e169e122c25931b89e2ed7a) |
| SRR12076396_2.fastq | [Download](https://edna-pipeline-data-yamuna.s3.us-east-1.amazonaws.com/data/SRR12076396_2.fastq?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIF%2FRwjq8HOXHGHIl2ZUCwiEdUVAgmXW6dRg%2BxnoDyvjeAiA4dqA24ygBtcs2ip1zRppqS87EGJi0%2Fj0nvomEIk%2FNOCq5Awg%2BEAAaDDg0MDg2NDYxNDM2OCIMseQuSWQceK%2BraJavKpYDDsVvBxXa9dxogVPPMmtD4BPFiYXU5a6cCKyvyifAZaSk8zIz%2FSqzgcQ%2BPA5M41FEpnuCPLXT8xW2kXQsL52FhCRKsaOcgnapbFwYW6lY8hGOAM8pjXxTmcFvp52jMLKmzEyyn8BdNDm%2Fhfc4iojA1qz6ymKZHWZB4jJ2M728uIVqQXwZ%2FP1gZU39aAaQTdw8JhqJckej%2FwIksHYf0iOLPjCDlPwQdZjD4Q4m0yapOt%2FZT2n4eLREgCJWsLu4awAbEarP7YFtme%2BMH6LJwjosNdO%2ByK5JXLYPVjkY62%2FlUOGHmx78%2FKMJWca7pJAqVnOycqUIPs7XXnFIggAtwTXomD2q1AeUkZqZdf0O5LWb7%2B7M58iWne%2BMGlsVarKLMLpQPPEG%2BP%2B%2F0HsOllgsWEXxczfpk%2Fspl8u%2F3LlMjnrzqqcpZgr0LsmG7HFkt3rnwwBVSQW7TfmrVq4BHTCDQ5FRMHLSEHDOgNOwHkodhFg0diK%2BfOwr9p86hXjNcEHgk52whSHnPzQ0nEL3GoKwJiBTzrD83XeYuDClv5PGBjrfAp2QOpweN4S3PjsHUNWUbW25cWQhhvBv2GF0xHn%2FMLcIbJ%2Fl5epgmigpbgT%2FVq0qGl1lmCht%2BAK55%2F5rJXAbu6Zg6GqO5qhhi7m0ssnD4n3aI1kH4mFOUAYLA%2Bec42LlsGhwyqW5jimdbCX%2FR7zANqPFOkLPAw6hYDADtZhpKt6Z%2BsVIbzPa4LFQBt1qMiD%2BkC3KEb5U5L7iBvDRekf%2FiWZxe3Z%2BSOsV2dTHAzgkH7fby85iIpvJGMu7k0XyqH3GVGEtKgsqNY%2FQcBKaqUCfXvw7LRIz3svafVGCKxq%2Fl8KgfNnyML3dfsXftDunrQrqSqUmGoqPOgaiM%2FdLXE3g68%2F3dF1l8a%2Bcz1BLaC2kRT%2Bvc7FqiUvRnFwA8Y1CfhfcnqeH3Gk9Xd94B0HCEiZTYBzl9bcznSKRuSyP3hAAYuG%2FUpfdX0Zn0thIMN%2FXi54Jh9n0uv6siskHcbWVtkxSTg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA4HR3PY7QMEH6C7YY%2F20250913%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250913T050614Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=9fa3d4e3425291a7b2708543ec9aea980ecbe3169ff7b2ff90666b68897a5de0) |

### Download Results Files (`results/`):

| File Name                   | Download Link               |
| --------------------------- | --------------------------- |
| forward_read_embeddings.npy | [Download](https://edna-pipeline-data-yamuna.s3.us-east-1.amazonaws.com/results/forward_read_embeddings.npy?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIF%2FRwjq8HOXHGHIl2ZUCwiEdUVAgmXW6dRg%2BxnoDyvjeAiA4dqA24ygBtcs2ip1zRppqS87EGJi0%2Fj0nvomEIk%2FNOCq5Awg%2BEAAaDDg0MDg2NDYxNDM2OCIMseQuSWQceK%2BraJavKpYDDsVvBxXa9dxogVPPMmtD4BPFiYXU5a6cCKyvyifAZaSk8zIz%2FSqzgcQ%2BPA5M41FEpnuCPLXT8xW2kXQsL52FhCRKsaOcgnapbFwYW6lY8hGOAM8pjXxTmcFvp52jMLKmzEyyn8BdNDm%2Fhfc4iojA1qz6ymKZHWZB4jJ2M728uIVqQXwZ%2FP1gZU39aAaQTdw8JhqJckej%2FwIksHYf0iOLPjCDlPwQdZjD4Q4m0yapOt%2FZT2n4eLREgCJWsLu4awAbEarP7YFtme%2BMH6LJwjosNdO%2ByK5JXLYPVjkY62%2FlUOGHmx78%2FKMJWca7pJAqVnOycqUIPs7XXnFIggAtwTXomD2q1AeUkZqZdf0O5LWb7%2B7M58iWne%2BMGlsVarKLMLpQPPEG%2BP%2B%2F0HsOllgsWEXxczfpk%2Fspl8u%2F3LlMjnrzqqcpZgr0LsmG7HFkt3rnwwBVSQW7TfmrVq4BHTCDQ5FRMHLSEHDOgNOwHkodhFg0diK%2BfOwr9p86hXjNcEHgk52whSHnPzQ0nEL3GoKwJiBTzrD83XeYuDClv5PGBjrfAp2QOpweN4S3PjsHUNWUbW25cWQhhvBv2GF0xHn%2FMLcIbJ%2Fl5epgmigpbgT%2FVq0qGl1lmCht%2BAK55%2F5rJXAbu6Zg6GqO5qhhi7m0ssnD4n3aI1kH4mFOUAYLA%2Bec42LlsGhwyqW5jimdbCX%2FR7zANqPFOkLPAw6hYDADtZhpKt6Z%2BsVIbzPa4LFQBt1qMiD%2BkC3KEb5U5L7iBvDRekf%2FiWZxe3Z%2BSOsV2dTHAzgkH7fby85iIpvJGMu7k0XyqH3GVGEtKgsqNY%2FQcBKaqUCfXvw7LRIz3svafVGCKxq%2Fl8KgfNnyML3dfsXftDunrQrqSqUmGoqPOgaiM%2FdLXE3g68%2F3dF1l8a%2Bcz1BLaC2kRT%2Bvc7FqiUvRnFwA8Y1CfhfcnqeH3Gk9Xd94B0HCEiZTYBzl9bcznSKRuSyP3hAAYuG%2FUpfdX0Zn0thIMN%2FXi54Jh9n0uv6siskHcbWVtkxSTg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA4HR3PY7QMEH6C7YY%2F20250913%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250913T050700Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=abf4d21b40918e11d01afe447d55f4401a0b9dca588185f435212e35da4c619b) |
| reverse_read_embeddings.npy | [Download](https://edna-pipeline-data-yamuna.s3.us-east-1.amazonaws.com/results/reverse_read_embeddings.npy?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIF%2FRwjq8HOXHGHIl2ZUCwiEdUVAgmXW6dRg%2BxnoDyvjeAiA4dqA24ygBtcs2ip1zRppqS87EGJi0%2Fj0nvomEIk%2FNOCq5Awg%2BEAAaDDg0MDg2NDYxNDM2OCIMseQuSWQceK%2BraJavKpYDDsVvBxXa9dxogVPPMmtD4BPFiYXU5a6cCKyvyifAZaSk8zIz%2FSqzgcQ%2BPA5M41FEpnuCPLXT8xW2kXQsL52FhCRKsaOcgnapbFwYW6lY8hGOAM8pjXxTmcFvp52jMLKmzEyyn8BdNDm%2Fhfc4iojA1qz6ymKZHWZB4jJ2M728uIVqQXwZ%2FP1gZU39aAaQTdw8JhqJckej%2FwIksHYf0iOLPjCDlPwQdZjD4Q4m0yapOt%2FZT2n4eLREgCJWsLu4awAbEarP7YFtme%2BMH6LJwjosNdO%2ByK5JXLYPVjkY62%2FlUOGHmx78%2FKMJWca7pJAqVnOycqUIPs7XXnFIggAtwTXomD2q1AeUkZqZdf0O5LWb7%2B7M58iWne%2BMGlsVarKLMLpQPPEG%2BP%2B%2F0HsOllgsWEXxczfpk%2Fspl8u%2F3LlMjnrzqqcpZgr0LsmG7HFkt3rnwwBVSQW7TfmrVq4BHTCDQ5FRMHLSEHDOgNOwHkodhFg0diK%2BfOwr9p86hXjNcEHgk52whSHnPzQ0nEL3GoKwJiBTzrD83XeYuDClv5PGBjrfAp2QOpweN4S3PjsHUNWUbW25cWQhhvBv2GF0xHn%2FMLcIbJ%2Fl5epgmigpbgT%2FVq0qGl1lmCht%2BAK55%2F5rJXAbu6Zg6GqO5qhhi7m0ssnD4n3aI1kH4mFOUAYLA%2Bec42LlsGhwyqW5jimdbCX%2FR7zANqPFOkLPAw6hYDADtZhpKt6Z%2BsVIbzPa4LFQBt1qMiD%2BkC3KEb5U5L7iBvDRekf%2FiWZxe3Z%2BSOsV2dTHAzgkH7fby85iIpvJGMu7k0XyqH3GVGEtKgsqNY%2FQcBKaqUCfXvw7LRIz3svafVGCKxq%2Fl8KgfNnyML3dfsXftDunrQrqSqUmGoqPOgaiM%2FdLXE3g68%2F3dF1l8a%2Bcz1BLaC2kRT%2Bvc7FqiUvRnFwA8Y1CfhfcnqeH3Gk9Xd94B0HCEiZTYBzl9bcznSKRuSyP3hAAYuG%2FUpfdX0Zn0thIMN%2FXi54Jh9n0uv6siskHcbWVtkxSTg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA4HR3PY7QMEH6C7YY%2F20250913%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250913T050728Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=b0d1a1e4c87301df623761e7f4eb79305e8322c328ddbddb8c1dc435a8c1d186) |

---

## 4. Set Up Python Environment

python -m venv venv

Activate:
On Windows:
venv\Scripts\activate

On macOS/Linux:
source venv/bin/activate

If requirements.txt is provided:
pip install -r requirements.txt

---

## 5. Run the Analysis

- Open any notebooks or scripts in `scripts/` using VS Code.
- Ensure you’ve placed data/results files in the right folders.
- Follow comments or pipeline documentation for specific usage.

---

## 6. Notes & Best Practices

- **DO NOT add or push large files to GitHub!**
  - The `.gitignore` protects the repository from accidental uploads.
- If download links expire, **request new S3 presigned URLs** from the project owner.
- If you create new results or datasets, upload to S3 and update README links (never share via git).

---

## 7. Contact

For questions, fresh data/result links, or getting started help:  
**Contact:** [your-email@example.com]  
(or open a GitHub issue)

---

### Quick Recap for New Users

1. Clone repo
2. Download from S3 links in README
3. Place files in `data/` and `results/`
4. Setup Python environment
5. Run pipeline or notebook

---

\* This project follows best practices:  
**GitHub for code, S3 for data!**
