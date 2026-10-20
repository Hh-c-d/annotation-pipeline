# YOLO Dataset Annotation Pipeline
From raw images to training-ready datasets: guidelines, tooling, QC reports.

> Status: project scaffold.

## What You Get
- 标注完成的数据集（YOLO / COCO 格式）
- 标注规范文档
- 数据质检报告

## Workflow
原始图 → 标注（CVAT / Label Studio）→ 格式转换 → 质检 → 交付训练就绪的数据集

## Annotation Guidelines
- 框贴边规则：[待填]
- 遮挡处理：[待填]
- 截断处理：[待填]

## Format Conversion
COCO ↔ YOLO ↔ VOC 互转脚本：[待脚本完成后补充用法]

## QC Report Sample
类别分布、框尺寸分布、空标注 / 越界框检查：[待生成样例报告后放截图]

## Tools
Python · CVAT / Label Studio · pandas · matplotlib
