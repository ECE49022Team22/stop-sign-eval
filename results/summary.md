| Detector | Signs found (real) | False alarms (real) | Accuracy | Smallest reliable sign (≥90%) | Time/frame* |
|---|---|---|---|---|---|
| YOLO26n (pretrained) | 96/97 = 99% | 1/100 = 1% | 99% | 28 px | 60 ms |
| YOLO11n (pretrained) | 95/97 = 98% | 3/100 = 3% | 97% | 28 px | 64 ms |
| Rules (team code, as-is) | 47/97 = 48% | 1/100 = 1% | 74% | 128 px | 2 ms |
| Rules, min area lowered to 150 | 56/97 = 58% | 12/100 = 12% | 73% | 128 px | 2 ms |

Sign width → distance (camera assumed 640 px wide, 70° horizontal FOV; f ≈ 457 px):

| Sign | width at 5 m | 8 m | 3 m |
|---|---|---|---|
| 12" (0.30 m) mini sign | 28 px | 17 px | 46 px |
| 18" (0.46 m) | 42 px | 26 px | 70 px |
| 24" (0.61 m) | 56 px | 35 px | 93 px |
| 30" (0.76 m) standard road sign | 70 px | 44 px | 116 px |

Hard negatives (red look-alikes, 60 each) — share that were called a stop sign:

| Detector | no_entry_disc | red_disc | red_triangle |
|---|---|---|---|
| YOLO26n (pretrained) | 68% | 65% | 0% |
| YOLO26n (pretrained), confidence ≥ 0.5 | 38% | 27% | 0% |
| YOLO11n (pretrained) | 90% | 65% | 0% |
| YOLO11n (pretrained), confidence ≥ 0.5 | 67% | 38% | 0% |
| Rules (team code, as-is) | 93% | 93% | 0% |
| Rules, min area lowered to 150 | 93% | 93% | 0% |

Conditions (detection rate, 40 images each):

| 40 px sign | frontal | angled | rotated | dark | overexposed | motion_blur | occluded |
|---|---|---|---|---|---|---|---|
| YOLO26n (pretrained) | 100% | 100% | 95% | 98% | 100% | 72% | 32% |
| YOLO11n (pretrained) | 100% | 95% | 98% | 100% | 100% | 50% | 20% |
| Rules (team code, as-is) | 80% | 0% | 92% | 8% | 0% | 80% | 0% |
| Rules, min area lowered to 150 | 80% | 0% | 92% | 10% | 0% | 80% | 0% |

| 80 px sign | frontal | angled | rotated | dark | overexposed | motion_blur | occluded |
|---|---|---|---|---|---|---|---|
| YOLO26n (pretrained) | 100% | 100% | 100% | 100% | 100% | 98% | 100% |
| YOLO11n (pretrained) | 100% | 100% | 100% | 100% | 100% | 92% | 78% |
| Rules (team code, as-is) | 82% | 0% | 88% | 12% | 0% | 95% | 0% |
| Rules, min area lowered to 150 | 82% | 0% | 88% | 12% | 0% | 95% | 0% |

YOLO confidence threshold sweep (real photos):

| Detector | threshold | found | false alarms |
|---|---|---|---|
| YOLO26n (pretrained) | 0.1 | 99% | 6% |
| YOLO26n (pretrained) | 0.25 | 99% | 1% |
| YOLO26n (pretrained) | 0.4 | 98% | 1% |
| YOLO26n (pretrained) | 0.5 | 96% | 0% |
| YOLO26n (pretrained) | 0.6 | 96% | 0% |
| YOLO26n (pretrained) | 0.7 | 93% | 0% |
| YOLO11n (pretrained) | 0.1 | 100% | 10% |
| YOLO11n (pretrained) | 0.25 | 98% | 3% |
| YOLO11n (pretrained) | 0.4 | 97% | 3% |
| YOLO11n (pretrained) | 0.5 | 97% | 2% |
| YOLO11n (pretrained) | 0.6 | 97% | 1% |
| YOLO11n (pretrained) | 0.7 | 96% | 0% |

*Median time per 640-px frame on the 2-vCPU cloud machine used for this test (PyTorch, CPU). Not a Raspberry Pi number.
