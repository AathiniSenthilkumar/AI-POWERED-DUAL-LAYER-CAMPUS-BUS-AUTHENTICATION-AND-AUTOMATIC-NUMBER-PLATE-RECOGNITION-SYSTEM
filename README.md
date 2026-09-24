# 🚍 AI-Powered Dual-Layer Campus Bus Authentication System

An AI-based campus bus verification system that combines **Automatic Number Plate Recognition (ANPR)** and **campus branding verification** to identify and authenticate authorized college buses.

The system uses **YOLOv8** for license plate detection, **PaddleOCR 3.7** for number plate recognition, and **OpenCV** for image preprocessing. A dual-layer verification approach is used to improve the reliability of campus bus authentication.

---

## 📌 Project Overview

Campus security personnel often verify buses manually by visually checking vehicle numbers and campus branding. This process can be time-consuming and may lead to human errors.

This project proposes an automated computer vision system that verifies a bus using two independent visual layers:

1. **License Plate Verification**
2. **Campus Branding Verification**

A bus is classified as **AUTHORIZED** only when the required verification conditions are satisfied.

---

## 🎯 Objectives

- Detect vehicle license plates automatically using YOLOv8.
- Recognize license plate characters using PaddleOCR 3.7.
- Validate Indian vehicle registration numbers.
- Verify authorized campus branding such as "St. Joseph's".
- Implement dual-layer bus authentication.
- Reduce manual verification at campus entry points.
- Maintain authorized vehicle information using a local database.
- Provide a user-friendly interface for bus verification.

---

## 🧠 System Architecture

```text
              Bus Image / Camera
                      │
                      ▼
                  YOLOv8
                      │
             License Plate Detection
                      │
                      ▼
             Image Preprocessing
                      │
                      ▼
               PaddleOCR 3.7
                      │
             Plate Number Recognition
                      │
                      ▼
              Plate Validation
                      │
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
 Plate Verification         Branding Verification
        │                     ("St. Joseph's")
        │                           │
        └─────────────┬─────────────┘
                      ▼
                Decision Engine
                      │
             ┌────────┴────────┐
             ▼                 ▼
        AUTHORIZED         UNAUTHORIZED
