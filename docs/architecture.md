# Architecture

Training ส่งออก checkpoint/ONNX พร้อม contract → controller สร้าง observations และ targets → Unitree SDK2/DDS → MuJoCo bridge → G1 physics → LowState กลับ controller

แยก training, simulator, controller และ evaluator เป็นคนละหน้าที่ Controller รับข้อมูลจาก DDS และ internal command/state เท่านั้น ส่วน simulator ground truth ใช้สำหรับ evaluation โดยไม่ย้อนเข้า actor เว้นแต่ contract มี estimator ที่ใช้ได้จริง

Transport หลักคือ `unitree_hg` LowCmd/LowState บน `rt/lowcmd` และ `rt/lowstate` ไม่ใช่ high-level Move RPC

ดู [แผนหลัก](implementation_plan.md) สำหรับ contract, timing, FSM และข้อจำกัด sim-to-real
